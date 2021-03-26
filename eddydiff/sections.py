import dcpy
import matplotlib.pyplot as plt
import numpy as np

import xarray as xr


def to_netcdf(infile, outfile, transect_name):
    """
    Converts CTD χpod .mat file to netCDF.

    Parameters
    ----------
    infile : str
        Input .mat file name
    outfile : str
        Output netCDF file name
    transect_name: str
        Transect name, added to attrs.

    Returns
    -------
    None
    """
    import seawater as sw
    from scipy.io import loadmat

    mat = loadmat(infile, squeeze_me=True)["grd"]

    ctd = xr.Dataset()

    dims = ["cast", "pres"]
    coords = {
        # "cast": np.arange(300) + 1,
        "pres": mat["P"]
        .item()
        .squeeze()
        .astype(np.float32),
    }

    for varname in ["chi", "eps", "Jq", "KT", "N2", "dTdz", "TPvar", "t", "s"]:
        if varname in mat.dtype.names:
            ctd[varname] = xr.DataArray(mat[varname].item(), dims=dims, coords=coords)

    for varname in ["lat", "lon", "dnum", "dist", "time"]:
        if varname in mat.dtype.names:
            ctd[varname] = xr.DataArray(mat[varname].item(), dims=["cast"])

    renamer = {"t": "T", "s": "S", "dnum": "time"}
    ctd = ctd.rename({k: v for k, v in renamer.items() if k in ctd})

    ctd["time"].values = dcpy.util.datenum2datetime(ctd["time"].values)

    ctd["sigma_0"] = xr.DataArray(
        sw.pden(ctd.S, ctd["T"], ctd["pres"], 0), dims=dims, coords=coords
    )
    ctd["sigma_1"] = xr.DataArray(
        sw.pden(ctd.S, ctd["T"], ctd["pres"], 1000), dims=dims, coords=coords
    )
    ctd["sigma_2"] = xr.DataArray(
        sw.pden(ctd.S, ctd["T"], ctd["pres"], 2000), dims=dims, coords=coords
    )
    ctd["sigma_4"] = xr.DataArray(
        sw.pden(ctd.S, ctd["T"], ctd["pres"], 4000), dims=dims, coords=coords
    )
    ctd["rho"] = xr.DataArray(
        sw.dens(ctd.S, ctd["T"], ctd["pres"]), dims=dims, coords=coords
    )

    ctd = ctd.rename({"pres": "P"})
    # code expects P as pressure index vector; pres as actual 2D pressure
    ctd["pres"] = xr.broadcast(ctd["cast"], ctd["P"])[1]

    ctd = ctd.set_coords(
        [v for v in ["time", "lon", "lat", "dist", "pres"] if v in ctd]
    )

    # reset co-ords to match ECCO/argo fields.
    if "lon" in ctd:
        ctd["lon"][ctd.lon <= 0] += 360
        ctd = ctd.drop([1, 2], dim="cast")

    # ctd = ctd.drop(np.arange(250, 301), dim="cast")

    # fill in NaNs
    if "lon" in ctd:
        ctd["lon"] = ctd.lon.interpolate_na(dim="cast")
    if "lat" in ctd:
        ctd["lat"] = ctd.lat.interpolate_na(dim="cast")

    # add in bathymetry
    if "lon" in ctd and "lat" in ctd:
        etopo = xr.open_dataset("/home/deepak/datasets/ETOPO2v2g_f4.nc4").drop(
            180.0, "x"
        )
        etopo = dcpy.oceans.dataset_center_pacific(
            etopo.rename({"x": "lon", "y": "lat"})
        )
        ctd["bathy"] = etopo.interp(lon=ctd.lon, lat=ctd.lat).z

    ctd.attrs["name"] = "ctd merged dataset"
    ctd.attrs["transect_name"] = "ctd"
    ctd.to_netcdf(outfile)


def add_ancillary_variables(ds, pref=0):
    """ Adds ancillary variables."""

    salt, temp, pres = (
        ds.cf["sea_water_salinity"],
        ds.cf["sea_water_temperature"],
        ds.cf["sea_water_pressure"],
    )

    if "theta" not in ds:
        ds["theta"] = dcpy.eos.ptmp(
            salt,
            temp,
            pres,
            pr=pref,
        )
    ds["theta"].attrs.update(long_name="$θ$")

    if "pden" not in ds:
        ds["pden"] = dcpy.eos.pden(salt, temp, pres, pr=pref)
    ds["pden"].attrs.update(long_name="$ρ$")

    if "neutral_density" not in ds.cf:
        ds["gamma_n"] = dcpy.oceans.neutral_density(ds)

    if "dTdz" in ds:
        ds = ds.rename_vars({"dTdz": "Tz"})
    if "Tz" not in ds:
        ds["Tz"] = -1 * ds.theta.interpolate_na("depth").differentiate("depth")
    ds["Tz"].attrs["long_name"] = "$θ_z$"

    if "N2" not in ds:
        ds["N2"] = (
            9.81 / 1025 * ds.gamma_n.interpolate_na("depth").differentiate("depth")
        )
    ds["N2"].attrs["long_name"] = "$N²$"

    Tz_mask = np.abs(ds.Tz) > 1e-3
    N2_mask = (ds.N2) > 1e-6

    ds["chi_masked"] = ds.chi.where(Tz_mask)

    if "eps" in ds:
        ds["Krho"] = (0.2 * ds.eps / ds.N2).where(N2_mask)
        ds["Krho"].attrs["long_name"] = "$K_ρ$"
        ds["Krho"].attrs["units"] = "m²/s"

        ds["KrhoTz"] = ds.Krho * ds.Tz.where(Tz_mask)
        ds["KrhoTz"].attrs["long_name"] = "$K_ρ θ_z$"

        # ds["KrhoN2"] = ds.Krho * np.sqrt(ds.N2)
        # ds["KrhoN2"].attrs["long_name"] = "$K_ρ N$"

        ds["eps_chi"] = (
            1 / 2 * ds.chi * ds.N2.where(N2_mask) / 0.2 / (ds.Tz.where(Tz_mask) ** 2)
        )
        ds["eps_chi"].attrs["long_name"] = "$ε_χ$"
        ds["eps_chi"].attrs["units"] = "W/kg"

    ds["Kt"] = (ds.chi / 2 / ds.Tz ** 2).where(Tz_mask)
    ds["Kt"].attrs["long_name"] = "$K_T$"
    ds["Kt"].attrs["units"] = "m²/s"

    ds["KtTz"] = ds.Kt * ds.Tz
    ds["KtTz"].attrs["long_name"] = "$K_t θ_z$"

    return ds


def bin_average_vertical(ds, stdname, bins):
    """ Bin averages in the vertical."""

    grouped = ds.reset_coords().cf.groupby_bins(stdname, bins=bins)
    chidens = grouped.mean()
    for var in chidens.variables:
        if var in ds.variables:
            chidens[var].attrs = ds[var].attrs

    var = ds.cf["neutral_density"]
    groupvar = f"{var.name}_bins"
    chidens[groupvar].attrs = var.attrs
    # chidens[groupvar].attrs["positive"] = "down"
    chidens[groupvar].attrs["axis"] = "Z"
    chidens = chidens.set_coords("pres")
    chidens["pres"].attrs["positive"] = "down"

    chidens.eps.attrs.update(long_name="$⟨ε⟩$")
    chidens.chi.attrs.update(long_name="$⟨χ⟩$")

    chidens["dTdz_m"] = -1 * grouped.apply(fit1D, var="theta", dim="depth")
    chidens.dTdz_m.attrs.update(dict(name="$∂_z θ_m$", units="°C/m"))

    chidens["N2_m"] = 9.81 / 1030 * grouped.apply(fit1D, var="pden", dim="depth")
    chidens.N2_m.attrs.update(dict(name="$∂_zb_m$", units="s$^{-2}$"))

    chidens["Krho_m"] = 0.2 * chidens.eps / chidens.N2_m
    chidens.Krho_m.attrs.update(dict(long_name="$K_ρ^m$", units="m²/s"))

    chidens["Kt_m"] = chidens.chi / 2 / chidens.dTdz_m ** 2
    chidens.Kt_m.attrs.update(dict(long_name="$K_T^m$", units="m²/s"))

    chidens.coords["num_obs"] = ds.chi.groupby_bins(ds.cf[stdname], bins=bins).count()
    chidens.num_obs.attrs = {"long_name": "count(χ) in bins"}

    chidens = chidens.cf.guess_coord_axis()
    # iso_slope = grouped.apply(fit2D)
    # chidens["dTiso"] = np.hypot(iso_slope.x, iso_slope.y)

    return chidens


def fit1D(group, var, dim="depth", debug=False):
    """
    Expects a bunch of profiles at differ lat, lons.
    Calculates mean profile and then takes linear fit to estimate gradient.
    """

    ds = group.unstack()
    if "depth" in ds.dims:
        ds["depth"].attrs["axis"] = "Z"
    group_over = set(ds.dims) - set(ds.cf.axes["Z"])
    stacked = ds[var].stack(latlon=group_over).drop("latlon")

    # move to a relative-depth reference frame for proper averaging
    stacked["z0"] = stacked[dim] - stacked[dim].where(stacked.notnull()).mean(dim)
    stacked -= stacked.mean(dim)

    if debug:
        f, ax = plt.subplots(2, 2, constrained_layout=True)
        kwargs = dict(add_legend=False, lw=0.5)
        stacked.plot.line(y=dim, ax=ax[0, 0], **kwargs)
        stacked.plot.line(y="z0", **kwargs, ax=ax[0, 1])

    binned = stacked.groupby_bins("z0", bins=np.arange(-10, 200, 2))
    count = binned.count()
    min_count = 300  # count.median() / 1.25
    mean_profile = binned.mean().where(count > min_count)
    mean_profile["z0_bins"] = mean_profile.indexes["z0_bins"].mid

    poly = (mean_profile).polyfit("z0_bins", deg=1)
    slope = poly.polyfit_coefficients.sel(degree=1, drop=True)

    if debug:
        mean_profile.plot(ax=ax[1, 0])
        xr.polyval(
            mean_profile.z0_bins.sel(z0_bins=slice(50)), poly.polyfit_coefficients
        ).plot(ax=ax[1, 0])
        ax[1, 0].text(x=0.1, y=0.8, s=str(slope.item()), transform=ax[1, 0].transAxes)

    return slope


def fit2D(group, debug=False):
    from scipy.interpolate import RectBivariateSpline

    pden = (
        group.unstack()
        .mean("depth")
        .theta.sortby("latitude")
        # .isel(longitude=slice(1, -1), latitude=slice(1, -1))
    )
    pden = pden.cf.guess_coord_axis()
    x, y = dcpy.util.latlon_to_xy(pden.cf["latitude"], pden.cf["longitude"])
    x -= x.min()
    y -= y.min()
    pden.coords["x"] = x.mean("latitude")
    pden.coords["y"] = y.mean("longitude")
    if debug:
        plt.figure()
        pden.cf.plot()

    spl = RectBivariateSpline(
        x=pden.cf["X"].data, y=pden.cf["Y"].data, z=pden.data, kx=1, ky=1, s=1
    )
    if debug:
        print(spl.get_residual())

    fit = pden.copy(data=spl(pden.cf["X"], pden.cf["Y"]))
    test = xr.concat([pden, fit], dim=xr.Variable("kind", ["actual", "fit"]))

    if debug:
        plt.figure()
        fit.cf.plot()

        plt.figure()
        test.isel(longitude=5).plot(hue="kind")

        plt.figure()
        test.isel(latitude=6).plot(hue="kind")

    slope = xr.Dataset()
    slope["x"] = (
        fit.polyfit("longitude", deg=1, use_coordinate="x")
        .sel(degree=1, drop=True)
        .mean()
        .to_array()
    )
    slope["y"] = (
        fit.polyfit("latitude", deg=1, use_coordinate="y")
        .sel(degree=1, drop=True)
        .mean()
        .to_array()
    )
    slope = slope.squeeze().drop_vars("variable")  # .expand_dims(pden_bins=[label])
    return slope


def plot_var_prod_diss(chidens, prefix="", ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    (chidens.chi / 2).cf.plot.step(
        y="Z", xscale="log", color="r", lw=2, label="$⟨χ⟩/2$", **kwargs
    )
    (chidens.KtTz * chidens.dTdz_m).cf.plot.step(
        color="k", label="$⟨K_T θ_z⟩ ∂_zθ_m$", **kwargs
    )
    ax.grid(True, which="both", lw=0.5)
    plt.legend()
    plt.xlabel("Variance production or dissipation [°C²/s]")
    plt.gcf().set_size_inches((4, 5))


def choose_bins(gamma, depth_range):
    mean_over = set(gamma.dims) - set(gamma.cf.axes["Z"])
    bins = gamma.mean(mean_over).cf.interp(Z=depth_range)
    bins[gamma.cf.axes["Z"][0]].attrs["axis"] = "Z"
    return bins.cf.dropna("Z").data
