import dcpy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


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
    from scipy.io import loadmat
    import seawater as sw

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


def add_ancillary_variables(ds):
    """ Adds ancillary variables."""

    ds["theta"] = dcpy.eos.ptmp(ds.salt, ds.temp, ds.pres, pr=1000)
    ds["theta"].attrs.update(long_name="$θ$")

    ds["pden"] = dcpy.eos.pden(ds.salt, ds.temp, ds.pres, pr=1000)
    ds["pden"].attrs.update(long_name="$ρ$")

    ds["gamma_n"] = dcpy.oceans.neutral_density(ds)

    ds["Tz"] = -1 * ds.theta.interpolate_na("depth").differentiate("depth")
    ds["Tz"].attrs["long_name"] = "$θ_z$"

    ds["N2"] = 9.81 / 1025 * ds.gamma_n.interpolate_na("depth").differentiate("depth")
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

    grouped = ds.cf.groupby_bins(stdname, bins=bins)
    chidens = grouped.mean()

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

    # iso_slope = grouped.apply(fit2D)
    # chidens["dTiso"] = np.hypot(iso_slope.x, iso_slope.y)

    return chidens


def fit1D(group, var, dim="depth", debug=False):
    """
    Expects a bunch of profiles at differ lat, lons.
    Calculates mean profile and then takes linear fit to estimate gradient.
    """
    ds = group.unstack()
    stacked = ds[var].stack(latlon=("latitude", "longitude")).drop("latlon")

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
