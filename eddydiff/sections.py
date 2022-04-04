import os

import cf_xarray as cfxr  # noqa
import dcpy
import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flox.xarray import xarray_reduce

import xarray as xr

from .eddydiff import intervals_to_bounds
from .utils import get_hashes

criteria = {
    "sea_water_salinity": {
        "standard_name": "sea_water_salinity|sea_water_practical_salinity"
    }
}


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


def add_ancillary_variables(ds):
    """Adds ancillary variables."""

    with cfxr.set_options(custom_criteria=criteria):
        salt, temp, pres = (
            ds.cf["sea_water_salinity"],
            ds.cf["sea_water_temperature"],
            ds.cf["sea_water_pressure"],
        )

    # if "theta" not in ds:
    #     ds["theta"] = dcpy.eos.ptmp(
    #         salt,
    #         temp,
    #         pres,
    #         pr=pref,
    #     )
    # ds["theta"].attrs.update(long_name="$θ$")

    # if "pden" not in ds:
    #    ds["pden"] = dcpy.eos.pden(salt, temp, pres, pr=pref)
    # ds["pden"].attrs.update(long_name="$ρ$")

    ds["SA"] = gsw.SA_from_SP(
        salt,
        pres,
        ds.cf["longitude"],
        ds.cf["latitude"],
    )
    ds.SA.attrs["standard_name"] = "sea_water_absolute_salinity"
    ds.SA.attrs["long_name"] = "$S_A$"
    ds.SA.attrs["units"] = "g/kg"

    ds["CT"] = gsw.CT_from_t(salt, temp, pres)
    ds.CT.attrs = {
        "standard_name": "sea_water_conservative_temperature",
        "long_name": "$Θ$",
        "units": "degC",
    }
    ds["Tu"] = dcpy.oceans.turner_angle(ds)

    if "neutral_density" not in ds.cf:
        ds["gamma_n"] = dcpy.oceans.neutral_density(ds)

    if "dTdz" in ds:
        ds = ds.rename_vars({"dTdz": "Tz_orig"})

    Z = "sea_water_pressure"
    if "Tz" not in ds:
        ds["Tz"] = -1 * ds.CT.cf.interpolate_na(Z).cf.differentiate(Z)
    ds["Tz"].attrs["long_name"] = "$θ_z$"
    ds["Tz"].attrs["units"] = "degC/m"

    def take_(arr, slicer, axis):
        idxr = [slice(None)] * arr.ndim
        idxr[axis] = slicer
        return arr[tuple(idxr)]

    if "N2" not in ds:
        zaxis = ds.SA.cf.get_axis_num("Z")
        N2, pmid = gsw.Nsquared(
            ds.SA,
            ds.CT,
            pres.broadcast_like(ds.SA),
            ds.cf["latitude"].broadcast_like(ds.SA),
            axis=zaxis,
        )
        ds["N2"] = xr.DataArray(
            (take_(N2, slice(-1), zaxis) + take_(N2, slice(1, None), zaxis)) / 2,
            dims=ds.SA.dims,
            coords={pres.name: (pres.dims, pres.data[1:-1], pres.attrs)},
        )
    ds["N2"].attrs["long_name"] = "$N²$"
    ds["N2"].attrs["units"] = "s-2"

    ds["τ1"] = gsw.spiciness1(ds.SA, ds.CT)
    ds.τ1.attrs = {"standard_name": "spiciness", "long_name": "$τ_1$"}

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

    ds["Kt"] = (ds.chi / 2 / ds.Tz**2).where(Tz_mask)
    ds["Kt"].attrs["long_name"] = "$K_T$"
    ds["Kt"].attrs["units"] = "m²/s"

    ds["KtTz"] = ds.Kt * ds.Tz
    ds["KtTz"].attrs["long_name"] = "$K_t θ_z$"

    return ds


def compute_mean_ci(data, dof=None, alpha=0.05):
    from scipy.stats import distributions

    assert data.ndim == 1
    data = data[~np.isnan(data)]
    if dof is None:
        dof = len(data)

    lower, upper = distributions.t.ppf((alpha / 2, 1 - alpha / 2), dof)

    mean = np.mean(data, keepdims=True)
    std = np.std(data, keepdims=True)
    lower = mean + lower * std / np.sqrt(dof)
    upper = mean + upper * std / np.sqrt(dof)
    return np.concatenate([lower, mean, upper])


def compute_bootstrapped_mean_ci(array, blocksize):
    from arch.bootstrap import MovingBlockBootstrap
    from numpy.random import RandomState

    rs = RandomState(1234)

    # drop nans, significantly faster to do this than bootstrap with nanmean
    assert array.ndim == 1
    array = array[~np.isnan(array)]

    return np.insert(
        MovingBlockBootstrap(blocksize, array, seed=rs)
        .conf_int(func=np.mean, method="bca")
        .squeeze(),
        1,
        np.mean(array),
    )


def average_density_bin(group, blocksize, skip_fits=False):
    # groupby_bins ends up stacking the pressure coordinate
    # which deletes attrs so we can't use cf-xarray here
    # Z = "sea_water_pressure"

    profiles = group.unstack().drop_vars("time")
    if "pres" in profiles.dims:
        Z = "pres"
    elif "pressure" in profiles.dims:
        Z = "pressure"
    else:
        raise ValueError("Don't know what the pressure dimension is.")

    # flatten for bootstrap
    flattened = (
        profiles[["chi", "eps", "KtTz"]].reset_coords(drop=True).stack(flat=[...])
    )

    ci = xr.apply_ufunc(
        compute_bootstrapped_mean_ci,
        flattened.chunk({"flat": -1}),
        input_core_dims=[["flat"]],
        exclude_dims={"flat"},
        # TODO: configure this
        kwargs={"blocksize": blocksize},
        output_core_dims=[["bound"]],
        dask_gufunc_kwargs=dict(output_sizes={"bound": 3}),
        dask="parallelized",
        output_dtypes=[float],
    ).assign_coords(bound=["lower", "center", "upper"])

    pres = profiles[Z].where(profiles.chi.notnull())
    hm = pres.max(Z) - pres.min(Z)
    hm = hm.where(hm > 1)

    ci["hm"] = xr.apply_ufunc(
        compute_mean_ci,
        hm,
        input_core_dims=[["cast"]],
        exclude_dims={"cast"},
        output_core_dims=[["bound"]],
        dask_gufunc_kwargs=dict(output_sizes={"bound": 3}),
        dask="parallelized",
    )
    ci["hm"].attrs = {
        "long_name": "$h_m$",
        "description": "separation between γ_n surfaces",
        "units": "m",
    }
    # ci["hm"] = ("bound", compute_mean_ci(hm.data, hm.count("latlon")))

    chidens = ci.sel(bound="center")
    bounds = ci.sel(bound=["lower", "upper"])
    delta = bounds.diff("bound").squeeze()

    unit = xr.DataArray([-1, 1], dims="bound", coords={"bound": ["lower", "upper"]})

    Γ = 0.2
    delta["Γ"] = 0.04

    if not skip_fits:
        # reference to mean pressure of obs in this bin
        pref = group[Z].mean().data
        group["theta"] = dcpy.eos.ptmp(group.salt, group.temp, group[Z], pr=pref)
        chidens["theta"] = group.theta.mean()
        chidens["salt"] = group.salt.mean()

        chidens["dTdz_m"] = -1 * fit1D(group, var="theta", dim=Z)
        chidens.dTdz_m.attrs.update(
            dict(
                name="$∂_z θ_m$",
                units="°C/m",
                description="vertical gradient of potential temperature θ with respect to depth",
            )
        )

        chidens["N2_m"] = 9.81 / 1030 * fit1D(group, var="gamma_n", dim=Z)
        chidens.N2_m.attrs.update(
            dict(
                name="$∂_zb_m$",
                units="s$^{-2}$",
                description="vertical gradient of neutral density γ_n with respect to depth",
            )
        )

        chidens["Krho_m"] = Γ * chidens.eps / chidens.N2_m
        chidens.Krho_m.attrs.update(dict(long_name="$K_ρ^m$", units="m²/s"))
        delta["Krho_m"] = chidens.Krho_m * np.sqrt(
            (delta.Γ / Γ) ** 2
            + (delta.eps / chidens.eps) ** 2
            + (delta.hm / chidens.hm) ** 2
        )

        chidens["Kt_m"] = 1 / 2 * chidens.chi / chidens.dTdz_m
        delta["Kt_m"] = chidens.Kt_m * np.sqrt(
            (delta.chi / chidens.chi) ** 2 + (delta.hm / chidens.hm) ** 2
        )
        chidens.Kt_m.attrs.update(dict(long_name="$K_T^m$", units="m²/s"))

        chidens["KρTz2"] = chidens.Krho_m * chidens.dTdz_m**2
        delta["KρTz2"] = chidens.KρTz2 * np.sqrt(
            (delta.Krho_m / chidens.Krho_m) ** 2 + 2 * (delta.hm / chidens.hm) ** 2
        )
        chidens.KρTz2.attrs = {"long_name": "$K_ρ ∂_zθ_m^2$"}

        chidens["KtTzTz"] = chidens.KtTz * chidens.dTdz_m
        delta["KtTzTz"] = chidens.KtTzTz * np.sqrt(
            (delta.KtTz / chidens.KtTz) ** 2 + (delta.hm / chidens.hm) ** 2
        )
        chidens.KtTzTz.attrs = {"long_name": "$⟨K_T θ_z⟩ ∂_zθ_m$"}

        chidens["residual"] = chidens.chi / 2 - chidens.Krho_m * chidens.dTdz_m**2
        delta["residual"] = chidens.residual * np.sqrt(
            (delta.chi / chidens.chi) ** 2 + (delta.KρTz2 / chidens.KρTz2) ** 2
        )

    for var in ["Krho_m", "Kt_m", "KρTz2", "KtTzTz", "residual"]:
        bounds[var] = chidens[var] + unit * delta[var]

    # Keep these as data_vars otherwise triggers compute at combine-stage
    delta["pres"] = chidens.hm / 2

    chidens["num_obs"] = profiles.chi.count().data
    chidens["pres"] = pres.mean().data
    chidens.pres.attrs = {
        "units": "dbar",
        "standard_name": "sea_water_pressure",
        "bounds": "pres_err",
    }
    chidens["pres_err"] = chidens.pres + unit * delta.pres

    chidens = chidens.update({f"{name}_err": var for name, var in bounds.items()})
    chidens = chidens.update({f"δ{name}": var for name, var in delta.items()})

    for v in set(chidens) & set(bounds):
        chidens[v].attrs.update({"bounds": f"{v}_err"})
    return chidens


def lazy_map(grouped, func, *args, **kwargs):
    return grouped.map(
        lambda g: func(
            g.chunk().assign(gamma_n=g.gamma_n),
            *args,
            **kwargs,
        )
    )


def bin_average_vertical(ds, stdname, bins, blocksize, skip_fits=False):
    """Bin averages in the vertical."""

    grouped = ds.reset_coords().cf.groupby_bins(stdname, bins=bins)
    chidens = lazy_map(
        grouped, average_density_bin, blocksize=blocksize, skip_fits=skip_fits
    )

    # for var in set(chidens.variables) & set(ds.variables):
    #    chidens[var].attrs = ds[var].attrs

    var = ds.cf[stdname]
    groupvar = f"{var.name}_bins"
    chidens[groupvar].attrs = var.attrs
    # chidens[groupvar].attrs["positive"] = "down"
    chidens[groupvar].attrs["axis"] = "Z"
    chidens = chidens.set_coords(["pres", "num_obs"])

    chidens["pres"].attrs.update({"positive": "down", "bounds": "pres_err"})

    chidens.coords["num_obs"] = ds.chi.groupby_bins(ds.cf[stdname], bins=bins).count()
    chidens.eps.attrs.update(long_name="$⟨ε⟩$")
    chidens.chi.attrs.update(long_name="$⟨χ⟩$")
    chidens.KtTz.attrs.update(long_name="$⟨K_T θ_z⟩$")
    chidens.num_obs.attrs = {"long_name": "count(χ) in bins"}

    chidens["chib2"] = chidens.chi / 2
    chidens["chib2_err"] = chidens.chi_err / 2
    chidens.chib2.attrs.update({"bounds": "chib2_err", "long_name": "$⟨χ⟩/2$"})

    chidens = chidens.cf.guess_coord_axis()
    # iso_slope = grouped.apply(fit2D)
    # chidens["dTiso"] = np.hypot(iso_slope.x, iso_slope.y)

    bounds = intervals_to_bounds(chidens.gamma_n_bins).rename({"bounds": "bound"})
    chidens = chidens.rename({"gamma_n_bins": "gamma_n"})
    chidens["gamma_n"] = bounds.gamma_n
    chidens.coords["gamma_n_bounds"] = bounds
    chidens["gamma_n"].attrs.update(ds.gamma_n.attrs)
    chidens["gamma_n"].attrs.update({"positive": "down", "axis": "Z"})

    hashes = get_hashes()
    hash_string = " |  ".join(f"{k}: {v}" for k, v in hashes.items())
    chidens.attrs["commit"] = hash_string

    return chidens


def fit1D(group, var, dim="depth", debug=False):
    ds = group.unstack()
    # Some weirdness about binning by dim in grouped variable
    if dim in ds.dims:
        ds = ds.rename({dim: f"{dim}_"})
        ds[dim] = ds[f"{dim}_"].broadcast_like(ds[var])
    bins2 = np.linspace(ds.gamma_n.min().data, ds.gamma_n.max().data, 11)
    mean = (
        ds[[var, "gamma_n", dim]]
        .groupby_bins("gamma_n", bins2)
        .mean()
        .swap_dims({"gamma_n_bins": dim})
    )
    fit = mean[var].polyfit(dim, deg=1)
    slope = fit.polyfit_coefficients.sel(degree=1).data
    if debug:
        recon = xr.polyval(mean[dim], fit)
        plt.figure()
        mean[var].cf.plot(marker=".")
        recon.polyfit_coefficients.cf.plot(marker="x")

    return slope


def fit1D_old(group, var, dim="depth", debug=False):
    """
    Expects a bunch of profiles at different lat, lons.
    Calculates mean profile and then takes linear fit to estimate gradient.
    This works by depth-space averaging...
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

    mean_profile = xarray_reduce(
        stacked,
        "z0",
        func="mean",
        min_count=300,
        fill_value=np.nan,
        expected_groups=(np.arange(-10, 200, 2),),
        isbin=True,
    )
    mean_profile["z0_bins"] = [
        interval.mid for interval in mean_profile.indexes["z0_bins"].values
    ]

    # binned = stacked.groupby_bins("z0", bins=np.arange(-10, 200, 2))
    # count = binned.count()
    # min_count = 300  # count.median() / 1.25
    # mean_profile = binned.mean().where(count > min_count)
    # mean_profile["z0_bins"] = mean_profile.indexes["z0_bins"].mid

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
        y="sea_water_pressure", xscale="log", color="r", lw=2, label="$⟨χ⟩/2$", **kwargs
    )
    (chidens.KtTz * chidens.dTdz_m).cf.plot.step(
        y="sea_water_pressure", color="k", label="$⟨K_T θ_z⟩ ∂_zθ_m$", **kwargs
    )
    ax.grid(True, which="both", lw=0.5)
    plt.legend()
    plt.xlabel("Variance production or dissipation [°C²/s]")
    plt.gcf().set_size_inches((4, 5))


def choose_bins(gamma, depth_range, decimals=2):
    Z = gamma.reset_coords(drop=True).cf.axes["Z"]
    mean_over = set(gamma.dims) - set(Z)
    bins = gamma.mean(mean_over).cf.interp(Z=depth_range)
    bins[Z[0]].attrs["axis"] = "Z"
    return np.round(bins.cf.dropna("Z").data, decimals)


def read_ctd_chipod_mat_file(chifile, ctdfile=None):
    from scipy.io import loadmat

    mat = loadmat(chifile)
    ds = xr.Dataset()

    print("Found variables: ", mat["XPsum"][0, 0].dtype.names)

    varnames = [
        "eps_f",
        "chi_f",
        "KT_f",
        "P_f",
        "cast",
        "sn_avail",
        "SN",
        "lon",
        "lat",
        "datenum",
    ]
    attrs = {
        "eps_f": {"long_name": "$ε$", "units": "W/kg"},
        "chi_f": {"long_name": "$χ$", "units": "C^2/s"},
        "KT_f": {"long_name": "$K_T$", "units": "m^2/s"},
        "P_f": {
            "standard_name": "sea_water_pressure",
            "units": "dbar",
            "axis": "Z",
            "positive": "down",
        },
        "lon": {"standard_name": "longitude"},
        "lat": {"standard_name": "latitude"},
    }

    for var in varnames:
        data = mat["XPsum"][0, 0][var].squeeze()
        if data.ndim == 2:
            if var == "SN":
                dims = ("cast", "sensor")
            else:
                dims = ("P", "cast")
        elif var not in ("cast", "SN", "sn_avail", "lat", "lon", "datenum"):
            dims = ("P",)
        else:
            dims = ("cast",)
        ds[var.split("_")[0]] = (dims, data, attrs.get(var, {}))
    ds = ds.rename({"sn": "num_sensors", "SN": "avail_sensors"}).set_coords(
        ["lat", "lon", "datenum", "num_sensors", "avail_sensors"]
    )
    # TODO
    # ds["time"] = dcpy.util.mdatenum2dt64(ds.datenum)
    ds = ds.cf.guess_coord_axis()
    ds["P"] = ds.P.astype(float)
    ds["cast"] = ("cast", np.arange(2, 145), {"cf_role": "profile_id"})
    ds = ds.rename({"P": "pressure"})

    for var in ["chi", "eps", "KT"]:
        ds[var] = ds[var].where(ds[var] > 0)

    if ctdfile is not None:
        # drop gappy lat, lon from Aurelie
        # We can use the CTD data instead
        ds = ds.drop_vars(["lat", "lon"])

        ctd_raw = xr.open_dataset(ctdfile)
        # line up cast numbers with CTD-χpod
        ctd_raw["cast"] = ctd_raw.station.astype(int)
        ctd_raw = ctd_raw.drop_vars("station").swap_dims({"N_PROF": "cast"})
        ctd_raw.cast.attrs["cf_role"] = "profile_id"

        P = ds.cf["sea_water_pressure"].data
        edges = (P[:-1] + P[1:]) / 2

        grouped = ctd_raw.cf.groupby_bins("sea_water_pressure", bins=edges)
        ctd = grouped.mean("N_LEVELS")
        index = pd.IntervalIndex(ctd.xindexes["pressure_bins"].to_pandas_index())
        ctd.coords["pressure"] = ("pressure_bins", index.mid.values)
        ctd = ctd.swap_dims({"pressure_bins": "pressure"})
        ctd.pressure.attrs = ctd_raw.pressure.attrs
        ctd.pressure.attrs["bounds"] = "pressure_bins"

        ds["temp"] = ctd.ctd_temperature
        ds["salt"] = ctd.ctd_salinity
        ds.coords["bottom_depth"] = ctd.btm_depth.isel(pressure=5, drop=True)
        ds["gamma_n"] = dcpy.oceans.neutral_density(ds)

        ds = add_ancillary_variables(ds)

    return ds


def compute_eke(section):
    """Computes climatological, monthly mean, and cruise EKE."""

    aviso = xr.open_dataset(
        os.path.expanduser("~/datasets/aviso_monthly.zarr"),
        chunks="auto",
        engine="zarr",
    )
    aviso["eke"] = (aviso.ugosa**2 + aviso.vgosa**2) / 2
    t0, t1 = tuple(pd.Timestamp(t) for t in section.time.values.ravel()[[0, -1]])
    months = np.concatenate([np.arange(t0.month, 13), np.arange(1, t1.month + 1)])

    eke = xr.Dataset()

    sub = aviso.eke.sel(time=slice(t0 - pd.DateOffset(months=1), t1)).mean("time")
    interp_kwargs = dict(
        latitude=section.cf["latitude"], longitude=section.cf["longitude"] + 360
    )
    eke["cruise"] = sub.interp(**interp_kwargs)
    eke["monthly"] = (
        aviso.eke.sel(time=aviso.time.dt.month.isin(months))
        .groupby("time.month")
        .mean()
        .mean("month")
        .interp(**interp_kwargs)
    )
    eke["clim"] = aviso.eke.mean("time").interp(**interp_kwargs)
    return eke
