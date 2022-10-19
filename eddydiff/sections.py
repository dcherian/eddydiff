import glob
import os
from functools import partial

import cf_xarray as cfxr  # noqa
import dask
import datatree
import dcpy
import flox.xarray
import gsw
import gsw_xarray
import matplotlib as mpl
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import xfilter
from arch.bootstrap import IIDBootstrap, MovingBlockBootstrap
from flox.xarray import xarray_reduce

import xarray as xr

from .eddydiff import intervals_to_bounds
from .utils import get_hashes

criteria = {
    "sea_water_salinity": {
        "standard_name": "sea_water_salinity|sea_water_practical_salinity"
    }
}

database = {
    "A05_2015": {
        "section": "A05",
        "expocode": "74EQ20151206",
        "station_bin_edges": [0, 28, 60, 82, 105, 130, 145],
    }
}


def get_filenames(section_id):
    try:
        data = database[section_id]
    except KeyError as e:
        raise KeyError(f"I only know about f{database.keys()!r}") from e

    section = data["section"]
    expocode = data["expocode"]
    rootdir = os.path.expanduser(f"~/work/eddydiff/datasets/goship/{section}")
    section = database[section_id]["section"]
    expocode = database[section_id]["expocode"]

    prefix = f"{section}_{expocode}_"

    files = {
        "ctd": glob.glob(f"{rootdir}/{prefix}ctd.nc")[0],
        "ctdchi": f"{rootdir}/{prefix}chipod.nc",
        "finescale": f"{rootdir}/{prefix}finescale.nc",
        "merged": f"{rootdir}/{prefix}merged.nc",
    }
    return files


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


def add_ctd_ancillary_variables(ctd):
    with cfxr.set_options(custom_criteria=criteria):
        salt, temp, pres = (
            ctd.cf["sea_water_salinity"],
            ctd.cf["sea_water_temperature"],
            ctd.cf["sea_water_pressure"],
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

    ctd["SA"] = gsw.SA_from_SP(
        salt,
        pres,
        ctd.cf["longitude"],
        ctd.cf["latitude"],
    )
    ctd.SA.attrs["standard_name"] = "sea_water_absolute_salinity"
    ctd.SA.attrs["long_name"] = "$S_A$"
    ctd.SA.attrs["units"] = "g/kg"

    ctd["CT"] = gsw.CT_from_t(salt, temp, pres)
    ctd.CT.attrs = {
        "standard_name": "sea_water_conservative_temperature",
        "long_name": "$Θ$",
        "units": "degC",
    }

    Z = "sea_water_pressure"
    # Match Aurelie
    lowpass = partial(
        xfilter.lowpass, coord=ctd.cf.standard_names[Z][0], freq=1 / 100, num_discard=0
    )
    ctd["Tfilt"] = lowpass(ctd.CT)
    ctd["Sfilt"] = lowpass(ctd.SA)

    Tu, Rρ, pmid = xr.apply_ufunc(
        gsw_xarray.Turner_Rsubrho,
        ctd.SA,
        ctd.CT,
        pres,
        input_core_dims=[[pres.name]] * 3,
        output_core_dims=[[pres.name]] * 3,
        exclude_dims={pres.name},
        kwargs=dict(axis=-1),
        dask="parallelized",
    )
    out = lowpass(xr.Dataset({"Tu": Tu, "Rρ": Rρ}))
    out.Tu.attrs = {
        "long_name": "$Tu$",
        "standard_name": "turner_angle",
        "units": "degrees",
    }
    out.Rρ.attrs = {
        "long_name": "$R_ρ$",
        "standard_name": "density_ratio",
        "units": "1",
    }
    # out is at cell-centers, average to faces and fix coordinate
    ctd = ctd.update(
        out.rolling({pres.name: 3}, center=True)
        .mean()
        .isel({pres.name: slice(1, None)})
        .assign({pres.name: pres[1:-1]})
    )
    # ctd["Tu"] = dcpy.oceans.turner_angle(ctd)

    if "neutral_density" not in ctd.cf:
        ctd["gamma_n"] = dcpy.oceans.neutral_density(ctd)

    if "dTdz" in ctd:
        ctd = ctd.rename_vars({"dTdz": "Tz_orig"})

    if "Tz" not in ctd:
        ctd["Tz"] = lowpass(
            ctd.CT.interpolate_na(Z).cf.differentiate(Z, positive_upward=True)
        )
    ctd["Tz"].attrs["long_name"] = "$θ_z$"
    ctd["Tz"].attrs["units"] = "degC/m"

    def take_(arr, slicer, axis):
        idxr = [slice(None)] * arr.ndim
        idxr[axis] = slicer
        return arr[tuple(idxr)]

    if "N2" not in ctd:
        zaxis = ctd.SA.cf.get_axis_num("Z")
        Zname = ctd.cf.axes["Z"][0]
        N2, pmid = lowpass(
            gsw.Nsquared(
                ctd.SA,
                ctd.CT,
                pres.broadcast_like(ctd.SA),
                ctd.cf["latitude"].broadcast_like(ctd.SA),
                axis=zaxis,
            )
        )
        ctd["N2"] = xr.DataArray(
            (take_(N2, slice(-1), zaxis) + take_(N2, slice(1, None), zaxis)) / 2,
            dims=ctd.SA.dims,
            coords={Zname: ctd[Zname].isel({Zname: slice(1, -1)})},
        )
    ctd["N2"].attrs["long_name"] = "$N²$"
    ctd["N2"].attrs["units"] = "s-2"

    ctd["τ0"] = lowpass(gsw.spiciness0(ctd.SA, ctd.CT))
    ctd.τ0.attrs = {"standard_name": "spiciness", "long_name": "$τ_0$"}

    ctd["τ1"] = lowpass(gsw.spiciness1(ctd.SA, ctd.CT))
    ctd.τ1.attrs = {"standard_name": "spiciness", "long_name": "$τ_1$"}


def add_turbulence_ancillary_variables(ds):
    Tz_mask = np.abs(ds.Tz) > 3e-4
    N2_mask = (ds.N2) > 1e-7

    # ds["chi_masked"] = ds.chi.where(Tz_mask)

    ds["chib2"] = ds.chi / 2
    ds["chib2"].attrs["long_name"] = "$χ/2$"

    if "eps" in ds:
        if "Krho" not in ds:
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


def add_ancillary_variables(ds):
    """Adds ancillary variables."""

    if isinstance(ds, datatree.DataTree):
        add_ctd_ancillary_variables(ds["ctd"].ds)
        add_ctd_ancillary_variables(ds["chipod"].ds)
        add_turbulence_ancillary_variables(ds["finescale"].ds)
        ctdchi = ds["chipod"].ds
    else:
        add_ctd_ancillary_variables(ds)
        ctdchi = ds

    add_turbulence_ancillary_variables(ctdchi)


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


def compute_bootstrapped_var(array, corrscales, dp):
    np.mean(IIDBootstrap(array[~np.isnan(array)]).apply(np.std))


def compute_bootstrapped_mean_ci(array, blocksize, clean=False, debug=False):
    from numpy.random import RandomState

    rs = RandomState(1234)

    # drop nans, significantly faster to do this than bootstrap with nanmean
    assert array.ndim == 1
    array = array[~np.isnan(array)]

    if debug:
        plt.figure()
        plt.plot(array, "x")

    if clean:
        # filter out some real outliers
        absarray = np.abs(array)
        thresh = np.mean(absarray) + 50 * np.std(absarray)
        array = np.where(np.abs(array) < thresh, array, np.nan)
        array = array[~np.isnan(array)]

    if debug:
        plt.plot(array, "x")
        plt.axhline(thresh)
        plt.axhline(-thresh)

    return np.insert(
        MovingBlockBootstrap(blocksize, array, seed=rs)
        .conf_int(func=np.mean, method="bca")
        .squeeze(),
        1,
        np.mean(array),
    )


def add_error(field, fields, delta, *terms):
    from functools import reduce
    from operator import add

    delta[field] = (
        fields[field]
        * np.sqrt(reduce(add, ((delta[var] / fields[var]) ** 2 for var in terms)))
    ).reset_coords(drop=True)


def get_dof(da, dp, corrscales):
    dof = 0
    for region, scale in corrscales:
        dof += (
            (
                da.sel(pres=region)
                .coarsen(pres=int(scale // dp), boundary="trim")
                .count()
                > 1
            )
            .sum()
            .data
        )
    return dof


def average_density_bin(group, dp, blocksize, skip_fits=False):
    # groupby_bins ends up stacking the pressure coordinate
    # which deletes attrs so we can't use cf-xarray here
    # Z = "sea_water_pressure"

    profiles = group.unstack()
    profilevar = group.cf.cf_roles["profile_id"][0]

    if "pres" in profiles.dims:
        Z = "pres"
    elif "pressure" in profiles.dims:
        Z = "pressure"
    else:
        raise ValueError("Don't know what the pressure dimension is.")

    # Mean χ, ε
    # flatten for bootstrap
    flattened = (
        profiles[["chi", "eps", "KtTz", "KtTz~"]]
        .reset_coords(drop=True)
        .stack(flat=[...], create_index=False)
        .chunk({"flat": -1})
    )

    ci = xr.apply_ufunc(
        compute_bootstrapped_mean_ci,
        flattened,
        input_core_dims=[["flat"]],
        exclude_dims={"flat"},
        # TODO: configure this
        kwargs={"blocksize": blocksize, "clean": True},
        output_core_dims=[["bound"]],
        dask_gufunc_kwargs=dict(output_sizes={"bound": 3}),
        dask="parallelized",
        output_dtypes=[float],
    ).assign_coords(bound=["lower", "center", "upper"])

    ci["chib2"] = ci["chi"] / 2

    # Mean h_m: separation between γ surfaces
    pres = profiles[Z].where(profiles.gamma_n.notnull())
    hm = pres.max(Z) - pres.min(Z)
    assert np.all(hm >= 0)
    hm = hm.where(hm > 5)
    ci["hm"] = xr.apply_ufunc(
        compute_mean_ci,
        hm,
        input_core_dims=[[profilevar]],
        exclude_dims={profilevar},
        output_core_dims=[["bound"]],
        dask_gufunc_kwargs=dict(output_sizes={"bound": 3}),
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )
    ci["hm"].attrs = {
        "long_name": "$h_m$",
        "description": "separation between γ_n surfaces",
        "units": "m",
    }

    # w'T' estimated in temperature space
    # wT = estimate_microscale_stirring(profiles)
    # ci["wT"] = xr.apply_ufunc(
    #     compute_mean_ci,
    #     wT,
    #     exclude_dims=set(wT.dims),
    #     input_core_dims=[wT.dims],
    #     # exclude_dims=set(wT.dims),
    #     output_core_dims=[["bound"]],
    #     dask_gufunc_kwargs=dict(output_sizes={"bound": 3}),
    #     dask="parallelized",
    #     output_dtypes=[float],
    #     keep_attrs=True,
    # )
    ci["wT"] = xr.full_like(ci.hm, fill_value=np.nan)

    # derived quantities with error
    chidens = ci.sel(bound="center")
    bounds = ci.sel(bound=["lower", "upper"])
    # Dividing by 2 is important here.
    # Because delta should the standard error BUT
    # diff("bound") is 2x standard error
    delta = bounds.diff("bound").squeeze() / 2

    unit = xr.DataArray([-1, 1], dims="bound", coords={"bound": ["lower", "upper"]})

    chidens["Γ"] = 0.2
    delta["Γ"] = 0.04

    if not skip_fits:
        with cfxr.set_options(custom_criteria=criteria):
            chidens["salt"] = profiles.cf["sea_water_salinity"].mean()

        chidens["CT"] = profiles.cf["sea_water_conservative_temperature"].mean()
        chidens["gamma_n_"] = profiles.gamma_n.mean()

        profiles["gamma_n_"] = profiles.gamma_n
        slopes = -1 * fit1D(profiles, var=["CT", "gamma_n_"], dim=Z)

        chidens["dTdz_m"] = slopes["CT_polyfit_coefficients"]
        chidens.dTdz_m.attrs.update(
            dict(
                long_name="$∂_z θ_m$",
                units="°C/m",
                description="vertical gradient of potential temperature θ with respect to depth",
            )
        )
        add_error("dTdz_m", chidens, delta, "hm")

        chidens["N2_m"] = (
            -9.81 / (1000 + chidens.gamma_n_) * slopes["gamma_n__polyfit_coefficients"]
        )
        chidens.N2_m.attrs.update(
            dict(
                long_name="$∂_zb_m$",
                units="s$^{-2}$",
                description="vertical gradient of neutral density γ_n with respect to depth",
            )
        )
        add_error("N2_m", chidens, delta, "hm")

        chidens["Krho_m"] = chidens.Γ * chidens.eps / chidens.N2_m
        add_error("Krho_m", chidens, delta, "Γ", "eps", "N2_m")
        chidens.Krho_m.attrs.update(dict(long_name="$K_ρ^m$", units="m²/s"))

        chidens["Kt_m"] = 0.5 * chidens.chi / chidens.dTdz_m**2
        add_error("Kt_m", chidens, delta, "chi", "hm", "hm")
        chidens.Kt_m.attrs.update(dict(long_name="$K_T^m$", units="m²/s"))

        chidens["KρTz2"] = chidens.Krho_m * chidens.dTdz_m**2
        add_error("KρTz2", chidens, delta, "Krho_m", "dTdz_m", "dTdz_m")
        chidens.KρTz2.attrs = {"long_name": "$K_ρ ∂_zθ_m^2$"}

        chidens["KtTzTz"] = chidens.KtTz * chidens.dTdz_m
        add_error("KtTzTz", chidens, delta, "KtTz", "dTdz_m")
        chidens.KtTzTz.attrs = {"long_name": "$⟨K_T θ_z⟩ ∂_zθ_m$"}

        chidens["KtTz~Tz"] = chidens["KtTz~"] * chidens.dTdz_m
        add_error("KtTz~Tz", chidens, delta, "KtTz~", "dTdz_m")
        chidens["KtTz~Tz"].attrs = {"long_name": "$⟨K_T θ_z⟩ ∂_zθ_m$"}

        chidens["wTTz"] = chidens.wT * chidens.dTdz_m
        add_error("wTTz", chidens, delta, "wT", "dTdz_m")
        chidens.wTTz.attrs = {"long_name": "$⟨w'θ'⟩ ∂_zθ_m$"}

        chidens["residual"] = chidens.chi / 2 - chidens.KρTz2
        add_error("residual", chidens, delta, "chib2", "KρTz2")

        chidens["residual_chi"] = chidens.chi / 2 - chidens["KtTz~Tz"]
        add_error("residual_chi", chidens, delta, "chib2", "KtTz~Tz")

    bounds = chidens + unit * delta

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


def bin_average_vertical(
    ds, stdname, bins, blocksize, return_group=False, skip_fits=False
):
    """Bin averages in the vertical."""

    needed_vars = [
        "chi",
        "eps",
        "KtTz",
        "KtTz~",
        "gamma_n",
        "sea_water_salinity",
        # "sea_water_temperature",
        "sea_water_conservative_temperature",
    ]
    dp = ds.cf["Z"].diff("Z").median().data
    with cfxr.set_options(custom_criteria=criteria):
        grouped = ds.reset_coords().cf[needed_vars].cf.groupby_bins(stdname, bins=bins)
    if return_group:
        return grouped
    chidens = grouped.map(
        average_density_bin,
        blocksize=blocksize,
        dp=dp,
        # corrscales=corrscales,
        skip_fits=skip_fits,
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

    chidens.eps.attrs.update(long_name="$⟨ε⟩$")
    chidens.chi.attrs.update(long_name="$⟨χ⟩$")
    chidens.chib2.attrs.update({"long_name": "$⟨χ⟩/2$"})
    chidens.KtTz.attrs.update(long_name="$⟨K_T θ_z⟩$")
    chidens.num_obs.attrs = {"long_name": "count(χ) in bins"}

    # chidens = chidens.cf.guess_coord_axis()
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

    # stations used
    profilevar = ds.cf.cf_roles["profile_id"][0]
    chidens = chidens.assign_coords({profilevar: ds[profilevar]})

    return chidens


def fit1D(ds, var, dim="depth", debug=False):
    if isinstance(var, str):
        var = [var]
    # ds = group.unstack()
    # Some weirdness about binning by dim in grouped variable
    if dim in ds.dims:
        ds = ds.rename({dim: f"{dim}_"})
        ds[dim] = ds[f"{dim}_"].broadcast_like(ds[var])
    bins2 = np.linspace(ds.gamma_n.min().data, ds.gamma_n.max().data, 11)
    mean = flox.xarray.xarray_reduce(
        ds[var + ["gamma_n", dim]],
        "gamma_n",
        func="mean",
        expected_groups=bins2,
        isbin=True,
        method="map-reduce",
        engine="numpy",
    ).swap_dims({"gamma_n_bins": dim})
    fit = mean[var].dropna(dim).polyfit(dim, deg=1)
    slope = fit.sel(degree=1, drop=True)
    if debug:
        if len(var) > 1:
            raise NotImplementedError
        recon = xr.polyval(mean[dim], fit)
        plt.figure()
        mean[var[0]].cf.plot(marker=".")
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
    # .expand_dims(pden_bins=[label])
    slope = slope.squeeze().drop_vars("variable")
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


def choose_bins(gamma, depth_range, decimals=2, sort=False):
    Z = gamma.reset_coords(drop=True).cf.axes["Z"]
    mean_over = set(gamma.dims) - set(Z)
    bins = gamma.mean(mean_over).cf.interp(Z=depth_range)
    bins[Z[0]].attrs["axis"] = "Z"
    if sort:
        bins = bins.copy(data=np.sort(bins.data))
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


def diapycnal_spiciness_curvature(section):
    import xfilter
    from xgcm.transform import linear_interpolation

    to = xr.DataArray(
        np.arange(23, 28.5, 0.02),
        dims="gamma_n",
        attrs={
            "axis": "Z",
            "positive": "down",
            "long_name": "$γ_n$",
            "standard_name": "neutral_density",
            "units": "kg/m3",
        },
    )
    interped = linear_interpolation(
        np.tan(section.Tu),
        section.gamma_n,
        to,
        "pressure",
        "pressure",
        "gamma_n",
    )
    pi = linear_interpolation(
        section.pressure,
        section.gamma_n,
        to,
        "pressure",
        "pressure",
        "gamma_n",
    )
    interped["gamma_n"] = to
    interped.coords["pressure"] = pi
    dsc = xfilter.lowpass(
        interped.differentiate("gamma_n"), "gamma_n", freq=1 / 0.08, num_discard=0
    )
    dsc.name = "dsc"
    dsc.attrs["standard_name"] = "diapycnal_spiciness_curvature"
    dsc.attrs["long_name"] = "$τ_{σσ}$"

    dsc["pressure"] = dsc.pressure.fillna(0)
    return dsc


def _process_finescale_single_cast(cast, **kwargs):
    import cf_xarray as cfxr
    import dcpy.finestructure

    with cfxr.set_options(warn_on_missing_variables=False):
        result = dcpy.finestructure.process_profile(cast, **kwargs)

    result = result.expand_dims("station").reset_coords(
        [v for v in result.coords if "mode" in v or "mld" in v]
    )
    return result


def process_finescale_estimate(section, **kwargs):
    profilevar = section.cf.cf_roles["profile_id"]
    assert len(profilevar) == 1
    profilevar = profilevar[0]

    tasks = [
        dask.delayed(_process_finescale_single_cast)(
            section.isel({profilevar: idx}), **kwargs
        )
        for idx in range(section.sizes[profilevar])
    ]
    (computed,) = dask.compute(tasks)

    sectionturb = xr.concat(computed, dim=profilevar)
    sectionturb[profilevar] = section[profilevar]
    return sectionturb


def calc_mean_dp(group, core_dim, name):
    pres = group.unstack()[name]
    dp = (pres.max(core_dim) - pres.min(core_dim)).rename("dp")
    return dp.where(dp > 0).mean()


def estimate_microscale_stirring(density_bin, dz=5, debug=False):
    """
    ERROR: This does not account for sign.

    Estimates \bar{w'θ'} = (\bar{χ}/2) / (\bar{dT/dz}) in temperature space.

    1. Sorts all profiles by temperature along pressure.
    2. Estimates a characteristic mean dTdz; then spaces bins at approximately dz m apart.
    3. Chooses temperature bins:
        a. first and last bin edge are the 1st and 99th percentile.
        b. spacing dT ~ dTdz * dz
    4. Estimates δpressure, mean(χ)

    Parameters
    ----------
    density_bin: xr.Dataset
        One neutral density bin of data
    dz: number
        Nominal bin spacing in depth
    """
    needed_vars = ["chi", "CT", "gamma_n"] + (["KtTz"] if debug else [])

    profiles = density_bin[needed_vars]

    core_dim = profiles.cf.axes["Z"][0]
    profile_id = profiles.cf.cf_roles["profile_id"][0]

    # 1. sort by temperature
    sortT = dcpy.oceans.thorpesort(
        profiles, by="CT", core_dim=core_dim, ascending=False
    )
    pnew = f"{core_dim}_broadcast"
    sortT[pnew] = sortT[core_dim].broadcast_like(sortT.CT).where(sortT.CT.notnull())
    if debug:
        sortT["KtTz"] = np.abs(sortT["KtTz"])
        sortT["Tz"] = -1 * sortT.CT.differentiate(core_dim)

    # 2. Find a characteristic dTdz, used to space temperature bins
    Tmean = sortT.CT.mean(profile_id).where(sortT.CT.count(profile_id) > 20)
    Pmean = sortT[core_dim].where(sortT.CT.notnull()).mean(profile_id)
    Tmean.coords[core_dim] = Pmean
    linear_fit = Tmean.polyfit(core_dim, deg=1).polyfit_coefficients
    fit = xr.polyval(Tmean[core_dim], linear_fit)
    dTdzmean = linear_fit.sel(degree=1).data

    if debug:
        sortT.CT.plot(hue=profile_id, lw=1, add_legend=False)
        sortT.CT.mean(profile_id).where(sortT.CT.count(profile_id) > 20).plot(
            color="k", lw=2
        )
        profiles.CT.mean(profile_id).where(profiles.CT.count(profile_id) > 20).plot(
            color="r", lw=2
        )
        plt.plot(Pmean, Tmean, color="b")
        fit.plot(color="cyan", marker=".")

    Tlims = sortT.CT.quantile(q=[0.05, 0.95])
    Tbins = np.arange(Tlims[0].data, Tlims[1].data, np.abs(dTdzmean) * dz)

    # 4. groupby-mean in Tbins
    sortT = sortT.set_coords("CT")
    grouped = sortT.groupby_bins("CT", bins=Tbins)
    count = sortT["chi"].groupby_bins("CT", bins=Tbins).count()
    mean = grouped.mean().where(count > 20)
    mean["dp"] = (
        sortT[[pnew, "CT"]]
        .groupby_bins("CT", bins=Tbins)
        .map(calc_mean_dp, name=pnew, core_dim=core_dim)
    )
    mean["dT"] = mean.CT_bins.copy(data=[b.right - b.left for b in mean.CT_bins.data])
    mean["dTdz"] = mean.dT / mean.dp
    mean["wT"] = mean.chi / 2 / mean.dTdz
    mean.wT.attrs = {
        "long_name": "$w'θ'$",
        "description": "calculated as mean(χ/2) * mean(δP) / δT",
        "units": "degC m/s",
    }

    if debug:
        f, ax = plt.subplots(2, 1)
        plt.sca(ax[0])
        mean.dTdz.plot()
        mean.Tz.plot(color="k")
        dcpy.plots.liney(np.abs(dTdzmean))

        plt.sca(ax[1])
        (mean.wT).plot(yscale="log")
        (mean.KtTz).plot(color="k")
        dcpy.plots.liney(mean.wT.mean(), color="r")
        dcpy.plots.liney(sortT.KtTz.mean(), color="k")

    return mean.wT


def estimate_interleaving(section, Zscale=10):
    nz = int(Zscale // section.SA.cf["Z"].cf.diff("Z").median().values)
    Zname = section.SA.cf["Z"].name
    smoothed = (
        section[["CT", "SA"]]
        .cf.rolling(Z=nz, center=True, min_periods=1)
        .mean()
        .cf.coarsen(Z=nz, boundary="trim")
        .mean()
    )

    smoothed["NT2"] = (
        -9.81
        * gsw.alpha(smoothed.SA, smoothed.CT, smoothed[Zname])
        * smoothed.CT.cf.interpolate_na("Z").cf.differentiate("Z")
    )
    smoothed["NS2"] = (
        9.81
        * gsw.beta(smoothed.SA, smoothed.CT, smoothed[Zname])
        * smoothed.SA.cf.interpolate_na("Z").cf.differentiate("Z")
    )

    N2, pmid = gsw.Nsquared(
        smoothed.SA, smoothed.CT, smoothed.cf["Z"].broadcast_like(smoothed.CT), axis=0
    )
    smoothed["N2"] = (("pmid", "cast"), N2)
    smoothed["pmid"] = pmid[:, 0]

    λ0 = smoothed.NS2 / smoothed.NT2
    λ = (1 - λ0) / (1 + λ0)
    λ.attrs["long_name"] = "$\\frac{(1 - |N_S^2/N_T^2|)}{(1 + |N_S^2/N_T^2|)}$"
    λ.cf.plot(x="cast", y=Zname, cmap=mpl.cm.plasma, vmin=-4, vmax=4)


def mode(obj, dim):
    import scipy as sp

    def wrapper(data, **kwargs):
        return sp.stats.mode(data, **kwargs).mode.data.squeeze()

    result = xr.apply_ufunc(
        wrapper,
        obj,
        input_core_dims=[[dim]],
        kwargs={"axis": -1, "nan_policy": "omit"},
    )
    result = result.where(np.abs(result) > 1e-6)
    return result


def estimate_microscale_stirring_depth_space(ds, filter_len, segment_len, debug=False):
    """
    Estimates $χ/2/T_z$ using sorting over segment_len long segments.

    Sign of the sorted gradient is set by sign of T_z calculated using central differences
    after filtering T over filter_len with a forward-backward first order butterworth filter

    Output is reindexed back to input grid.

    Parameters
    ----------
    ds: Dataset
        Gets updated in place.
    filter_len: Number, units: dbar or m
        Filter length
    segment_dp: Number, units: dbar or m
        Length of segment over which to estimate gradient
    """
    import xfilter

    ds = ds.copy(deep=True)

    debug_profile = ds.chi.ndim == 1 and debug

    Zname = ds.cf.axes["Z"][0]
    dp = ds[Zname].diff(Zname).median().item()
    nfilter = int(filter_len // dp)
    ncoarse = int(segment_len // dp) + 1
    print(ncoarse, nfilter)
    pcoarse = f"pres_{segment_len}"

    CTinterp = ds.CT.interpolate_na(Zname)
    ds["Tfilt"] = xfilter.lowpass(CTinterp, coord=Zname, freq=1 / nfilter, order=1)
    ds["Tfilt"].attrs = {
        "long_name": r"$\widetilde{T}$",
        "units": "degrees_Celsius",
        "description": f"Butterworth filter, second order, {nfilter} points over {filter_len} dbar",
    }
    ds["Tzfilt"] = ds.Tfilt.cf.differentiate(Zname, positive_upward=True)
    ds["Tzfilt"].attrs = {
        "long_name": r"$\widetilde{T_z}$",
        "units": "degrees_Celsius / m",
        "description": f"Butterworth filter, second order, {nfilter} points over {filter_len} dbar",
    }

    # return ds.Tzfilt
    Tzsign = xr.apply_ufunc(
        get_gradient_sign,
        ds[Zname],
        CTinterp,
        ds.Tzfilt,
        5,
        ncoarse,
        input_core_dims=[[Zname]] * 3 + [[], []],
        output_core_dims=[[Zname]],
    )

    to_coarsen = ds[["gamma_n"]].interpolate_na(Zname)
    to_coarsen["CT"] = CTinterp
    to_coarsen["chi"] = ds.chi
    to_coarsen["Tzsign"] = Tzsign
    # to_coarsen["Tzfilt"] = ds.Tzfilt

    # return to_coarsen

    coarse = (
        to_coarsen.coarsen({Zname: ncoarse}, boundary="trim")
        .construct({Zname: (pcoarse, "window")})
        .reset_coords(Zname)
    )

    # 1. Sort by CT, to get a stable gradient
    Tsort = dcpy.oceans.thorpesort(
        coarse.CT, by=coarse.CT, ascending=False, core_dim="window"
    )

    # Assign "pressure" values for fitting
    assert (Tsort.diff("window") > 0).sum().astype(int).item() == 0
    Tsort["window"] = -1 * np.arange(0, ncoarse * dp, dp)

    if debug and not debug_profile:
        Tsort.count("window").plot.hist(
            bins=np.arange(-dp / 2, segment_len + dp, dp), yscale="log"
        )

    clean = xr.Dataset()
    clean["Tz~"] = (
        # assert min number of points to get a "nice" gradient
        Tsort.where(Tsort.count("window") > 3)
        .polyfit(dim="window", deg=1)
        .polyfit_coefficients.sel(degree=1, drop=True)
    )

    # 2. Choose sign from Tztilde, i.e. CT filtered over filter_len
    # Tzsign = mode(np.sign(coarse.Tzfilt), "window")
    # Tzsign = np.sign(coarse.Tzfilt.median("window"))
    # Tzsign = xr.where((coarse.Tzfilt < 0).any("window"), -1, 1)
    # 2.a Applying sign of first element in window. This could be improved
    Tzsign = coarse.Tzsign.isel(window=0, drop=True)
    # print((Tzsign > 0).sum().data, " values < 0")
    clean["Tz~"] *= Tzsign
    clean["Tz~"] = clean["Tz~"].where(np.abs(clean["Tz~"]) > 1e-4)
    print((clean["Tz~"] == 0).sum().data, " values == 0")

    clean["chi~"] = coarse.chi.where(coarse.chi.count("window") > 3).mean("window")
    clean["Kt~"] = clean["chi~"] / 2 / clean["Tz~"] ** 2
    clean["KtTz~"] = clean["chi~"] / 2 / clean["Tz~"]
    clean["KtTz~"] = clean["KtTz~"].where(np.abs(clean["KtTz~"]) < 2e-5)

    # mean of "pressure" along window might be aligned for the reindex step.
    clean[pcoarse] = coarse[Zname].min("window")
    clean = clean.reindex({pcoarse: ds[Zname].data}).rename({pcoarse: Zname})

    # N = Tsort.sizes["window"] * Tsort.sizes[pcoarse]
    # Need to flip "window" based on Tz_sign
    # flipping_idx = Tsort.window.copy(data=np.arange(Tsort.sizes["window"])) * Tzsign.fillna(0)
    # clean["T~"] = Tsort[flipping_idx.astype(int)].stack(
    #    {Zname: (pcoarse, "window")}, create_index=False
    # ).assign_coords({Zname: ds[Zname][:N]})
    # del clean["T~"].attrs["standard_name"]

    print(clean["Tz~"].count().item())

    if debug_profile:
        filled = clean.cf.ffill("Z")
        f, ax = plt.subplots(4, 1, sharex=True, constrained_layout=True)
        ds.CT.plot(ax=ax[0])
        ds.Tfilt.plot(ax=ax[0])

        filled["Tz~"].plot(ax=ax[1])
        (ds.Tzfilt).plot(ax=ax[1])

        (ds.chi).plot(yscale="log", ylim=(1e-12, None), ax=ax[2])
        (filled["chi~"]).plot(yscale="log", ylim=(1e-12, None), ax=ax[2])

        ax2 = ax[2].twinx()
        filled["Tz~"].plot(ax=ax2, color="k")
        # (dTdz.where(np.abs(dTdz) < 2e-4)).plot(color="b", marker="o", ls="none", ax=ax2)
        dcpy.plots.set_axes_color(ax2, "k")

        (filled["KtTz~"]).plot(ax=ax[3], yscale="symlog")
        (ds.chi / ds.Tzfilt).plot(ax=ax[3], ylim=(-5e-5, 5e-5))
        # (1025 * 4000 * -chi / dTdz.where(np.abs(dTdz) < 2e-4)).plot(
        #    color="b", marker="o", ls="none", ax=ax[3]
        # )

        dcpy.plots.clean_axes(np.atleast_2d(ax).T)
        f.set_size_inches((10, 7))

    elif debug:
        plt.figure()
        np.log10(np.abs(clean["KtTz~"].where(clean["KtTz~"] > 0))).plot.hist(
            bins=101, density=True, histtype="step", lw=2
        )
        np.log10(np.abs(clean["KtTz~"].where(clean["KtTz~"] < 0))).plot.hist(
            bins=101, density=True, histtype="step", lw=2
        )

        np.log10(np.abs(clean["Tz~"].where(clean["Tz~"] > 0))).plot.hist(
            bins=101, density=True, histtype="step", color="C0"
        )
        np.log10(np.abs(clean["Tz~"].where(clean["Tz~"] < 0))).plot.hist(
            bins=101, density=True, histtype="step", color="C1"
        )

    return clean


@numba.njit()
def _clean_numba(idx, P, ΔP, window):
    newidx = [idx[0]]
    for left, right in zip(idx[:-1], idx[1:]):
        Pl, Pr = P[left], P[right]
        # print(np.abs(Pr - Pl))
        if np.abs(Pr - Pl) > ΔP:
            if ((right - left) % window) < window // 2:
                right += 1
            newidx.append(right)

    if newidx[-1] != idx[-1]:
        # Last index was dropped
        # instead to preserve all data;
        # we drop the previous one and re-add the last one
        newidx.pop()
        newidx.append(idx[-1])
    # print(idx, newidx)
    return newidx


@numba.guvectorize(
    "(float64[:], float64[:], float64[:], float64, float64, float64[:])",
    "(m), (m), (m), (), () -> (m)",
    nopython=True,
)
def get_gradient_sign(P, CT, Tzfilt, ΔP, window, out):
    out[:] = np.nan

    naninds = np.nonzero(~np.isnan(CT))[0]
    i0 = naninds[0]
    i1 = naninds[-1]

    (idx,) = np.nonzero(np.abs(np.diff(np.sign(Tzfilt[i0:i1]))) > 0)

    if len(idx) == 0:
        out[i0:i1] = 1
    else:
        idx = np.concatenate((np.array([0]), idx, np.array([i1 - i0])))

        idx = _clean_numba(idx, P[i0:i1], ΔP=ΔP, window=window)
        idx = _clean_numba(idx, CT[i0:i1], 1e-4 * ΔP, window=1e10)
        idx = np.array(idx) + i0
        numel = np.diff(idx)
        # print(idx)

        signs = np.sign(np.repeat(CT[idx[:-1]], numel) - np.repeat(CT[idx[1:]], numel))
        # print(CT[idx])
        out[i0:i1] = signs
