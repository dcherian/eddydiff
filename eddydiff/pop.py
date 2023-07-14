import glob

import cf_xarray  # noqa
import numpy as np
import pop_tools
from dcpy.util import to_base_units

import xarray as xr

from .eddydiff import plane_fit_gradient

pop_metric_vars = ["UAREA", "TAREA", "DXU", "DXT", "DYU", "DYT"]

metrics = {
    ("X",): ["DXU", "DXT"],  # X distances
    ("Y",): ["DYU", "DYT"],  # Y distances
    # ("Z",): ["DZU", "DZT"],  # Z distances
    ("X", "Y"): ["UAREA", "TAREA"],
}


def read_1deg():
    path = (
        "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/"
        "g.e21.GOMIPECOIAF_JRA.TL319_g17.CMIP6-omip2.001/"
        "ocn/proc/tseries/month_1"
    )
    paths = []
    for substring in [".TEMP.", "KAPPA_ISOP", ".SALT.", "SSH"]:
        paths += glob.glob(f"{path}/*{substring}*.nc")

    pop1_ = xr.open_mfdataset(
        paths,
        compat="override",
        data_vars="minimal",
        coords="minimal",
        chunks={"time": 200, "nlon": 50, "nlat": 50},
    )

    pop1 = (
        preprocess_pop_dataset(pop1_)
        .assign_coords(cycle=("time", np.repeat(np.arange(6), 12 * 61)))
        .set_xindex("cycle")
    )
    return pop1


def subset_1deg_to_natre(ds):
    if "nlon" in ds.dims:
        return ds.cf.isel(nlon=slice(7, 17), nlat=slice(265, 276))
    else:
        return ds.cf.isel(X=slice(7, 17), Y=slice(265, 276))


def preprocess_pop_dataset(ds):
    ds = ds.drop_vars(["lat_aux_grid"], errors="ignore")
    ds["time"] = ds.time - xr.coding.cftime_offsets.MonthBegin(1)
    ds["σ"] = pop_tools.eos(ds.SALT, ds.TEMP, depth=xr.full_like(ds.z_t, 2000)) - 1000
    ds["σ"].attrs["long_name"] = "$σ_2$"
    ds.σ.attrs["grid_loc"] = ds.TEMP.attrs["grid_loc"]
    return ds


def gridder(da, grid, target):
    if "cycle" in da.dims:
        interped = xr.concat(
            [
                grid.transform(
                    da.sel(cycle=cycle),
                    axis="Z",
                    target=target,
                    target_data=da.σ.sel(cycle=cycle),
                    method="linear",
                )
                for cycle in da.cycle
            ],
            dim="cycle",
            join="exact",
        )
    else:
        interped = grid.transform(
            da, axis="Z", target=target, target_data=da.σ, method="linear"
        )

    interped.attrs = da.attrs
    return interped


def regrid_to_density(xds, grid, bins, varnames):
    import dask

    first_var = xds[varnames[0]]
    zdata = xds.cf["Z"].data
    if dask.base.is_dask_collection(xds):
        zdata = dask.array.from_array(zdata, chunks=xds.cf.chunks["Z"])
        kwargs = {"chunks": first_var.data.chunks}
        xp = dask.array
    else:
        kwargs = {}
        xp = np

    xds["z_σ"] = xr.DataArray(
        xp.broadcast_to(
            zdata[np.newaxis, :, np.newaxis, np.newaxis], first_var.shape, **kwargs
        ),
        dims=first_var.dims,
        coords=first_var.coords,
        attrs={"axis": "Z", "positive": "down", "units": "centimeters"},
    )

    regridded = (
        xds.pint.dequantify()
        .set_coords("σ")[["z_σ", *varnames]]
        .map(gridder, grid=grid, target=bins)
    )

    for var in regridded.variables:
        if var in xds:
            regridded[var].attrs = xds[var].attrs
            try:
                regridded[var].attrs["units"] = str(xds[var].data.units)
            except AttributeError:
                pass
    regridded = regridded.pint.quantify()

    # linear interpolating to surfaces. No bounds!
    # regridded["sigma_bounds"] = (
    #   ("bounds", "σ"), np.stack([bins[:-1], bins[1:]]), regridded.σ.attrs.copy()
    # )
    # regridded.σ.attrs["bounds"] = "sigma_bounds"
    regridded.σ.attrs.update({"axis": "Z", "positive": "down"})

    return regridded


def estimate_redi_terms(xds, grid, bins=None):
    if bins is not None:
        regridded = regrid_to_density(
            xds, grid, bins, [v for v in ["TEMP", "SALT", "KAPPA_ISOP"] if v in xds]
        )

    if "yearmonth" in xds:
        regridded["yearmonth"] = xds.yearmonth

    dTdy = grid.derivative(regridded.TEMP, axis="Y")
    dTdx = grid.derivative(regridded.TEMP, axis="X")
    regridded["delT2"] = (
        grid.interp(dTdx, axis="X") ** 2 + grid.interp(dTdy, axis="Y") ** 2
    ).cf.chunk({"X": -1, "Y": -1})
    regridded["delT2"].attrs = {"long_name": "$|∇_ρT|^2$"}

    if "KAPPA_ISOP" in xds:
        regridded["RediVar"] = regridded.KAPPA_ISOP * regridded.delT2
        regridded.update(regridded[["KAPPA_ISOP"]].map(to_base_units))
        regridded["KAPPA_ISOP"].attrs = {"long_name": "$K_{redi}$"}
        regridded["RediVar"].attrs = {"long_name": "$K_{redi} |∇_ρT|^2$"}

    regridded.time.attrs.clear()
    regridded.coords.update(grid._ds[pop_metric_vars])

    return regridded


def calc_mean_redivar_profile(ds):
    profile = ds.cf.mean(["X", "Y"])
    T = ds.cf["sea_water_potential_temperature"]
    reduce_dims = [T.cf.axes[ax][0] for ax in ["X", "Y"]]
    profile["delT2_plane"] = plane_fit_gradient(
        T.pint.dequantify(), reduce_dims=reduce_dims, debug=False
    )

    profile.coords["z_σ"] = profile.z_σ.cf.ffill("Z")
    profile = profile.cf.add_bounds("z_σ", dim="Z")
    return profile


def get_edges(pop):
    ybounds = pop.z_σ_bounds
    bdim = pop.cf.get_bounds_dim_name("z_σ")
    yedges = np.append(ybounds.isel({bdim: 0}).data, ybounds.data[-1, -1])
    return yedges
