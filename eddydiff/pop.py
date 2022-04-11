import cf_xarray  # noqa
import numpy as np
import pop_tools
from dcpy.util import to_base_units

import xarray as xr

pop_metric_vars = ["UAREA", "TAREA", "DXU", "DXT", "DYU", "DYT"]

metrics = {
    ("X",): ["DXU", "DXT"],  # X distances
    ("Y",): ["DYU", "DYT"],  # Y distances
    # ("Z",): ["DZU", "DZT"],  # Z distances
    ("X", "Y"): ["UAREA", "TAREA"],
}


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
    interped.attrs = da.attrs
    return interped


def regrid_to_density(xds, grid, bins, varnames):

    import dask

    if dask.base.is_dask_collection(xds.TEMP.data):
        zdata = dask.array.from_array(xds.z_t.data, chunks=xds.TEMP.chunks[1])
        kwargs = {"chunks": xds.TEMP.data.chunks}
        xp = dask.array
    else:
        zdata = xds.z_t.data
        kwargs = {}
        xp = np

    xds["z_σ"] = xr.DataArray(
        xp.broadcast_to(
            zdata[np.newaxis, :, np.newaxis, np.newaxis], xds.TEMP.shape, **kwargs
        ),
        dims=xds.TEMP.dims,
        coords=xds.TEMP.coords,
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
    regridded.σ.attrs["axis"] = "Z"

    return regridded


def estimate_redi_terms(xds, grid, bins):
    regridded = regrid_to_density(
        xds, grid, bins, [v for v in ["TEMP", "SALT", "KAPPA_ISOP"] if v in xds]
    )

    if "yearmonth" in xds:
        regridded["yearmonth"] = xds.yearmonth

    dTdy = grid.derivative(regridded.TEMP, axis="Y")
    dTdx = grid.derivative(regridded.TEMP, axis="X")
    regridded["delT2"] = (
        grid.interp(dTdx, axis="X") ** 2 + grid.interp(dTdy, axis="Y") ** 2
    )
    regridded["delT2"].attrs = {"long_name": "$|∇T|^2$"}

    if "KAPPA_ISOP" in xds:
        regridded["RediVar"] = regridded.KAPPA_ISOP * regridded.delT2
        regridded.update(regridded[["KAPPA_ISOP"]].map(to_base_units))
        regridded["KAPPA_ISOP"].attrs = {"long_name": "$K_{redi}$"}
        regridded["RediVar"].attrs = {"long_name": "$K_{redi} |∇T|^2$"}

    regridded.time.attrs.clear()
    regridded.coords.update(grid._ds[pop_metric_vars])

    return regridded
