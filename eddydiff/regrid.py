#!/usr/bin/env python3

import numba
import xarray as xr

"""
Classes and functions for working with vertical grids.
"""

import numpy as np
from numba import jit, guvectorize, float32, float64, boolean


@guvectorize(
    [
        (float64[:], float64[:], float64[:], boolean, float64[:]),
        (float32[:], float32[:], float32[:], boolean, float32[:]),
    ],
    "(n),(n),(m),()->(m)",
    nopython=True,
)
def _interp_1d_linear(phi, theta, target_theta_levels, mask_outcrops, output):
    output[:] = np.interp(target_theta_levels, theta, phi)

    if mask_outcrops:
        theta_max = np.nanmax(theta)
        theta_min = np.nanmin(theta)
        for i in range(len(target_theta_levels)):
            theta_lev = target_theta_levels[i]
            if (theta_lev < theta_min) or (theta_lev > theta_max):
                output[i] = np.nan


def interp_1d_linear(phi, theta, target_theta_levels, mask_outcrops=False):
    """
    Vectorized interpolation of scalar phi to isosurfaces of scalar theta
    along the final axis.

    Parameters
    ----------
    phi : array_like
        Array of shape (..., n), scalar field to be interpolated
    theta : array_like
        Array of shape (..., n), scalar field which defines the isosurfaces
    target_theta_levels : array_like
        Array of shape (m) specificying target isosurface levels
    mask_outcrops : bool, optional
        Determines how to handle theta values that exceed the bounds of
        target_theta_levels. If False, fill with nearest valid values. If
        True, fill with NaNs.

    Returns
    -------
    phi_interp : array
        Array of shape (..., m) of phi interpolated to theta isosurfaces.
    """
    return _interp_1d_linear(phi, theta, target_theta_levels, mask_outcrops)


@guvectorize(
    [
        (float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]),
        (float32[:], float32[:], float32[:], float32[:], float32[:], float32[:]),
    ],
    "(n),(n),(n),(m),(m)->(m)",
    nopython=True,
)
def _interp_1d_conservative(phi, theta_1, theta_2, theta_hat_1, theta_hat_2, output):
    output[:] = 0

    n = len(theta_1)
    m = len(theta_hat_1)

    for i in range(n):
        # handle non-monotonic stratification
        if theta_1[i] < theta_2[i]:
            theta_min = theta_1[i]
            theta_max = theta_2[i]
        else:
            theta_min = theta_2[i]
            theta_max = theta_1[i]

        for j in range(m):
            if (theta_hat_1[j] > theta_max) or (theta_hat_2[j] < theta_min):
                # there is no overlap between the cell and the bin
                pass
            else:
                # from here on there is some overlap
                theta_hat_min = max(theta_min, theta_hat_1[j])
                theta_hat_max = min(theta_max, theta_hat_2[j])
                alpha = (theta_hat_max - theta_hat_min) / (theta_max - theta_min)
                # now assign based on this weight
                output[j] += alpha * phi[i]


def interp_1d_conservative(phi, theta, target_theta_bins):
    """
    Accumulate extensive cell-centered quantity phi into new vertical coordinate
    defined by scalar theta.

    Parameters
    ----------
    phi : array_like
        Array of shape (..., n) defining an extensive quanitity in a cell
        bounded by two vertices.
    theta : array_like
        Array of shape (..., n+1) giving values of scalar theta  on the
        cell vertices. Phi is assumed to vary linearly between vertices.
    target_theta_bins : array_like
        Array of shape (m) defining the bounds of bins in which to accumulate
        phi.

    Returns
    -------
    phi_accum : array_like
        Array of shape (..., m-1) giving the values of phi partitioned into
        specified theta bins.
    """

    assert phi.shape[-1] == (theta.shape[-1] - 1)
    # assert target_theta_bins.ndim == 1
    # assert all(np.diff(target_theta_bins) > 0)

    theta_1 = theta[..., :-1]
    theta_2 = theta[..., 1:]
    # theta_hat_1 = target_theta_bins[:-1]
    # theta_hat_2 = target_theta_bins[1:]

    theta_1 = theta[..., :-1]
    theta_2 = theta[..., 1:]
    #     theta_hat_1 = target_theta_bins[:-1]
    #     theta_hat_2 = target_theta_bins[1:]
    theta_hat_1 = target_theta_bins[..., :-1]
    theta_hat_2 = target_theta_bins[..., 1:]

    return _interp_1d_conservative(phi, theta_1, theta_2, theta_hat_1, theta_hat_2)


def wrapper_linear(data, target_data, target, dim):
    return xr.apply_ufunc(
        interp_1d_linear,
        data,
        target_data,
        target,
        kwargs={"mask_outcrops": True},
        input_core_dims=[[dim], [dim], ["target"]],
        output_core_dims=[["target"]],
        dask="parallelized",
        output_dtypes=[target_data.dtype],
    )


def wrapper_conservative(data, source_bounds, target_bounds, dim):
    return xr.apply_ufunc(
        interp_1d_conservative,
        data,
        source_bounds,
        target_bounds,
        input_core_dims=[[dim], [f"{dim}_bounds"], ["target"]],
        output_core_dims=[["remapped"]],
        dask="parallelized",
        output_dtypes=[data.dtype],
        output_sizes={"remapped": len(target_bounds.target) - 1},
    )


def remap_full(ds, target_data, target, dim, target_kind="center"):

    target_labels = target.isel(target=slice(0, -1))
    target_labels.data = (
        target.isel(target=slice(1, None)).data + target.isel(target=slice(0, -1)).data
    ) / 2

    remapped_dict = {}

    z_regrid = wrapper_linear(ds[dim], target_data, target, dim)

    # calculate the layer thickness of the new coord bounds
    dz_target = (
        z_regrid.diff("target")
        .rename({"target": "remapped"})
        .drop("remapped", errors="ignore")
    )

    bounds = f"{dim}_bounds"
    bdim = list(set(ds[bounds].dims) - set([dim]))[0]

    # Construct the source bounding coordinates from the `lev_bounds` coordinate
    source_bounds = xr.DataArray(
        np.hstack(
            [
                ds[bounds].isel({bdim: 0}).load().data,
                ds[bounds].isel({dim: -1, bdim: 1}).load().data,
            ]
        ),
        dims=bounds,
    )

    dz = ds[dim].diff(dim).reindex({dim: ds[dim]}).ffill(dim)

    for var in ds.data_vars:
        # for now treat them all as intensive...
        remapped_dict[var] = (
            wrapper_conservative(ds[var] * dz, source_bounds, z_regrid, dim) / dz_target
        )

    out = xr.Dataset(remapped_dict)

    # assign actual coordinate values
    out = out.assign_coords(remapped=target_labels.rename({"target": "remapped"}))

    # carry the layer bounds and the layer thickness as coordinates

    out = out.assign_coords(dz_remapped=dz_target)  # not the best naming

    # rename according to the target data name
    if target_data.name is not None:
        out = out.rename({"remapped": f"{target_data.name}_bins"})

    return out
