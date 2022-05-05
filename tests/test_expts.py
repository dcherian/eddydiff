# Field experiment regression tests

import os

import cf_xarray as cfxr
import distributed
import numpy as np
import pandas as pd

import eddydiff as ed
import xarray as xr

# from distributed.utils_test import client, cluster_fixture, loop
from eddydiff.natre import read_natre
from xarray.testing import assert_allclose
from xarray.tests import raise_if_dask_computes

xr.set_options(keep_attrs=True)


def test_interval_roundtrip():
    v = np.arange(4.0)
    data = [pd.Interval(left, right) for left, right in zip(v[:-1], v[1:])]
    da = xr.DataArray(
        data, name="gamma_n_bins", coords={"gamma_n_bins": data}, dims="gamma_n_bins"
    )
    bounds = ed.intervals_to_bounds(da)
    encoded = cfxr.bounds_to_vertices(bounds, bounds_dim="bounds")
    decoded = ed.intervals_from_vertex(encoded).rename(
        {"gamma_n_v_bins": "gamma_n_bins"}
    )
    xr.testing.assert_allclose(da, decoded)


def test_natre():

    os.chdir("./notebooks/")

    with distributed.Client(
        n_workers=2,
        threads_per_worker=3,
        env={"OMP_NUM_THREADS": 1, "NUMBA_NUM_THREADS": 1, "MKL_NUM_THREADS": 1},
    ) as client:

        natre = read_natre().load(client=client)
        bins = ed.sections.choose_bins(
            natre.gamma_n, depth_range=np.arange(150, 2001, 100)
        )
        with raise_if_dask_computes():
            actual = ed.sections.bin_average_vertical(
                natre.cf.stack(
                    {"cast": ("latitude", "longitude")}, create_index=False
                ).assign_coords(
                    cast=("cast", np.arange(100), {"cf_role": "profile_id"})
                ),
                "neutral_density",
                bins,
                blocksize=20,
            )
        actual.load(client=client)
        actual.attrs.pop("commit", None)

        expected = xr.load_dataset("../tests/estimates/natre.nc")
        expected.attrs.pop("commit", None)
        assert_allclose(expected, actual, atol=0, rtol=1e-5)
