# Field experiment regression tests

import cf_xarray as cfxr
import numpy as np
import pandas as pd

import eddydiff as ed
import xarray as xr

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
    natre = xr.open_dataset(
        "datasets/natre_large_scale.nc", chunks={"latitude": 5, "longitude": 5}
    )
    natre = natre.where(natre.chi.notnull() & natre.eps.notnull())
    natre = natre.set_coords(["time", "pres"])
    natre = natre.cf.guess_coord_axis()
    natre["depth"].attrs.update(units="m", positive="down")

    natre = ed.sections.add_ancillary_variables(natre, pref=1000)
    natre = natre.where(natre.chi > 1e-14)
    natre.load()

    bins = ed.sections.choose_bins(natre.gamma_n, depth_range=np.arange(150, 2001, 100))
    actual = ed.sections.bin_average_vertical(
        natre.reset_coords("pres"), "neutral_density", bins
    )

    expected = xr.open_dataset("tests/estimates/natre.nc")
    xr.testing.assert_allclose(expected, actual)
