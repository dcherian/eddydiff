# Field experiment regression tests

import pandas as pd
import numpy as np
import xarray as xr
import eddydiff as ed


def test_interval_roundtrip():
    v = np.arange(4.0)
    data = [pd.Interval(left, right) for left, right in zip(v[:-1], v[1:])]
    da = xr.DataArray(
        data, name="gamma_n_bins", coords={"gamma_n_bins": data}, dims="gamma_n_bins"
    )
    encoded = ed.intervals_to_vertex(da)
    decoded = ed.intervals_from_vertex(encoded)
    xr.testing.assert_allclose(da, decoded)


def test_natre():
    natre = xr.open_dataset(
        "datasets/natre_large_scale.nc", chunks={"latitude": 5, "longitude": 5}
    )
    natre = natre.where(natre.chi.notnull() & natre.eps.notnull())
    natre = natre.set_coords(["time", "pres"])
    natre = natre.cf.guess_coord_axis()
    natre["depth"].attrs.update(units="m", positive="down")

    natre = ed.sections.add_ancillary_variables(natre)
    natre = natre.where(natre.chi > 1e-14)
    natre.load()

    bins = (
        natre.gamma_n.mean(["latitude", "longitude"])
        .interp(depth=np.arange(150, 2001, 100))
        .dropna("depth")
        .data
    )
    actual = ed.sections.bin_average_vertical(
        natre.reset_coords("pres"), "neutral_density", bins
    )

    # chidens.assign(gamma_n_bounds=ed.intervals_to_vertex(chidens.gamma_n_bins)).drop("gamma_n_bins").to_netcdf("../tests/estimates/natre.nc")
    expected = xr.open_dataset("tests/estimates/natre.nc")
    expected = expected.assign(
        gamma_n_bins=ed.intervals_from_vertex(expected.gamma_n_bounds)
    ).drop_vars("gamma_n_bounds")
    xr.testing.assert_allclose(expected, actual)
