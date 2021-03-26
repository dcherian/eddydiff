# from . import tests
import xarray as xr

from . import jmd95, plot, regrid, sections  # noqa
from .eddydiff import *  # noqa


def intervals_to_vertex(da):
    assert da.ndim == 1
    name = da.dims[0][:-5]
    vertex = [interval.left for interval in da.data]
    vertex.append(da.data[-1].right)
    return xr.DataArray(vertex, dims=f"{name}_bounds", name=f"{name}_bounds")


def intervals_from_vertex(vertex):
    import pandas as pd

    assert vertex.ndim == 1
    name = f"{vertex.dims[0][:-7]}_bins"
    data = [
        pd.Interval(left, right)
        for left, right in zip(vertex.data[:-1], vertex.data[1:])
    ]
    return xr.DataArray(data, dims=name, name=name, coords={name: data})
