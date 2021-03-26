import glob

import numpy as np

import xarray as xr


def preprocess_natre(ds):
    def _regrid(ds):
        start = 10
        step = 0.4
        rounded = np.arange(start, start + step * 7962, step)
        return ds.groupby_bins(
            "DEPTH", rounded, labels=0.5 * (rounded[:-1] + rounded[1:])
        )

    ds.load()

    count = _regrid(ds["CHI-T"]).count()
    if not (count.dropna("DEPTH_bins") < 2).all():
        raise ValueError(
            f"Regridding may be wrong for file: {ds.encoding['source'].split('/')[-1]}"
        )

    ds["LATITUDE"] = np.round(ds.LATITUDE.data, 1)
    ds["LONGITUDE"] = np.round(ds.LONGITUDE.data, 1)

    if ds.LONGITUDE.data > -26.9:
        ds["LONGITUDE"] = -26.8

    return (
        _regrid(ds)
        .first()
        .rename(
            {
                "DEPTH_bins": "depth",
                "LATITUDE": "latitude",
                "LONGITUDE": "longitude",
                "CHI-T": "chi",
                "EPSILON": "eps",
                "TEMPERATURE": "temp",
                "PSAL": "salt",
                "PRESSURE": "pres",
                "TIME": "time",
            }
        )
        .squeeze()
        .reset_coords("time")
        .expand_dims(["latitude", "longitude"])
    )


def filenum(filename):
    """ Gets file number from microstructure file name"""
    return int(filename.split("/")[-1][6:].split(".")[0])


def combine_natre_files():

    files = glob.glob(
        "/home/deepak/datasets/microstructure/natre_microstructure/natre_*.nc"
    )
    files = sorted(files, key=filenum)
    nested = np.reshape(files[:100], (10, 10))
    for a in np.arange(1, 10, 2):
        nested[a, :] = np.flip(nested[a, :])

    ds = xr.open_mfdataset(
        nested.tolist(),  # this is not numpy array friendly
        combine="nested",
        concat_dim=["longitude", "latitude"],  # order matters
        preprocess=preprocess_natre,
        parallel=True,
    )
    ds.to_netcdf("../datasets/natre_large_scale.nc")
