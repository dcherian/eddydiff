import glob

import dcpy
import numpy as np

import xarray as xr

from . import sections


def preprocess_natre(ds):
    def _regrid(ds):
        start = 10
        step = 0.5
        # There is one 3000m profile...
        rounded = np.arange(start, 3100, step)
        # pressure is on a regular 0.5dbar grid
        regridded = (
            ds.dropna("DEPTH")
            .swap_dims({"DEPTH": "PRESSURE"})
            .reindex(PRESSURE=rounded, method="nearest", tolerance=0.1)
        )
        assert (
            regridded.EPSILON.count().compute().data
            == ds.EPSILON.count().compute().data
        )
        return regridded

        # return ds.groupby_bins(
        #    "DEPTH", rounded, labels=0.5 * (rounded[:-1] + rounded[1:])
        # )

    # ds.load()

    ds["LATITUDE"] = np.round(ds.LATITUDE.data, 1)
    ds["LONGITUDE"] = np.round(ds.LONGITUDE.data, 1)

    if ds.LONGITUDE.data > -26.9:
        ds["LONGITUDE"] = -26.8

    regridded = _regrid(ds)

    # count = _regrid(ds["CHI-T"]).count()
    # if not (count.dropna("DEPTH_bins") < 2).all():
    #    raise ValueError(
    #        f"Regridding may be wrong for file: {ds.encoding['source'].split('/')[-1]}"
    #    )
    # regridded = _regrid(ds).first().rename({"DEPTH_bins": "depth"})
    # regridded = ds.drop_vars("DEPTH").rename({"DEPTH": "idepth"})
    # regridded["depth"] = ("idepth", ds.DEPTH.data)

    return (
        regridded.rename(
            {
                "LATITUDE": "latitude",
                "LONGITUDE": "longitude",
                "CHI-T": "chi",
                "EPSILON": "eps",
                "TEMPERATURE": "temp",
                "PSAL": "salt",
                "PRESSURE": "pres",
                "TIME": "time",
                "DEPTH": "depth",
            }
        )
        .squeeze()
        .reset_coords("time")
        .expand_dims(["latitude", "longitude"])
    )


def filenum(filename):
    """Gets file number from microstructure file name"""
    return int(filename.split("/")[-1][6:].split(".")[0])


def combine_natre_files():

    import os

    home = os.path.expanduser("~/")
    files = glob.glob(f"{home}/datasets/microstructure/natre_microstructure/natre_*.nc")
    files = sorted(files, key=filenum)
    # First 100 are stations 3-102; part of the large-scale survey
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
    ds.load().to_netcdf("../datasets/natre_large_scale.nc")
    return ds


def read_natre():
    natre = xr.open_dataset(
       "../datasets/natre_large_scale.nc", chunks={"latitude": 5, "longitude": 5}
    )
    # natre = natre.where(natre.chi.notnull() & natre.eps.notnull())
    # natre = combine_natre_files()
    natre = natre.set_coords(["time", "pres"])

    natre = natre.cf.guess_coord_axis()
    natre.chi.attrs["long_name"] = "$χ$"
    natre.eps.attrs["long_name"] = "$ε$"
    natre["depth"].attrs.update(units="m", positive="down")
    natre["pres"].attrs.update(positive="down")

    if "neutral_density" not in natre.cf:
        natre["gamma_n"] = dcpy.oceans.neutral_density(natre)
    natre = dcpy.oceans.thorpesort(
        natre, natre.gamma_n.drop("depth").interpolate_na("pres"), core_dim="pres"
    )

    natre = sections.add_ancillary_variables(natre, pref=1000)
    natre = natre.where(natre.chi > 1e-14)

    # messes up cf-xarray
    if "depth" in natre.time.dims:
        natre["time"] = natre.time.isel(depth=0)

    # messes up pint
    del natre.salt.attrs["units"]

    return natre
