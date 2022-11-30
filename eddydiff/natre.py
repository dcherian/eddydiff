import glob

import dcpy
import matplotlib.pyplot as plt
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
    ds = ds.cf.guess_coord_axis()
    ds["gamma_n"] = dcpy.oceans.neutral_density(ds)
    ds.load().to_netcdf("../datasets/natre_large_scale.nc")
    return ds


def read_natre(load=False, stack=False, sort=True):
    if load:
        natre = xr.load_dataset("../datasets/natre_large_scale.nc")
    else:
        natre = xr.open_dataset(
            "../datasets/natre_large_scale.nc", chunks={"latitude": 5, "longitude": 5}
        )
    # natre = natre.where(natre.chi.notnull() & natre.eps.notnull())
    # natre = combine_natre_files()
    natre = natre.set_coords(["time", "pres"])

    natre.chi.attrs["long_name"] = "$χ$"
    natre.eps.attrs["long_name"] = "$ε$"
    del natre["depth"].attrs["axis"]
    natre["depth"].attrs.update(units="m", positive="down")
    natre["pres"].attrs.update(positive="down")

    assert "neutral_density" in natre.cf

    if sort:
        natre = dcpy.oceans.thorpesort(
            natre, natre.gamma_n.drop("depth").interpolate_na("pres"), core_dim="pres"
        )

    sections.add_ancillary_variables(natre)
    natre = natre.where(natre.chi > 1e-14)

    # messes up cf-xarray
    if "depth" in natre.time.dims:
        natre["time"] = natre.time.isel(depth=0)

    # messes up pint
    del natre.salt.attrs["units"]

    natre = natre.sel(pres=slice(2000))

    natre = natre.update(
        sections.estimate_microscale_stirring_depth_space(
            natre, filter_len=20, segment_len=6
        )
    )

    if stack:
        natre = natre.cf.stack({"cast": ("latitude", "longitude")})
        natre.cast.attrs = {"cf_role": "profile_id"}
    return natre


def compare_chi_distributions(a05_grouped, natre_grouped):
    from eddydiff.sections import compute_bootstrapped_mean_ci

    binsize = 20
    for label, group in a05_grouped:
        print(label)
        group.attrs["name"] = "A05"
        natre_group = natre_grouped[label]
        natre_group.attrs["name"] = "NATRE"

        f, ax = plt.subplots(1, 2, sharey=True, constrained_layout=True)

        for group_, yerr in zip((group, natre_group), [0.5, 0.6]):
            mask = (group_.chi < 1e-7) & (group_.eps < 1e-7)
            group_ = group_.pint.quantify().pint.dequantify("~P")

            for varname, axx in zip(["chi", "eps"], ax):

                kwargs = dict(ax=axx, density=True, histtype="step", lw=1.5)

                var = group_[varname]
                npts = var.notnull().sum().item()
                err = compute_bootstrapped_mean_ci(var.to_numpy(), binsize)
                _, histbins, hdl = np.log10(var.as_numpy()).plot.hist(
                    bins=101, label=f"{group_.attrs['name']}, {npts} pts", **kwargs
                )

                var = group_[varname].where(mask)
                npts = var.notnull().sum().item()
                mask_err = compute_bootstrapped_mean_ci(
                    var.where(mask).to_numpy(), binsize
                )
                _, _, hdl1 = np.log10(var.as_numpy()).plot.hist(
                    bins=histbins,
                    label=f"{group_.attrs['name']}, masked,  {npts} pts",
                    **kwargs,
                )

                ecolors = (hdl[0].get_edgecolor(), hdl1[0].get_edgecolor())
                for e, dy, c in zip([err, mask_err], [-0.01, +0.01], ecolors):
                    axx.plot(
                        np.log10(e), np.ones((3,)) * (yerr + dy), color=c, marker="."
                    )
                axx.set_xticks(np.arange(-13, -5, 1))
                axx.set_title("")
        axx.legend(bbox_to_anchor=(1, 1))
        f.suptitle(
            f"$γ_n$ bin = {label} | depth bin = {group.pressure.mean().item():.2f}, {natre_group.pres.mean().item():.2f}"
        )
        f.set_size_inches((8, 3))


def plot_terms(ds, *, ax, x, y, color, **kwargs):
    dcpy.plots.fill_between_bounds(ds, x, color=color, y=y, ax=ax)


def plot_lines(ds, col):
    fg = xr.plot.FacetGrid(ds, col=col, col_wrap=5, aspect=1 / 1.8)
    fg.map_dataset(plot_terms, x="chib2", hue=None, y="pres", color="r")
    fg.map_dataset(plot_terms, x="KρTz2", hue=None, y="pres", color="k")
    ax = fg.axes[0, 0]
    ax.set_ylim((2000, 200))
    ax.set_xlim(1e-11, 1e-8)
    ax.set_xscale("log")

    return fg
