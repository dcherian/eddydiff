import dcpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .eddydiff import read_cole


def check_bins_with_climatology(bins, argo, ecco):
    """
    Given bins, makes plots to decide whether these are a good choice for the climatology.
    E.g. check whether the bins outcrop or not, that they are not too closely spaced together.
    """
    centers = (bins[1:] + bins[:-1]) / 2

    f, axx = plt.subplots(
        3, 1, sharex=True, sharey=True, constrained_layout=True, squeeze=False
    )

    for ds, ax in zip([argo, ecco], axx.flat):
        ds.pden.cf.plot.contourf(
            y="Z",
            ax=ax,
            levels=np.arange(1022, 1027, 0.1),
            cmap=mpl.cm.RdYlBu_r,
            add_colorbar=False,
            ylim=(200, 0),
        )
        ds.pden.cf.plot.contour(y="Z", colors="k", levels=bins, ax=ax)
        ds.pden.cf.plot.contour(
            y="Z", colors="k", linestyles="--", levels=centers, ax=ax
        )
        dcpy.plots.linex(0, zorder=10, color="w", ax=ax)
        ax.set_title(ds.dataset)

    hargo = argo.pden.plot.contour(levels=bins, colors="r", ax=axx.flat[-1])
    hecco = ecco.pden.plot.contour(
        levels=bins, colors="k", ylim=(180, 0), ax=axx.flat[-1]
    )
    dcpy.plots.add_contour_legend(hargo, "argo", numel=1, loc="upper right")
    dcpy.plots.add_contour_legend(hecco, "ecco", numel=1, loc="upper left")
    dcpy.plots.linex(0, ax=axx.flat[-1])

    dcpy.plots.clean_axes(axx)
    axx[1, 0].set_title("ecco")
    f.set_size_inches((5, 8))


def histogram_turb_estimates(ds):
    f, ax = plt.subplots(2, 2, sharex=False, constrained_layout=True)
    kwargs = dict(bins=250, histtype="step", density=False)

    plt.sca(ax[0, 0])
    np.log10(ds.chi).plot.hist(**kwargs)
    np.log10(ds.chi_masked).plot.hist(**kwargs)
    plt.legend(["$χ$", "$χ_{T_z}^{mask}$"])

    if "eps" in ds:
        plt.sca(ax[0, 1])
        np.log10(ds.eps).plot.hist(**kwargs)
        np.log10(ds.eps_chi).plot.hist(**kwargs)
        plt.legend(["$ε$", "$ε_χ$"])

    plt.sca(ax[1, 0])
    np.log10(ds.Kt).plot.hist(**kwargs)
    if "Krho" in ds:
        np.log10(ds.Krho).plot.hist(**kwargs)
    plt.legend(["$K_T$", "$K_ρ$"])

    plt.sca(ax[1, 1])
    np.log10(ds.KtTz).plot.hist(**kwargs)
    if "Krho" in ds:
        np.log10(ds.KrhoTz).plot.hist(**kwargs)
    plt.legend(["$K_T θ_z$", "$K_ρ θ_z$"])

    [aa.set_title("") for aa in ax.flat]


def plot_cole_profile(**sel_kwargs):
    cole = read_cole()
    subset = cole.sel(**sel_kwargs, method="nearest")

    kwargs = dict(hue="lon", y="vertical", xscale="log", yincrease=False)

    f, ax = plt.subplots(1, 4, sharey=True, constrained_layout=True)
    subset.salinity_gradient.cf.plot(**kwargs, ax=ax[0], add_legend=True)
    subset.salinity_std.cf.plot(**kwargs, ax=ax[1], add_legend=False)
    subset.mixing_length.cf.plot(**kwargs, ax=ax[2], add_legend=False)
    subset.diffusivity.cf.plot(**kwargs, ax=ax[3], add_legend=False)

    dcpy.plots.clean_axes(ax)
    [axx.set_title("") for axx in ax]
    f.suptitle(sel_kwargs)


def compare_section_estimates(averages, finescale=None, KρTz2=False, colors=None):
    if colors is None:
        colors = [f"C{N}" for N in range(len(averages))]
    f, ax = plt.subplots(1, 4, sharey=True)

    for avg, color in zip(averages, colors):
        project = avg.attrs.get("title", None)
        for var, axx in zip(["dTdz_m", "N2_m", "chi", "eps"], ax):
            dcpy.plots.fill_between_bounds(
                avg, var, y="pres", ax=axx, label=project, title=True, color=color
            )

    ax[2].set_xscale("log")
    ax[3].set_xscale("log")

    # dcpy.plots.fill_between_bounds(avg, "Krho_m", y="pres", ax=ax[3])
    # dcpy.plots.fill_between_bounds(avg, "Kt_m", y="pres", ax=ax[3])
    # ax[3].set_xscale("log")

    # dcpy.plots.fill_between_bounds(avg, "chib2", y="pres", ax=ax[4])
    # # dcpy.plots.fill_between_bounds(avg, "KtTzTz", y="pres", ax=ax[4])
    # dcpy.plots.fill_between_bounds(avg, "KtTz~Tz", y="pres", ax=ax[4])
    # if finescale is not None:
    #     finescale.plot.line(hue="criteria", y="pressure", ax=ax[4], _labels=False)
    # if KρTz2:
    #     dcpy.plots.fill_between_bounds(avg, "KρTz2", y="pres", color="k", ax=ax[4])
    # ax[4].set_xlim([1e-10, 2 * avg.chib2.max().item()])
    # ax[4].set_xscale("log")

    [axx.legend(loc="lower right") for axx in ax]
    plt.gcf().set_size_inches((14, 4))


def debug_section_estimate(avg, finescale=None, KρTz2=False):
    f, ax = plt.subplots(1, 5, sharey=True)
    dcpy.plots.fill_between_bounds(avg, "dTdz_m", y="pres", ax=ax[0])
    dcpy.plots.fill_between_bounds(avg, "N2_m", y="pres", ax=ax[0].twiny(), color="C1")

    dcpy.plots.fill_between_bounds(avg, "hm", y="pres", ax=ax[1])
    ax[1].set_xlim([0.9 * avg.hm.min().item(), 1.1 * avg.hm.max().item()])

    dcpy.plots.fill_between_bounds(avg, "chi", y="pres", ax=ax[2])
    dcpy.plots.fill_between_bounds(avg, "eps", y="pres", ax=ax[2])
    ax[2].set_xscale("log")

    dcpy.plots.fill_between_bounds(avg, "Krho_m", y="pres", ax=ax[3])
    dcpy.plots.fill_between_bounds(avg, "Kt_m", y="pres", ax=ax[3])
    ax[3].set_xscale("log")

    dcpy.plots.fill_between_bounds(avg, "chib2", y="pres", ax=ax[4])
    # dcpy.plots.fill_between_bounds(avg, "KtTzTz", y="pres", ax=ax[4])
    dcpy.plots.fill_between_bounds(avg, "KtTz~Tz", y="pres", ax=ax[4])
    if finescale is not None:
        finescale.plot.line(hue="criteria", y="pressure", ax=ax[4], _labels=False)
    if KρTz2:
        dcpy.plots.fill_between_bounds(avg, "KρTz2", y="pres", color="k", ax=ax[4])
    ax[4].set_xlim([1e-10, 2 * avg.chib2.max().item()])
    ax[4].set_xscale("log")

    [axx.legend(loc="lower right") for axx in ax]
    plt.gcf().set_size_inches((14, 4))


def plot_Tu_relationships(data, title=None):
    Tu_bins = [-90, -51, -45, 0, 45, 72, 90]

    # f, ax = plt.subplots(2, 2, constrained_layout=True)
    f, ax = plt.subplot_mosaic(
        [
            ["eps-profile", "Tu-count", "eps-dist", "eps-dist"],
            ["eps-profile", "Tu-count", "Tu-section", "eps-section"],
        ],
        constrained_layout=True,
        gridspec_kw=dict(width_ratios=[2, 1, 2, 2]),
    )

    kwargs = dict(
        y="Z",
        x="profile_id",
        cbar_kwargs={"orientation": "horizontal"},
    )
    data.Tu.cf.plot(ax=ax["Tu-section"], levels=Tu_bins, **kwargs)
    data.eps.cf.coarsen(Z=20, boundary="trim").mean().cf.plot(
        ax=ax["eps-section"],
        norm=mpl.colors.LogNorm(5e-10, 1e-7),
        **kwargs,
        cmap=mpl.cm.turbo,
    )

    gb = data.eps.groupby_bins(data.Tu, bins=Tu_bins)
    gb.count().plot.step(label="count", ax=ax["eps-dist"])
    ax["eps-dist"].set_ylabel("N obs")
    ax["eps-dist"].tick_params(axis="y", labelcolor="r")
    ax["eps-dist"].set_xticks(Tu_bins)

    ax2 = ax["eps-dist"].twinx()
    gb.mean().plot.step(label="mean", ax=ax2, yscale="log", color="k")
    ax2.set_ylabel("Mean ε")

    N = int(20 // np.median(np.diff(data.Tu.cf["Z"].data)))
    for left, right in [(72, 90), (50, 72), (0, 45), (-45, 0), (-90, -45)]:
        (
            data.eps.where((data.Tu > left) & (data.Tu < right))
            .cf.mean("profile_id")
            .cf.coarsen(Z=N, boundary="trim")
            .mean()
            .cf.plot(ax=ax["eps-profile"], xscale="log", label=f"{left} < Tu < {right}")
        )
        (
            data.eps.where((data.Tu > left) & (data.Tu < right))
            .cf.coarsen(Z=N, boundary="trim")
            .count()
            .cf.sum("profile_id")
            .cf.plot(ax=ax["Tu-count"], label=f"{left} < Tu < {right}")
        )

    ax["Tu-count"].set_xlabel("N obs")
    ax["Tu-count"].set_ylabel("")
    ax["Tu-count"].tick_params(axis="y", labelleft=False)
    ax["eps-profile"].set_xlabel("Mean ε")

    ax["Tu-count"].legend(bbox_to_anchor=(1, 1))
    # hl.set_in_layout(False)
    ax2.legend()
    ax["eps-dist"].legend()

    [axx.set_title("") for axx in ax.values()]
    ax2.set_title("")
    if title is not None:
        f.suptitle(title)
    f.set_size_inches((10, 5))


def make_nice_natre_map(ax_):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax_.add_patch(
        mpl.patches.Rectangle(
            [-31, 23.0],
            5,
            5,
            lw=2,
            transform=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="k",
        )
    )

    ax_.set_facecolor("w")
    ax_.set_extent([-35, -5, 15, 40])
    ax_.coastlines(lw=1.1, zorder=10)
    ax_.set_title("")
    ax_.add_feature(cfeature.LAND)
