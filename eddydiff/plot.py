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
    subset.salinity_gradient.cf.plot(**kwargs, ax=ax[0], add_legend=False)
    subset.salinity_std.cf.plot(**kwargs, ax=ax[1], add_legend=False)
    subset.mixing_length.cf.plot(**kwargs, ax=ax[2], add_legend=False)
    subset.diffusivity.cf.plot(**kwargs, ax=ax[3], add_legend=False)

    dcpy.plots.clean_axes(ax)
    [axx.set_title("") for axx in ax]
    f.suptitle(sel_kwargs)
