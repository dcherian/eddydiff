import matplotlib.pyplot as plt
import numpy as np


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
