import cf_xarray as cfxr  # noqa
import dcpy
import matplotlib.pyplot as plt
import numpy as np
from dcpy.finestructure import estimate_turb_segment

import xarray as xr


def trim_mld_mode_water(profile):
    """
    Follows Whalen's approach of using a threshold criterion first to identify MLD,
    trimming that; and then applying again to find mode water. I apply both T, σ criteria
    and pick the max depth.
    """

    def find_thresh_delta(delta, thresh):
        """Finds threshold in delta."""
        return delta.PRES.isel(PRES=delta > thresh)[0].data

    near_surf = profile[["σ_θ", "TEMP"]].sel(PRES=10, method="nearest")
    delta = np.abs(profile - near_surf)

    Tmld = find_thresh_delta(delta.TEMP, 0.2)
    σmld = find_thresh_delta(delta.σ_θ, 0.03)

    trimmed = profile.sel(PRES=slice(max(Tmld, σmld), None))

    delta = np.abs(trimmed - trimmed[["σ_θ", "TEMP"]].isel(PRES=0))
    Tmode = find_thresh_delta(delta.TEMP, 0.2)
    σmode = find_thresh_delta(delta.σ_θ, 0.03)

    return profile.sel(PRES=slice(max(Tmode, σmode), None)).assign_coords(
        Tmode=Tmode, σmode=σmode, Tmld=Tmld, σmld=σmld
    )


def results_to_xarray(results, profile):
    data_vars = {
        var: ("pressure", results[var])
        for var in [
            "ε",
            "Kρ",
            "N2mean",
            "ξvar",
            "ξvargm",
            "Tzlin",
            "Tzmean",
            "mean_dTdz_seg",
        ]
    }
    coords = {
        var: ("pressure", results[var]) for var in ["flag", "pressure", "npts", "γmean"]
    }
    coords.update(
        {
            "latitude": profile["LATITUDE"].data,
            "longitude": profile["LONGITUDE"].data,
            "γ_bounds": (("pressure", "nbnds"), results["γbnds"]),
            "p_bounds": (("pressure", "nbnds"), results["pbnds"]),
        }
    )

    turb = xr.Dataset(data_vars, coords)

    turb.ε.attrs = {"long_name": "$ε$", "units": "W/kg"}
    turb.Kρ.attrs = {"long_name": "$K_ρ$", "units": "m²/s"}
    turb.ξvar.attrs = {"long_name": "$ξ$", "description": "strain variance"}
    turb.ξvargm.attrs = {"long_name": "$ξ_{gm}$", "description": "GM strain variance"}
    turb.N2mean.attrs = {
        "long_name": "$N²$",
        "description": "mean of quadratic fit of N² with pressure",
    }
    turb.Tzlin.attrs = {
        "long_name": "$T_z^{lin}$",
        "description": "linear fit of T vs pressure",
    }
    turb.Tzmean.attrs = {
        "long_name": "$T_z^{quad}$",
        "description": "mean of quadratic fit of Tz with pressure; like N² fitting for strain",
    }
    turb.mean_dTdz_seg.attrs = {
        "description": "mean of dTdz values in segment",
        "long_name": "$⟨T_z⟩$",
    }
    turb.pressure.attrs = {
        "axis": "Z",
        "standard_name": "sea_water_pressure",
        "positive": "down",
        "bounds": "p_bounds",
    }
    turb.npts.attrs = {"description": "number of points in segment"}
    turb.γmean.attrs = {"bounds": "γ_bounds"}
    turb.γmean.attrs.update(profile.γ.attrs)

    turb.cf.guess_coord_axis()

    for var in ["Tmld", "σmld", "Tmode", "σmode"]:
        turb[var] = profile[var].data
        turb[var].attrs["units"] = "m"
        if "mld" in var:
            turb[var].attrs["description"] = f"Δ{var[0]} criterion applied first time"
        if "mode" in var:
            turb[var].attrs["description"] = f"Δ{var[0]} criterion applied second time"

    turb.flag.attrs = {
        "flag_values": [-2, -1, 1, 2, 3, 4, 5],
        "flag_meanings": "too_coarse too_short N²_variance_too_high too_unstratified too_little_bandwidth no_internal_waves good_data",
    }

    for var in [
        "CONFIG_MISSION_NUMBER",
        "PLATFORM_NUMBER",
        "CYCLE_NUMBER",
        "DIRECTION",
    ]:
        turb.coords[var] = profile[var].data

    turb["χ"] = 2 * turb.Kρ * turb.Tzmean ** 2
    turb.χ.attrs = {"long_name": "$χ$", "units": "°C²/s"}
    turb["KtTz"] = turb.Kρ * turb.Tzmean
    turb.KtTz.attrs = {"long_name": "$K_ρθ_z$", "units": "°Cm/s"}

    return turb


def choose_bins(pres, dz_segment):
    lefts = np.sort(
        np.hstack(
            [
                np.arange(1000 - dz_segment // 2, pres[0], -dz_segment // 2),
                np.arange(1000, pres[-1] + 1, dz_segment // 2),
            ]
        )
    )
    rights = lefts + dz_segment

    # lefts = np.arange(pres[0], pres[-1] + 1, dz_segment // 2)
    # rights = lefts + dz_segment

    return lefts, rights


def process_profile(profile, dz_segment=200, debug=False):
    """
    Processes finestructure turbulence estimate for Argo profiles
    in half-overlapping segments of length dz_segment.

    Parameters
    ----------

    profile: xr.DataArray
        Argo profile.
    dz_segment: optional
        Length of segment in dbar.

    Returns
    -------

    xr.Dataset if profile is not bad or too short.
    """
    for var in ["PRES", "TEMP", "PSAL"]:
        if profile[f"{var}_QC"] != 1:
            if debug:
                raise ValueError("bad_quality")
            return ["bad_quality"]

    profile["σ_θ"] = dcpy.eos.pden(profile.PSAL, profile.TEMP, profile.PRES, 0)
    profile["γ"] = dcpy.oceans.neutral_density(profile)
    profile = profile.isel(N_LEVELS=profile.PRES.notnull()).swap_dims(
        {"N_LEVELS": "PRES"}
    )

    profile_original = profile
    profile = trim_mld_mode_water(profile)
    profile = profile.isel(PRES=profile.PRES.notnull())

    if profile.sizes["PRES"] < 13:
        if debug:
            raise ValueError("empty")
        return ["empty!"]

    lefts, rights = choose_bins(profile.PRES.data, dz_segment)

    results = {
        var: np.full((len(lefts),), fill_value=np.nan)
        for var in [
            "Kρ",
            "ε",
            "ξvar",
            "ξvargm",
            "N2mean",
            "γmean",
            "flag",
            "npts",
            "pressure",
            "Tzlin",
            "Tzmean",
            "mean_dTdz_seg",
        ]
    }
    for var in ["γbnds", "pbnds"]:
        results[var] = np.full((len(lefts), 2), fill_value=np.nan)

    # N² calculation is the expensive step; do it only once
    N2full, _ = dcpy.eos.bfrq(
        profile.PSAL, profile.TEMP, profile.PRES, dim="PRES", lat=profile.LATITUDE
    )
    dTdzfull = (
        -1 * profile.TEMP.diff("PRES") / profile.PRES.diff("PRES")
    ).assign_coords({"PRES": N2full.PRES_mid.data})

    for idx, (l, r) in enumerate(zip(lefts, rights)):
        seg = profile.sel(PRES=slice(l, r))

        if seg.sizes["PRES"] == 1:
            results["flag"][idx] = -1
            continue

        # NATRE region: At 1000m sampling dz changes drastically;
        # it helps to just drop that one point
        seg = seg.where(seg.PRES.diff("PRES") < 21, drop=True)

        results["npts"][idx] = seg.sizes["PRES"]

        # max dz of 20m; ensure min number of points
        if results["npts"][idx] < np.ceil(dz_segment / 20):
            results["flag"][idx] = -1
            continue

        # if seg.PRES.diff("PRES").max() > 18:
        #    results["flag"][idx] = -2
        #    continue

        # TODO: despike
        # TODO: unrealistic values
        P = seg.PRES

        N2 = N2full.sel(PRES_mid=slice(P[0], P[-1]))

        # TODO: Is this interpolation sensible?
        dp = P.diff("PRES")
        if dp.max() - dp.min() > 2:
            dp = dp.median()
            seg = seg.interp(PRES=np.arange(P[0], P[-1], dp.median()))
            Pn2 = N2.PRES_mid
            N2 = N2.interp(PRES_mid=np.arange(Pn2[0], Pn2[-1], dp.median()))

        results["pressure"][idx] = (P.data[0] + P.data[-1]) / 2
        results["pbnds"][idx, 0] = P.data[0]
        results["pbnds"][idx, 1] = P.data[-1]

        results["γmean"][idx] = seg.γ.mean()
        results["γbnds"][idx, 0] = seg.γ.data[0]
        results["γbnds"][idx, 1] = seg.γ.data[-1]

        # TODO: move earlier?
        # N2, _ = dcpy.eos.bfrq(
        #    seg.PSAL, seg.TEMP, seg.PRES, dim="PRES", lat=seg.LATITUDE
        # )
        (
            results["Kρ"][idx],
            results["ε"][idx],
            results["ξvar"][idx],
            results["ξvargm"][idx],
            results["N2mean"][idx],
            results["flag"][idx],
        ) = estimate_turb_segment(
            N2.PRES_mid.data,
            N2.data,
            seg.cf["latitude"].data,
            max_wavelength=dz_segment,
            debug=debug,
        )

        results["Tzlin"][idx] = (
            seg.TEMP.polyfit("PRES", deg=1).sel(degree=1).polyfit_coefficients.values
            * -1
        )
        dTdz = dTdzfull.sel(PRES=slice(l, r))

        results["mean_dTdz_seg"][idx] = dTdz.mean("PRES")

        dTdz_fit = xr.polyval(
            N2.PRES_mid, dTdz.polyfit("PRES", deg=2).polyfit_coefficients
        )
        results["Tzmean"][idx] = dTdz_fit.mean().data

        # if debug:
        #     import matplotlib.pyplot as plt

        #     plt.figure()
        #     dTdz.plot()
        #     dTdz_fit.plot()

    dataset = results_to_xarray(results, profile)

    if debug:
        plot_profile_turb(profile_original, dataset)
    return dataset


def plot_profile_turb(profile, result):
    if all(result.ε.isnull().data):
        print("no output!")
        return
    p_edges = cfxr.bounds_to_vertices(result.p_bounds, bounds_dim="nbnds")

    f, axx = plt.subplots(1, 4, sharey=True)

    ax = dict(zip(["T", "ξ", "strat", "turb"], axx.flat))
    xlabels = ["$T$", "$ξ$ var", "", ""]

    ax["S"] = ax["T"].twiny()
    ax["γ"] = ax["T"].twiny()
    dcpy.plots.set_axes_color(ax["S"], "r")
    dcpy.plots.set_axes_color(ax["γ"], "teal")

    profile.TEMP.cf.plot(ax=ax["T"], marker=".", markersize=4)
    profile.PSAL.cf.plot(ax=ax["S"], color="r", _labels=False)
    profile.γ.cf.plot(ax=ax["γ"], color="teal", _labels=False)

    title = ax["T"].get_title()
    [a.set_title("") for a in axx.flat]

    result.ξvar.cf.plot(ax=ax["ξ"], _labels=False)
    result.ξvargm.cf.plot(ax=ax["ξ"], _labels=False)
    ax["ξ"].legend(["obs", "GM"])

    (9.81 * 1.7e-4 * result.Tzmean).cf.plot(ax=ax["strat"], _labels=False)
    result.N2mean.cf.plot(ax=ax["strat"], _labels=False)
    ax["strat"].legend(["$gαT_z$", "$N^2$"])

    result.ε.cf.plot(ax=ax["turb"], _labels=False)
    result.χ.cf.plot(ax=ax["turb"], _labels=False, xscale="log")
    ax["turb"].legend(["χ", "ε"])

    dcpy.plots.liney([result.Tmld, result.Tmode], color="k", ax=axx.flat)
    dcpy.plots.liney([result.σmld, result.σmode], color="b", ax=axx.flat)

    for lab, a in zip(xlabels, axx.flat):
        a.set_xlabel(lab)
        a.set_yticks(p_edges, minor=True)
        a.grid(True, axis="y", which="minor")

    f.suptitle(title)
