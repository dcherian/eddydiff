import cf_xarray  # noqa
import dcpy
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
        for var in ["ε", "K", "N2mean", "ξvar", "ξvargm"]
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

    turb.ξvar.attrs = {"long_name": "strain variance"}
    turb.ξvargm.attrs = {"long_name": "GM strain variance"}
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

    return turb


def process_profile(profile, dz_segment=200):
    """
    Processes finestructure turbulence estimate for Argo profiles
    in half-overlapping segments of length dz_segment.

    Parameters
    ----------

    profile: xr.DataArray
        Argo profile.
    dz_segment: optional
        Length of segment in dbar.
    """
    for var in ["PRES", "TEMP", "PSAL"]:
        if profile[f"{var}_QC"] != 1:
            print("bad quality!")
            return []

    profile["σ_θ"] = dcpy.eos.pden(profile.PSAL, profile.TEMP, profile.PRES, 0)
    profile["γ"] = dcpy.oceans.neutral_density(profile)
    profile = profile.isel(N_LEVELS=profile.PRES.notnull()).swap_dims(
        {"N_LEVELS": "PRES"}
    )

    profile = trim_mld_mode_water(profile)
    profile = profile.isel(PRES=profile.PRES.notnull())

    if profile.sizes["PRES"] < 13:
        print("empty!")
        return []

    lefts = np.arange(profile.PRES.data[0], profile.PRES.data[-1] + 1, dz_segment // 2)
    rights = lefts + dz_segment

    results = {
        var: np.full((len(lefts),), fill_value=np.nan)
        for var in [
            "K",
            "ε",
            "ξvar",
            "ξvargm",
            "N2mean",
            "γmean",
            "flag",
            "npts",
            "pressure",
        ]
    }
    for var in ["γbnds", "pbnds"]:
        results[var] = np.full((len(lefts), 2), fill_value=np.nan)

    for idx, (l, r) in enumerate(zip(lefts, rights)):
        seg = profile.sel(PRES=slice(l, r))

        results["npts"][idx] = seg.sizes["PRES"]

        if seg.sizes["PRES"] > 1 and seg.PRES.diff("PRES").min() > 18:
            results["flag"][idx] = -2
            continue

        if results["npts"][idx] < 12:
            results["flag"][idx] = -1
            continue

        # TODO: despike
        # TODO: unrealistic values

        # TODO: Is this interpolation sensible?
        P = seg.PRES
        dp = P.diff("PRES").median()
        seg = seg.interp(PRES=np.arange(P[0], P[-1], dp))
        results["pbnds"][idx, 0] = P.data[0]
        results["pbnds"][idx, 1] = P.data[-1]

        results["γmean"][idx] = seg.γ.mean()
        results["γbnds"][idx, 0] = seg.γ.data[0]
        results["γbnds"][idx, 1] = seg.γ.data[-1]

        results["pressure"][idx] = P.mean()

        N2, _ = dcpy.eos.bfrq(
            seg.PSAL, seg.TEMP, seg.PRES, dim="PRES", lat=seg.LATITUDE
        )
        (
            results["K"][idx],
            results["ε"][idx],
            results["ξvar"][idx],
            results["ξvargm"][idx],
            results["N2mean"][idx],
            results["flag"][idx],
        ) = estimate_turb_segment(
            N2.PRES_mid.data,
            N2.data,
            seg.cf["latitude"].item(),
            max_wavelength=dz_segment,
        )

    return results_to_xarray(results, profile)
