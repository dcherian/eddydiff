import dcpy


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

    profile = profile.isel(N_LEVELS=profile.PRES.notnull()).swap_dims(
        {"N_LEVELS": "PRES"}
    )

    dataset = dcpy.finestructure.process_profile(profile, dz_segment)

    for var in [
        "CONFIG_MISSION_NUMBER",
        "PLATFORM_NUMBER",
        "CYCLE_NUMBER",
        "DIRECTION",
    ]:
        dataset.coords[var] = profile[var].data

    return dataset
