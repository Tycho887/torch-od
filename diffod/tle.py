import torch
from dsgp4.tle import TLE
from dsgp4.sgp4init import sgp4init, sgp4
from dsgp4.util import get_gravity_constants


from .state import StateDefinition


def extract_orbit_params(tle: TLE) -> dict[str, torch.Tensor | float | int]:
    """
    Helper to convert a TLE object into a dictionary of parameters.
    This is used to populate 'static_tle_params' and can be reused in unpacking.
    """
    return {
        "satellite_catalog_number": tle.satellite_catalog_number,
        "epoch_year": tle.epoch_year,
        "epoch_days": tle.epoch_days,
        "b_star": tle._bstar,
        "mean_motion": tle.mean_motion,
        "eccentricity": tle.eccentricity,
        "inclination": tle.inclination,
        "raan": tle.raan,
        "argument_of_perigee": tle.argument_of_perigee,
        "mean_anomaly": tle.mean_anomaly,
        "mean_motion_first_derivative": tle.mean_motion_first_derivative,
        "mean_motion_second_derivative": tle.mean_motion_second_derivative,
        "classification": tle.classification,
        "ephemeris_type": tle.ephemeris_type,
        "international_designator": tle.international_designator,
        "revolution_number_at_epoch": tle.revolution_number_at_epoch,
        "element_number": tle.element_number,
    }



def update(tle: TLE, state_def: StateDefinition, x: torch.Tensor, gravity_constant_name: str = "WGS84") -> TLE:
    """
    Constructs a NEW TLE object using values from x_state where applicable,
    and falling back to init_tle for static parameters.

    This maintains the computational graph because we pass the 'x' tensors
    directly into the dictionary used to initialize the TLE.
    """

    # Helper to decide: Get from Tensor (Optimization) OR Get from Object (Static)
    def get_val(key, default_val) -> torch.Tensor:
        if key in state_def.map_param_to_idx:
            idx = state_def.map_param_to_idx[key]
            return x[idx]  # This is a Tensor, keeps Grad!
        return default_val

    # 1. Construct the Dictionary with mixed Tensor/Float values
    arguments = {
        # --- Dynamic Parameters (Potentially Tensors) ---
        "mean_motion": get_val(key="mean_motion", default_val=tle.mean_motion),
        "eccentricity": get_val(key="eccentricity", default_val=tle.eccentricity),
        "inclination": get_val(key="inclination", default_val=tle.inclination),
        "raan": get_val(key="raan", default_val=tle.raan),
        "argument_of_perigee": get_val(key="argument_of_perigee", default_val=tle.argument_of_perigee),
        "mean_anomaly": get_val(key="mean_anomaly", default_val=tle.mean_anomaly),
        "b_star": get_val(key="b_star", default_val=tle.bstar),
        # --- Static Parameters (Pass-through) ---
        "satellite_catalog_number": tle.satellite_catalog_number,
        "epoch_year": tle.epoch_year,
        "epoch_days": tle.epoch_days,
        "mean_motion_first_derivative": tle.mean_motion_first_derivative,
        "mean_motion_second_derivative": tle.mean_motion_second_derivative,
        "classification": tle.classification,
        "ephemeris_type": tle.ephemeris_type,
        "international_designator": tle.international_designator,
        "revolution_number_at_epoch": tle.revolution_number_at_epoch,
        "element_number": tle.element_number,
    }

    # 2. Return a fresh TLE object
    # dSGP4 will store the tensors inside. When you call propagate() on this object,
    # the operations will trace back to x_state.
    tle_new = TLE(data=arguments)

    whichconst = get_gravity_constants(gravity_constant_name=gravity_constant_name)

    sgp4init(
    whichconst=whichconst, opsmode="i", satn=tle_new.satellite_catalog_number,
    epoch=(tle_new._jdsatepoch + tle_new._jdsatepochF) - 2433281.5,
    xbstar=tle_new._bstar, xndot=tle_new._ndot, xnddot=tle_new._nddot,
    xecco=tle_new._ecco, xargpo=tle_new._argpo, xinclo=tle_new._inclo,
    xmo=tle_new._mo, xno_kozai=tle_new._no_kozai, xnodeo=tle_new._nodeo, satellite=tle_new,
    )
