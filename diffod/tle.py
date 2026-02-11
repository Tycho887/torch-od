import copy

import torch
from dsgp4.sgp4init import sgp4, sgp4init
from dsgp4.tle import TLE
from dsgp4.util import from_datetime_to_mjd, get_gravity_constants


def get_tle_epoch_unix(tle: TLE) -> float:
    """
    Extracts the TLE epoch and converts it to UNIX seconds since 1970-01-01.
    """
    tle_epoch_mjd = from_datetime_to_mjd(datetime_obj=tle._epoch)
    return (tle_epoch_mjd - 40587.0) * 86400.0


def val(key, default, state_def, x):
    if state_def and key in state_def.map_param_to_idx:
        return x[state_def.map_param_to_idx[key]]
    else:
        return default


def propagate(
    tle: TLE,
    timestamps: torch.Tensor,
    x: torch.Tensor | None = None,
    state_def=None,
    gravity_constant: str = "wgs-84",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unified propagation function.

    Args:
        tle: Base TLE object.
        timestamps: (N,) Tensor of UNIX seconds.
        x: (Optional) State vector tensor. If provided, enables gradients.
        state_def: (Optional) Mapping for the state vector.
        gravity_constant: Gravity model name.

    Returns:
        pos: (N, 3) TEME Position (km)
        vel: (N, 3) TEME Velocity (km/s)
    """

    # 1. Time Setup (UNIX -> Minutes since TLE Epoch)
    tle_epoch = get_tle_epoch_unix(tle)
    tsince_min = (timestamps - tle_epoch) / 60.0

    # Handle batch dimension for dSGP4 (1, N)
    if tsince_min.ndim == 1:
        tsince_min = tsince_min.unsqueeze(0)

    # 2. TLE Setup (Static vs Differentiable)
    if x is None:
        sat_obj = tle
    else:
        # Create a shallow copy so we don't mutate the original TLE
        # during the in-place sgp4init process
        # sat_obj = copy.copy(tle)

        print(tle)

        arguments = {
            # --- Dynamic Parameters (Potentially Tensors) ---
            "mean_motion": float(
                val("mean_motion", tle._no_kozai, state_def, x).detach() / 60
            ),
            "eccentricity": float(
                val("eccentricity", tle._ecco, state_def, x).detach()
            ),
            "inclination": float(val("inclination", tle._inclo, state_def, x).detach()),
            "raan": float(val("raan", tle._nodeo, state_def, x).detach()),
            "argument_of_perigee": float(
                val("argument_of_perigee", tle._argpo, state_def, x).detach()
            ),
            "mean_anomaly": float(val("mean_anomaly", tle._mo, state_def, x).detach()),
            "b_star": float(val("b_star", tle._bstar, state_def, x).detach()),
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
        sat_obj = TLE(arguments)

        # 3. Differentiable Initialization (Graph Access)
        # We pass the tensors directly into sgp4init.
        # This records the operations (constants calculation) on the graph.
        whichconst = get_gravity_constants(gravity_constant)

        sgp4init(
            whichconst=whichconst,
            opsmode="i",
            satn=sat_obj.satellite_catalog_number,
            epoch=(sat_obj._jdsatepoch + sat_obj._jdsatepochF) - 2433281.5,
            xbstar=val("b_star", sat_obj._bstar, state_def, x),
            xndot=sat_obj._ndot,
            xnddot=sat_obj._nddot,
            xecco=val("eccentricity", sat_obj._ecco, state_def, x),
            xargpo=val("argument_of_perigee", sat_obj._argpo, state_def, x),
            xinclo=val("inclination", sat_obj._inclo, state_def, x),
            xmo=val("mean_anomaly", sat_obj._mo, state_def, x),
            xno_kozai=val("mean_motion", sat_obj._no_kozai, state_def, x),
            xnodeo=val("raan", sat_obj._nodeo, state_def, x),
            satellite=sat_obj,
        )

        print(sat_obj)

    # 4. Propagation
    # Returns (Batch, Time, 2, 3)
    state = sgp4(sat_obj, tsince_min)

    # 5. Unpack to (N, 3)
    pos = state[:, 0]
    vel = state[:, 1]

    return pos, vel
