import torch
from dsgp4.sgp4init import sgp4, sgp4init
from dsgp4.mldsgp4 import mldsgp4
from dsgp4.tle import TLE
from dsgp4.util import from_datetime_to_mjd, get_gravity_constants


def get_tle_epoch_unix(tle: TLE) -> float:
    """
    Extracts the TLE epoch and converts it to UNIX seconds since 1970-01-01.
    """
    tle_epoch_mjd = from_datetime_to_mjd(datetime_obj=tle._epoch)
    return (tle_epoch_mjd - 40587.0) * 86400.0


def val(key, default, map_param_to_idx, x) -> torch.Tensor:
    if key in map_param_to_idx:
        y = x[map_param_to_idx[key]]
        return y
    else:
        return default


def update(
    tle: TLE, x: torch.Tensor, map_param_to_idx, gravity_constant: str = "wgs-84"
) -> TLE:
    arguments = {
        # --- Dynamic Parameters (Potentially Tensors) ---
        "mean_motion": float(
            val(key="mean_motion", default=tle._no_kozai, map_param_to_idx=map_param_to_idx, x=x).detach() / 60
        ),
        "eccentricity": float(val(key="eccentricity", default=tle._ecco, map_param_to_idx=map_param_to_idx, x=x).detach()),
        "inclination": float(val(key="inclination", default=tle._inclo, map_param_to_idx=map_param_to_idx, x=x).detach()),
        "raan": float(val(key="raan", default=tle._nodeo, map_param_to_idx=map_param_to_idx, x=x).detach()),
        "argument_of_perigee": float(
            val(key="argument_of_perigee", default=tle._argpo, map_param_to_idx=map_param_to_idx, x=x).detach()
        ),
        "mean_anomaly": float(val(key="mean_anomaly", default=tle._mo, map_param_to_idx=map_param_to_idx, x=x).detach()),
        "b_star": float(val(key="b_star", default=tle._bstar, map_param_to_idx=map_param_to_idx, x=x).detach()),
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
        xbstar=val(key="b_star", default=sat_obj._bstar, map_param_to_idx=map_param_to_idx, x=x),
        xndot=sat_obj._ndot,
        xnddot=sat_obj._nddot,
        xecco=val(key="eccentricity", default=sat_obj._ecco, map_param_to_idx=map_param_to_idx, x=x),
        xargpo=val(key="argument_of_perigee", default=sat_obj._argpo, map_param_to_idx=map_param_to_idx, x=x),
        xinclo=val(key="inclination", default=sat_obj._inclo, map_param_to_idx=map_param_to_idx, x=x),
        xmo=x[0],  # val("mean_anomaly", sat_ob-j._mo, ssv, x),
        xno_kozai=val(key="mean_motion", default=sat_obj._no_kozai, map_param_to_idx=map_param_to_idx, x=x),
        xnodeo=val(key="raan", default=sat_obj._nodeo, map_param_to_idx=map_param_to_idx, x=x),
        satellite=sat_obj,
    )

    return sat_obj


# def dsgp4(
#     tle: TLE,
#     timestamps: torch.Tensor,
#     x: torch.Tensor | None = None,
#     map_param_to_idx=None,
#     gravity_constant: str = "wgs-84",
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Unified propagation function.

#     Args:
#         tle: Base TLE object.
#         timestamps: (N,) Tensor of UNIX seconds.
#         x: (Optional) State vector tensor. If provided, enables gradients.
#         map_param_to_idx: (Optional) Mapping for the state vector.
#         gravity_constant: Gravity model name.

#     Returns:
#         pos: (N, 3) TEME Position (km)
#         vel: (N, 3) TEME Velocity (km/s)
#     """

#     # 1. Time Setup (UNIX -> Minutes since TLE Epoch)
#     tle_epoch = get_tle_epoch_unix(tle)
#     tsince_min = (timestamps - tle_epoch) / 60.0

#     # Handle batch dimension for dSGP4 (1, N)
#     if tsince_min.ndim == 1:
#         tsince_min = tsince_min.unsqueeze(0)

#     # 2. TLE Setup (Static vs Differentiable)
#     if x is None:
#         sat_obj = tle.copy()
#     else:
#         sat_obj = update(
#             tle=tle, x=x, map_param_to_idx=map_param_to_idx, gravity_constant=gravity_constant
#         )

#     # 4. Propagation
#     # Returns (Batch, Time, 2, 3)
#     state = sgp4(satellite=sat_obj, tsince=tsince_min)

#     # 5. Unpack to (N, 3)
#     pos = state[:, 0]
#     vel = state[:, 1]

#     return pos, vel
