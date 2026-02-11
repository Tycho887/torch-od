import copy

import torch
import torch.nn as nn
from dsgp4.sgp4_batched import sgp4_batched
from dsgp4.sgp4init import sgp4, sgp4init
from dsgp4.tle import TLE
from dsgp4.util import get_gravity_constants

# from dsgp4.newton_method import newton_rapshon
from src.utils import extract_orbit_params, get_tle_epoch_tai


class SGP4Layer(nn.Module):
    """
    Differentiable SGP4 Propagator.
    Input: Tensor of orbital elements (SGP4 units: rad/min, rad, etc.)
    Output: Position/Velocity (TEME, km, km/s)
    """

    def __init__(self, init_tle: TLE, gravity_constant="wgs-84") -> None:
        super().__init__()
        self.gravity = gravity_constant
        self.sat_id = init_tle.satellite_catalog_number
        self.epoch = (init_tle._jdsatepoch + init_tle._jdsatepochF) - 2433281.5

        # Derivatives are usually 0 for standard fitting
        self.ndot = init_tle._ndot
        self.nddot = init_tle._nddot

        # Store a template to clone efficiently (avoids TLE parsing overhead)
        self.template_tle = copy.deepcopy(x=init_tle)

        self.system_kwargs = extract_orbit_params(tle=init_tle)

        # Store TLE epoch in TAI for relative time conversion
        self.tle_epoch_tai = get_tle_epoch_tai(init_tle)

    def forward(self, sgp4_params, t_tai) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sgp4_params: Tensor (N_batch, 7) -> [n, e, i, raan, argp, ma, bstar]
        t_tai: Tensor (T_steps) of TAI seconds
        """
        # 1. Unpack params
        n, e, i, raan, argp, ma, bstar = sgp4_params.unbind(-1)

        # 2. Clone the template TLE
        # Note: In a loop, you might want to avoid deepcopy for performance,
        # but for AD correctness with dsgp4's internal mutation, it's safest.
        # pass_tle = copy.deepcopy(self.template_tle)

        self.system_kwargs.update(
            {
                "mean_motion": n / 60,  # Convert rad/min to rad/sec
                "eccentricity": e,
                "inclination": i,
                "raan": raan,
                "argument_of_perigee": argp,
                "mean_anomaly": ma,
                "b_star": bstar,
            }
        )

        pass_tle = TLE(self.system_kwargs)

        # 3. Initialize SGP4 (This mutates pass_tle with the tensor values)
        whichconst = get_gravity_constants(gravity_constant_name=self.gravity)

        sgp4init(
            whichconst=whichconst,
            opsmode="i",
            satn=self.sat_id,
            epoch=self.epoch,
            xbstar=bstar,
            xndot=self.ndot,
            xnddot=self.nddot,
            xecco=e,
            xargpo=argp,
            xinclo=i,
            xmo=ma,
            xno_kozai=n,  # Convert from rad/min to rad/sec for SGP4
            xnodeo=raan,
            satellite=pass_tle,
        )

        # 4. Propagate
        # SGP4 expects minutes since TLE epoch
        t_diff_sec = t_tai - self.tle_epoch_tai
        t_minutes = t_diff_sec / 60.0

        # Returns: (Batch, Time, 2, 3) if batched, or (Time, 2, 3)
        state = sgp4(satellite=pass_tle, tsince=t_minutes)

        # Return Position and Velocity
        return state[:, 0], state[:, 1]


# We want to restructure to have a physics layer: SGP4 -> Physics projections (Range, Doppler) -> Sensor models + Biases


class GeometryLayer(nn.Module):
    """
    Computes relative kinematics between Satellite and Station (or Target).

    This layer is now stateless regarding the station's position; it expects
    the SystemObject to provide the synchronized states for both entities.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        sat_pos: torch.Tensor,
        sat_vel: torch.Tensor,
        station_pos: torch.Tensor,
        station_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sat_pos: (N, 3) TEME position of satellite
            sat_vel: (N, 3) TEME velocity of satellite
            station_pos: (N, 3) TEME position of ground station
            station_vel: (N, 3) TEME velocity of ground station

        Returns:
            Dictionary containing:
            - 'range': Slant range (km)
            - 'range_rate': Relative velocity projected on line-of-sight (km/s)
            - 'look_vector': Normalized Line-of-Sight vector
            - 'azimuth', 'elevation': (Optional) Topocentric angles
            + Original inputs passed through for convenience.
        """

        # 1. Relative State Vectors
        # r_rel = r_sat - r_station
        # v_rel = v_sat - v_station
        rel_pos = sat_pos - station_pos
        rel_vel = sat_vel - station_vel

        # 2. Slant Range (Distance)
        # Add epsilon to avoid div/0 gradients if range is 0
        dist = rel_pos.norm(dim=1, keepdim=True) + 1e-9

        # 3. Line of Sight (Unit Vector)
        u_los = rel_pos / dist

        # 4. Range Rate (Doppler Projection)
        # Projects relative velocity onto the line of sight
        # dot_product(v_rel, u_los)
        range_rate = (rel_vel * u_los).sum(dim=1, keepdim=True)

        return {
            # Primitives
            "range": dist,
            "range_rate": range_rate,
            "look_vector": u_los,
            "rel_pos": rel_pos,
            "rel_vel": rel_vel,
            # Pass-throughs (needed by some sensors or for debug)
            "sat_pos": sat_pos,
            "sat_vel": sat_vel,
            "station_pos": station_pos,
            "station_vel": station_vel,
        }
