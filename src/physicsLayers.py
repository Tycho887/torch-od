import torch
import torch.nn as nn
from dsgp4.sgp4init import sgp4init, sgp4
from dsgp4.util import get_gravity_constants
from dsgp4.tle import TLE
import copy
# from dsgp4.newton_method import newton_rapshon
from src.utils import extract_orbit_params, list_elements

class SGP4Layer(nn.Module):
    """
    Differentiable SGP4 Propagator.
    Input: Tensor of orbital elements (SGP4 units: rad/min, rad, etc.)
    Output: Position/Velocity (TEME, km, km/s)
    """
    def __init__(self, init_tle: TLE, gravity_constant='wgs-84'):
        super().__init__()
        self.gravity = gravity_constant
        self.sat_id = init_tle.satellite_catalog_number
        self.epoch = (init_tle._jdsatepoch + init_tle._jdsatepochF) - 2433281.5
        
        # Derivatives are usually 0 for standard fitting
        self.ndot = init_tle._ndot
        self.nddot = init_tle._nddot
        
        # Store a template to clone efficiently (avoids TLE parsing overhead)
        self.template_tle = copy.deepcopy(init_tle)

        self.system_kwargs = extract_orbit_params(init_tle)

    def forward(self, sgp4_params, t_minutes):
        """
        sgp4_params: Tensor (N_batch, 7) -> [n, e, i, raan, argp, ma, bstar]
        t_minutes: Tensor (T_steps)
        """
        # 1. Unpack params
        n, e, i, raan, argp, ma, bstar = sgp4_params.unbind(-1)

        # 2. Clone the template TLE
        # Note: In a loop, you might want to avoid deepcopy for performance,
        # but for AD correctness with dsgp4's internal mutation, it's safest.
        # pass_tle = copy.deepcopy(self.template_tle)

        self.system_kwargs.update({
            'mean_motion': n / 60,  # Convert rad/min to rad/sec
            'eccentricity': e,
            'inclination': i,
            'raan': raan,
            'argument_of_perigee': argp,
            'mean_anomaly': ma,
            'b_star': bstar
        })

        pass_tle = TLE(self.system_kwargs)

        # 3. Initialize SGP4 (This mutates pass_tle with the tensor values)
        whichconst = get_gravity_constants(self.gravity)


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
            xno_kozai=n, # Convert from rad/min to rad/sec for SGP4
            xnodeo=raan, 
            satellite=pass_tle
        )

        # 4. Propagate
        # Returns: (Batch, Time, 2, 3) if batched, or (Time, 2, 3)
        state = sgp4(pass_tle, t_minutes)
        
        # Return Position and Velocity
        return state[:, 0], state[:, 1]

# We want to restructure to have a physics layer: SGP4 -> Physics projections (Range, Doppler) -> Sensor models + Biases

class GeometryLayer(nn.Module):
    """
    Computes relative kinematics between Satellite and Station.
    This is the 'ground truth' geometry before any sensor physics.
    """
    def __init__(self, station_teme_pos, station_teme_vel):
        super().__init__()
        self.register_buffer('station_pos', station_teme_pos)
        self.register_buffer('station_vel', station_teme_vel)

    def forward(self, sat_pos, sat_vel, contact_indices):
        """
        Returns a dictionary of geometric primitives.
        """
        # 1. Match Station to Satellite (by index)
        # Assuming station_pos is pre-computed for all timestamps
        st_pos = self.station_pos
        st_vel = self.station_vel

        # 2. Relative Vectors (The "Geometry")
        rel_pos = sat_pos - st_pos
        rel_vel = sat_vel - st_vel
        
        # 3. Derived Primitives (Computed once, used by all sensors)
        range_dist = rel_pos.norm(dim=1, keepdim=True) + 1e-9
        unit_vec = rel_pos / range_dist
        
        # Projection of velocity onto line-of-sight (Range Rate)
        range_rate = (rel_vel * unit_vec).sum(dim=1, keepdim=True)

        return {
            'rel_pos': rel_pos,
            'rel_vel': rel_vel,
            'range': range_dist,
            'range_rate': range_rate,
            'unit_vec': unit_vec,
            'contact_indices': contact_indices
        }

class RangeRatePhysics(nn.Module):
    def forward(self, geometry):
        return geometry['range_rate'] # km/s

class RangePhysics(nn.Module):
    def forward(self, geometry):
        return geometry['range'] # km