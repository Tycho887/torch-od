import torch
import torch.nn as nn
from dsgp4.sgp4init import sgp4init, sgp4
from dsgp4.util import get_gravity_constants
from dsgp4.tle import TLE
import copy

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
        pass_tle = copy.deepcopy(self.template_tle)

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
            xno_kozai=n, 
            xnodeo=raan, 
            satellite=pass_tle
        )

        # 4. Propagate
        # Returns: (Batch, Time, 2, 3) if batched, or (Time, 2, 3)
        state = sgp4(pass_tle, t_minutes)
        
        # Return Position and Velocity
        return state[:, 0], state[:, 1]

class DopplerSensor(nn.Module):
    """
    Pure Physics Layer: Range Rate -> Doppler.
    No learnable parameters here (they are passed as inputs or system biases).
    """
    def __init__(self, station_teme_pos, station_teme_vel, center_freq):
        super().__init__()
        # Register as buffers so they move to GPU with the model
        self.register_buffer('station_pos', station_teme_pos)
        self.register_buffer('station_vel', station_teme_vel)
        self.center_freq = center_freq

    def forward(self, sat_pos, sat_vel):
        # 1. Relative State
        # Broadcasting: (T, 3) - (T, 3)
        rel_pos = sat_pos - self.station_pos
        rel_vel = sat_vel - self.station_vel
        
        # 2. Range Rate
        dist = rel_pos.norm(dim=1, keepdim=True) + 1e-9
        los = rel_pos / dist
        range_rate = (rel_vel * los).sum(dim=1)
        
        # 3. Doppler
        c = 299792.458 # km/s
        doppler = -(range_rate / c) * self.center_freq
        
        return doppler