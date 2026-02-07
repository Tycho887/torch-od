import torch
import torch.nn as nn
from dsgp4.sgp4init import sgp4init, sgp4
from dsgp4.util import get_gravity_constants
from dsgp4.newton_method import newton_method
from dsgp4.tle import TLE
import numpy as np
import copy

class OrbitalParameterLayer(nn.Module):
    """
    Layer 1: The 'Embedding' of the Satellite.
    Stores the initial TLE and learns specific orbital elements.
    Output: A batch of valid SGP4 parameters.
    """
    def __init__(self, init_tle: TLE, trainable_keys: list = ['n', 'e', 'ma']):
        super().__init__()
        
        self.canonical_keys = ['n', 'e', 'i', 'raan', 'argp', 'ma', 'bstar']
        
        # 1. Store the Template (Fixed Values)
        # We extract all values from the TLE. These serve as the "Background" values.
        ref_values = [
            init_tle._no_kozai, init_tle._ecco, init_tle._inclo, 
            init_tle._nodeo, init_tle._argpo, init_tle._mo, init_tle._bstar
        ]
        self.register_buffer('template_params', torch.tensor(ref_values, dtype=torch.float64))
        
        # 2. Initialize Learnable Parameters
        # We replace the fixed template values with learnable nn.Parameters for the requested keys.
        self.params = nn.ParameterDict()
        
        for key in trainable_keys:
            if key not in self.canonical_keys:
                raise ValueError(f"Unknown orbital element: {key}")
            
            # Find the index of this key in the canonical list
            idx = self.canonical_keys.index(key)
            initial_val = ref_values[idx]
            
            # Create a learnable parameter (scalar)
            self.params[key] = nn.Parameter(torch.tensor(initial_val, dtype=torch.float64))

        # Store TLE metadata for regeneration
        self.sat_id = init_tle.satellite_catalog_number
        self.epoch = (init_tle._jdsatepoch + init_tle._jdsatepochF) - 2433281.5
        self.ndot = init_tle._ndot
        self.nddot = init_tle._nddot

    def forward(self):
        """
        Reconstructs the full 7-element SGP4 state vector.
        Mixing fixed 'template' values with learned 'params'.
        """
        # Start with the fixed template
        full_state = self.template_params.clone()
        
        # Overwrite with learnable parameters
        # (This is differentiable because we use direct assignment of tensors)
        for key, param in self.params.items():
            idx = self.canonical_keys.index(key)
            full_state[idx] = param
            
        return full_state

class SGP4Units:
    """
    Centralizes unit conversions for SGP4.
    """
    # SGP4 Constants
    XP_DOT_P = 1440.0 / (2.0 * np.pi)  # 229.183118...
    
    @staticmethod
    def rev_day_to_rad_s(rev_per_day):
        """
        Converts Mean Motion from [rev/day] (TLE standard) 
        to [rad/min] (SGP4 internal standard).
        Formula: n_rad_min = n_rev_day * (2*pi / 1440)
        """
        # (2 * pi) / 1440 = 1 / XP_DOT_P
        if torch.is_tensor(rev_per_day):
            return rev_per_day * (2.0 * np.pi / 1440.0) / 60.0
        return rev_per_day * (2.0 * np.pi / 1440.0) / 60.0

    @staticmethod
    def deg_to_rad(deg):
        if torch.is_tensor(deg):
            return deg * (np.pi / 180.0)
        return deg * (np.pi / 180.0)

class SGP4Layer(nn.Module):
    def __init__(self, init_tle: TLE, gravity_constant='wgs-84'):
        super().__init__()
        # We hold the TLE *only* as a template for metadata 
        # (catalog number, strings, epoch integers).
        # We never pass this object to sgp4init directly.
        self.template_tle = init_tle
        self.gravity = gravity_constant
        
        # Pre-extract immutable metadata to avoid lookup overhead
        self.sat_id = init_tle.satellite_catalog_number
        self.epoch = (init_tle._jdsatepoch + init_tle._jdsatepochF) - 2433281.5
        
        # These are standard TLE derivatives, usually 0 for numerical optimization
        # unless you are specifically fitting them.
        self.ndot = init_tle._ndot
        self.nddot = init_tle._nddot

        self.tle_kwargs = {
            'satellite_catalog_number': init_tle.satellite_catalog_number,
            'epoch_year': init_tle.epoch_year,
            'epoch_days': init_tle.epoch_days,
            'b_star': init_tle._bstar,
            # Note: We store the "static" parts here. 
            # The "dynamic" parts (n, e, i, etc.) will be injected during forward().
            'mean_motion': init_tle.mean_motion, 
            'eccentricity': init_tle.eccentricity, 
            'inclination': init_tle.inclination, 
            'raan': init_tle.raan, 
            'argument_of_perigee': init_tle.argument_of_perigee, 
            'mean_anomaly': init_tle.mean_anomaly, 
            'mean_motion_first_derivative': init_tle.mean_motion_first_derivative,
            'mean_motion_second_derivative': init_tle.mean_motion_second_derivative,
            'classification': init_tle.classification,
            'ephemeris_type': init_tle.ephemeris_type,
            'international_designator': init_tle.international_designator,
            'revolution_number_at_epoch': init_tle.revolution_number_at_epoch,
            'element_number': init_tle.element_number
        }


    def forward(self, sgp4_params, t_minutes):
        """
        sgp4_params: Tensor containing [n, e, i, raan, argp, ma, bstar]
                     Units must match SGP4 requirements:
                     n: rad/min
                     angles: rad
                     bstar: 1/ER
        """
        # 1. Unpack (Indices based on your canonical order)
        n, e, i, raan, argp, ma, bstar = (
            sgp4_params[0], sgp4_params[1], sgp4_params[2], 
            sgp4_params[3], sgp4_params[4], sgp4_params[5], sgp4_params[6]
        )

        # 2. Create a "Disposable" TLE for this pass
        # CRITICAL: We must deepcopy to ensure we have a fresh object 
        # that sgp4init can mutate without affecting other passes.
        # This isolates the computation graph.

        self.tle_kwargs.update({
            'mean_motion': n/60,
            'eccentricity': e,
            'inclination': i,
            'raan': raan,
            'argument_of_perigee': argp,
            'mean_anomaly': ma,
            'b_star': bstar
        })

        pass_tle = TLE(self.tle_kwargs)  # Create a new TLE object with updated parameters

        # pass_tle = copy.deepcopy(self.template_tle)

        # 3. Initialize SGP4
        # sgp4init mutates 'pass_tle' in-place, attaching the tensors 
        # (n, e, i...) to it as internal attributes.
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
            satellite=pass_tle  # <--- Mutated here!
        )

        # 4. Propagate
        # sgp4 reads the internal tensors from pass_tle and computes pos/vel
        state = sgp4(pass_tle, t_minutes)
        
        # State shape is (T, 2, 3) -> [Pos, Vel]

        pos = state[:, 0]
        vel = state[:, 1]
            
        return pos, vel


class DopplerSensor(nn.Module):
    """
    Layer 3: The Measurement Model.
    Input: Position/Velocity (from Layer 2), Contact Indices
    Output: Predicted Doppler
    """
    def __init__(self, station_pos, station_vel, center_freq, num_contacts):
        super().__init__()
        
        # Station Ephemerides (Fixed)
        self.register_buffer('station_pos', station_pos)
        self.register_buffer('station_vel', station_vel)
        self.center_freq = center_freq
        
        # --- The Bias Embedding ---
        # This replaces the complex "Index Map".
        # If we have 50 contacts, we create 50 learnable bias values.
        # Input: Contact ID [0, 5, 2...] -> Output: Bias Value [120.5, -40.2, ...]
        self.bias_embedding = nn.Embedding(num_embeddings=num_contacts, embedding_dim=1)
        
        # Initialize biases to 0
        nn.init.zeros_(self.bias_embedding.weight)

    def forward(self, sat_pos, sat_vel, contact_indices):
        """
        contact_indices: Tensor (T,) mapping each timestamp to a specific pass/contact ID.
        """
        # 1. Physics: Calculate Range Rate
        rel_pos = sat_pos - self.station_pos
        rel_vel = sat_vel - self.station_vel
        
        dist = rel_pos.norm(dim=1, keepdim=True) + 1e-9
        los = rel_pos / dist
        range_rate = (rel_vel * los).sum(dim=1)
        
        # 2. Physics: Calculate Doppler
        c = 299792.458
        doppler_pred = -(range_rate / c) * self.center_freq
        
        # 3. Learnable Correction: Apply Bias
        # The Embedding layer handles the lookup automatically!
        # shape: (T, 1) -> squeeze to (T,)
        bias = self.bias_embedding(contact_indices).squeeze()
        
        return doppler_pred + bias
    
