import torch
import numpy as np

class StateDefinition:
    def __init__(self, keys_to_fit, num_passes, init_tle):
        self.keys = keys_to_fit
        self.num_passes = num_passes
        
        # SGP4 Constants for Unit Conversion
        # XP_DOT_P = 1440.0 / (2.0 * np.pi) 
        # TLE mean motion is rev/day. SGP4 needs rad/min.
        # factor = (2pi / 1440)
        self.revday_to_radmin = (2.0 * np.pi) / 1440.0

        # 1. Extract Initial Values (SGP4 Units)
        # We assume the input TLE object has attributes: _no_kozai, _ecco, etc.
        # These are usually populated by dsgp4 parsing.
        self.defaults = {
            'n': init_tle._no_kozai,  # rad/min
            'e': init_tle._ecco,
            'i': init_tle._inclo,     # rad
            'raan': init_tle._nodeo,  # rad
            'argp': init_tle._argpo,  # rad
            'ma': init_tle._mo,       # rad
            'bstar': init_tle._bstar
        }
        
        # 2. Build Index Map
        self.idx_map = {k: i for i, k in enumerate(self.keys)}
        self.bias_start = len(self.keys)
        self.total_dim = self.bias_start + num_passes

    def get_initial_state_vector(self):
        """Returns x0 (Orbit Elements + Zeros for Biases)"""
        x = torch.zeros(self.total_dim, dtype=torch.float64)
        
        # Fill Orbit Parameters
        for key in self.keys:
            val = self.defaults.get(key)
            if torch.is_tensor(val):
                val = val.item() # Ensure scalar
            x[self.idx_map[key]] = val
            
        return x

    def unpack(self, state_vector):
        """
        Returns:
            sgp4_tensor: Shape (7,) -> [n, e, i, raan, argp, ma, bstar]
            pass_biases: Shape (Num_Passes,)
        """
        # 1. Reconstruct full parameter set
        # Start with defaults
        current_vals = self.defaults.copy()
        
        # Overwrite with active state variables
        for key in self.keys:
            idx = self.idx_map[key]
            current_vals[key] = state_vector[idx]
            
        # 2. Stack into SGP4 Tensor Order
        # Order MUST be: n, e, i, raan, argp, ma, bstar
        sgp4_tensor = torch.stack([
            torch.as_tensor(current_vals['n']),
            torch.as_tensor(current_vals['e']),
            torch.as_tensor(current_vals['i']),
            torch.as_tensor(current_vals['raan']),
            torch.as_tensor(current_vals['argp']),
            torch.as_tensor(current_vals['ma']),
            torch.as_tensor(current_vals['bstar'])
        ])
        
        # 3. Extract Biases
        pass_biases = None
        if self.num_passes > 0:
            pass_biases = state_vector[self.bias_start:]
            
        return sgp4_tensor, pass_biases