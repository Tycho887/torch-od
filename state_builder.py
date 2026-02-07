import torch
import numpy as np

class StateDefinition:
    def __init__(self, orbital_keys: list, sensor_config: dict, init_tle):
        """
        sensor_config: {'doppler': num_passes_dop, 'range': num_passes_rng}
        """
        self.orbital_keys = orbital_keys
        self.sensor_config = sensor_config # e.g. {'doppler': 5, 'range': 2}
        
        # --- 1. Orbital Elements Map ---
        self.idx_map = {k: i for i, k in enumerate(orbital_keys)}
        current_idx = len(orbital_keys)
        
        # --- 2. Sensor Bias Maps ---
        # We store start/end indices for each sensor's bias block
        self.bias_slices = {}
        for name, num_passes in sensor_config.items():
            self.bias_slices[name] = slice(current_idx, current_idx + num_passes)
            current_idx += num_passes
            
        self.total_dim = current_idx
        
        # --- 3. Static Defaults (Standardization) ---
        self.revday_to_radsec = (2.0 * np.pi) / (1440.0) # rev/day to rad/sec
        self.defaults = {
            'n': init_tle._no_kozai, # * self.revday_to_radsec, 
            'e': init_tle._ecco,
            'i': init_tle._inclo,     
            'raan': init_tle._nodeo,  
            'argp': init_tle._argpo,  
            'ma': init_tle._mo,       
            'bstar': init_tle._bstar
        }

    def unpack(self, state_vector):
        """
        Returns:
            sgp4_tensor: (7,)
            bias_dict: {'doppler': Tensor(N_d,), 'range': Tensor(N_r,), ...}
        """
        # A. Reconstruct SGP4 Inputs
        current_vals = self.defaults.copy()
        for key in self.orbital_keys:
            current_vals[key] = state_vector[self.idx_map[key]]
            
        sgp4_tensor = torch.stack([
            torch.as_tensor(current_vals['n']),
            torch.as_tensor(current_vals['e']),
            torch.as_tensor(current_vals['i']),
            torch.as_tensor(current_vals['raan']),
            torch.as_tensor(current_vals['argp']),
            torch.as_tensor(current_vals['ma']),
            torch.as_tensor(current_vals['bstar'])
        ])
        
        # B. Extract Biases per Sensor
        bias_dict = {}
        for name, sl in self.bias_slices.items():
            # Crucial: This slice maintains the gradient graph
            bias_dict[name] = state_vector[sl]
            
        return sgp4_tensor, bias_dict

    def get_initial_state(self):
        """
        Returns x0 populated with the initial TLE values for the active keys,
        and zeros for the biases.
        """
        # 1. Start with zeros (correct for biases)
        x = torch.zeros(self.total_dim, dtype=torch.float64)
        
        # 2. Overwrite the slots for the active orbital elements
        for key in self.orbital_keys:
            # Get the index in the flat vector
            idx = self.idx_map[key]
            
            # Get the initial value (e.g., 15.49 revs/day or 51.6 deg)
            val = self.defaults[key]
            
            # Safety: Ensure we extract a standard float if it's a 0-dim tensor
            if torch.is_tensor(val):
                val = val.item()
                
            x[idx] = val
            
        return x