import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseSensor(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass

    @abstractmethod
    def forward(self, geometry_data, sensor_params, metadata):
        pass

    def get_initial_params(self):
        """
        Returns the initial guess for this sensor's parameters.
        Default: Vector of zeros.
        Override this if you need non-zero initialization (e.g. nominal freq).
        """
        return torch.zeros(self.num_params, dtype=torch.float64)

class DopplerSensor(BaseSensor):
    def __init__(self, station_teme_pos, station_teme_vel, center_freq, num_passes, fit_center_freq=False):
        super().__init__()
        self.register_buffer('station_pos', station_teme_pos)
        self.register_buffer('station_vel', station_teme_vel)
        self.nominal_freq = center_freq
        self.fit_fc = fit_center_freq
        self.n_passes = num_passes
        
        # [Bias_Pass_1, Bias_Pass_2, ..., Delta_Freq]
        self._n_params = num_passes + (1 if fit_center_freq else 0)

    @property
    def num_params(self):
        return self._n_params

    def get_initial_params(self):
        # Biases start at 0.0
        # Freq offset starts at 0.0 (since we model it as delta from nominal)
        return torch.zeros(self.num_params, dtype=torch.float64)

    def forward(self, geometry_data, sensor_params, contact_indices):
        # 1. Physics: Range Rate
        # Note: We re-calculate relative kinematics here to support per-sensor timestamps
        sat_pos = geometry_data['pos']
        sat_vel = geometry_data['vel']
        
        rel_pos = sat_pos - self.station_pos
        rel_vel = sat_vel - self.station_vel
        
        dist = rel_pos.norm(dim=1, keepdim=True) + 1e-9
        los_vec = rel_pos / dist
        range_rate = (rel_vel * los_vec).sum(dim=1).view(-1) # (N,)

        # 2. Determine Effective Frequency
        # If we are fitting Fc, the last parameter is the correction (delta_hz)
        freq_to_use = self.nominal_freq
        pass_biases = sensor_params
        
        if self.fit_fc:
            # Split params: [Pass_Biases (N) | Freq_Bias (1)]
            pass_biases = sensor_params[:-1]
            delta_freq = sensor_params[-1]
            freq_to_use = freq_to_use + delta_freq

        # 3. Ideal Doppler
        c = 299792.458 # km/s
        doppler_ideal = -(range_rate / c) * freq_to_use
        
        # 4. Apply Per-Pass Bias
        # contact_indices maps measurement -> pass ID
        if self.n_passes > 0:
            bias_correction = pass_biases[contact_indices]
            return doppler_ideal + bias_correction
            
        return doppler_ideal

class RadarSensor(BaseSensor):
    def __init__(self, station_teme_pos, num_passes):
        super().__init__()
        self.register_buffer('station_pos', station_teme_pos)
        self.n_passes = num_passes

    @property
    def num_params(self):
        return self.n_passes
    
    def forward(self, geometry_data, sensor_params, contact_indices):
        sat_pos = geometry_data['pos']
        rel_pos = sat_pos - self.station_pos
        
        # 1. Ideal Range
        range_km = rel_pos.norm(dim=1).view(-1)
        
        # 2. Apply Bias
        if self.n_passes > 0:
            bias_correction = sensor_params[contact_indices]
            return range_km + bias_correction
            
        return range_km

# import torch
# import torch.nn as nn
# from abc import ABC, abstractmethod

# class BaseSensor(nn.Module, ABC):
#     """
#     Abstract contract for a sensor.
#     """
#     def __init__(self):
#         super().__init__()

#     @property
#     @abstractmethod
#     def num_params(self) -> int:
#         """
#         The number of learnable parameters (biases, coefficients) 
#         this sensor needs in the global state vector.
#         """
#         pass

#     @abstractmethod
#     def forward(self, geometry_data, sensor_params, metadata):
#         """
#         Apply instrument physics and biases.
        
#         Args:
#             geometry_data: Dict containing 'rel_pos', 'range_rate', etc.
#             sensor_params: Tensor of shape (num_params,) - slice from global state.
#             metadata: Dict or Tensor containing local indices (e.g., pass IDs, integration times).
#         """
#         pass

# class ComplexTelescope(BaseSensor):
#     def __init__(self, num_passes, station_pos):
#         super().__init__()
#         self.num_passes = num_passes
#         self.register_buffer('station_pos', station_pos)
        
#         # We need: 3 Mount Angles (Euler) + 1 Time Bias per pass
#         self._n_params = 3 + num_passes 

#     @property
#     def num_params(self):
#         return self._n_params

#     def forward(self, geometry_data, sensor_params, contact_indices):
#         # 1. Unpack My Parameters (The "Local Map")
#         # I know that params[0:3] are Mount Errors
#         mount_euler_angles = sensor_params[:3]
        
#         # I know that params[3:] are Time Biases
#         time_biases = sensor_params[3:]
        
#         # 2. Physics: Apply Time Bias *Before* Geometry?
#         # (For strict correctness, time bias affects the SGP4 query time. 
#         #  If the bias is small, we can approximate it as a rotation or RA/Dec shift.)
#         # Let's assume we apply it as a RA/Dec shift for this example.
#         current_pass_bias = time_biases[contact_indices]
        
#         # 3. Physics: Geometry -> Angles
#         rel_pos = geometry_data['rel_pos'] # (N, 3) TEME
        
#         # ... (Coordinate transform TEME -> Mount Frame) ...
#         # Here we use the learnable mount angles to rotate the vector
#         # This creates the "Non-linear bias" you mentioned.
#         corrected_pos = self.apply_mount_rotation(rel_pos, mount_euler_angles)
        
#         # 4. Compute Observables
#         x, y, z = corrected_pos.unbind(1)
#         az = torch.atan2(y, x) + current_pass_bias # Apply pass bias
#         el = torch.asin(z / corrected_pos.norm(dim=1))
        
#         return torch.stack([az, el], dim=1)

#     def apply_mount_rotation(self, vec, euler):
#         # Placeholder for 3D rotation logic using the 3 parameters
#         # This makes the bias non-linear w.r.t the state.
#         return vec # + some_rotation(euler)