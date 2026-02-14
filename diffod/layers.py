import torch
import torch.nn as nn
# from diffod.gse import propagate_stations
from diffod.physics import apply_linear_bias, compute_doppler
from diffod.functional.sgp4 import sgp4_propagate
from diffod.utils import BiasGroup

# Ensure this matches the definition in your functional implementation
class GravConsts:
    tumin = 1.0 / 13.446839
    mu = 398600.8
    radiusearthkm = 6378.135
    xke = 0.0743669161
    j2 = 0.001082616
    j3 = -0.00000253881
    j4 = -0.00000165597
    j3oj2 = j3 / j2

class SGP4Layer(nn.Module):
    def __init__(self, timestamps: torch.Tensor, state_def) -> None:
        """
        Layer 1: Maps State Vector -> Satellite Kinematics (TEME)
        
        Args:
            timestamps: Tensor (N,) of 'time since epoch' in minutes.
            state_def: StateDefinition object containing TLE metadata and mapping.
        """
        super().__init__()
        # Register timestamps as a buffer so they are moved to device automatically
        # but not treated as a learnable parameter.
        self.register_buffer('tsince', timestamps) 
        self.state_def = state_def
        self.consts = GravConsts()

    def forward(self, x_full) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagates SGP4 using the functional implementation.
        """
        # 1. Prepare Arguments
        # Extracts tensors from x_full for active params, and static tensors for others
        sgp4_args = self.state_def.get_functional_args(x_full)
        
        # 2. Functional Propagation
        # We unpack (**sgp4_args) into the function.
        # This is compatible with torch.compile / torch.jit
        pos, vel = sgp4_propagate(
            tsince=self.tsince,  # pyright: ignore[reportArgumentType]
            # consts=self.consts, # Uncomment if your sgp4_propagate accepts consts
            **sgp4_args
        )
        
        # Output: (N, 3), (N, 3)
        return pos, vel

# The rest of your pipeline (StationLayer, DopplerPhysicsLayer, BiasLayer) 
# remains compatible as long as the shapes match.

# class StationLayer(nn.Module):
#     def __init__(self, stations, timestamps, station_indices) -> None:
#         """
#         Layer 2: Provides Ground Station Kinematics (TEME)
#         """
#         super().__init__()
#         self.stations = stations
#         self.register_buffer(name='timestamps', tensor=timestamps)
#         self.register_buffer(name='station_indices', tensor=station_indices)

#     def forward(self, sat_pos, sat_vel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         # This layer is technically independent of the satellite, 
#         # but we chain it here to align the data flow.
        
#         st_pos, st_vel = propagate_stations(
#             stations=self.stations,
#             t_tai=self.timestamps,               # pyright: ignore[reportArgumentType]
#             station_indices=self.station_indices # pyright: ignore[reportArgumentType]
#         )
        
#         return sat_pos, sat_vel, st_pos, st_vel

# class DopplerPhysicsLayer(nn.Module):
#     def __init__(self, center_freq) -> None:
#         """
#         Layer 3: Geometry -> Physics (Doppler)
#         Uses Einops for clarity.
#         """
#         super().__init__()
#         self.center_freq = center_freq
#         self.c_light = 299792.458

#     def forward(self, inputs) -> torch.Tensor:
#         sat_pos, sat_vel, st_pos, st_vel = inputs
        
#         # Einops Pattern: 
#         # Ensure inputs are (Batch, 3). 
#         # We can enforce shapes or rearrange if needed.
        
#         # 1. Relative State
#         r_rel = sat_pos - st_pos
#         v_rel = sat_vel - st_vel
        
#         # 2. Range & Look Vector
#         # "b c -> b 1" keeps dimensions for broadcasting
#         dist = torch.norm(r_rel, dim=1, keepdim=True) 
#         u_los = r_rel / (dist + 1e-9)
        
#         # 3. Range Rate (Projection)
#         # Sum over coordinate dimension 'c'
#         # "b c, b c -> b"
#         range_rate = (v_rel * u_los).sum(dim=1)
        
#         # 4. Doppler
#         # f_d = - (rr / c) * f_c
#         doppler_pred = -(range_rate / self.c_light) * self.center_freq
        
#         return doppler_pred

class BiasLayer(nn.Module):
    def __init__(self, bias_group: BiasGroup) -> None:
        super().__init__()
        self.bias_group = bias_group

    def forward(self, predictions, x_state) -> torch.Tensor:
        return apply_linear_bias(predictions=predictions, x_state=x_state, bias_group=self.bias_group)
    
class ResidualStack(nn.Module):
    """
    A strictly stateless pipeline. 
    Safe to instantiate once, compile once, and share across API threads.
    """
    def __init__(self, state_def, bias_group=None) -> None:
        super().__init__()
        # Only structural metadata is kept here. No observation tensors!
        self.state_def = state_def
        self.bias_group = bias_group

    def forward(self, x: torch.Tensor, tsince: torch.Tensor, 
                st_pos: torch.Tensor, st_vel: torch.Tensor, 
                center_freq: torch.Tensor) -> torch.Tensor:
        """
        All observation data is injected at inference time.
        """
        # 1. Propagate Orbit
        sgp4_args = self.state_def.get_functional_args(x)
        sat_pos, sat_vel = sgp4_propagate(tsince=tsince, **sgp4_args)
        
        # 2. Physics Projection 
        raw_doppler = compute_doppler(sat_pos=sat_pos, sat_vel=sat_vel, st_pos=st_pos, st_vel=st_vel, center_freq=center_freq)
        
        # 3. Bias Correction
        if self.bias_group is not None:
            return apply_linear_bias(predictions=raw_doppler, x_state=x, bias_group=self.bias_group)
        return raw_doppler