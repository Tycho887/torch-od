import torch
import torch.nn as nn
from diffod.gse import propagate_stations
from diffod.physics import apply_linear_bias
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

class StationLayer(nn.Module):
    def __init__(self, stations, timestamps, station_indices) -> None:
        """
        Layer 2: Provides Ground Station Kinematics (TEME)
        """
        super().__init__()
        self.stations = stations
        self.register_buffer(name='timestamps', tensor=timestamps)
        self.register_buffer(name='station_indices', tensor=station_indices)

    def forward(self, sat_pos, sat_vel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # This layer is technically independent of the satellite, 
        # but we chain it here to align the data flow.
        
        st_pos, st_vel = propagate_stations(
            stations=self.stations,
            t_tai=self.timestamps,               # pyright: ignore[reportArgumentType]
            station_indices=self.station_indices # pyright: ignore[reportArgumentType]
        )
        
        return sat_pos, sat_vel, st_pos, st_vel

class DopplerPhysicsLayer(nn.Module):
    def __init__(self, center_freq) -> None:
        """
        Layer 3: Geometry -> Physics (Doppler)
        Uses Einops for clarity.
        """
        super().__init__()
        self.center_freq = center_freq
        self.c_light = 299792.458

    def forward(self, inputs) -> torch.Tensor:
        sat_pos, sat_vel, st_pos, st_vel = inputs
        
        # Einops Pattern: 
        # Ensure inputs are (Batch, 3). 
        # We can enforce shapes or rearrange if needed.
        
        # 1. Relative State
        r_rel = sat_pos - st_pos
        v_rel = sat_vel - st_vel
        
        # 2. Range & Look Vector
        # "b c -> b 1" keeps dimensions for broadcasting
        dist = torch.norm(r_rel, dim=1, keepdim=True) 
        u_los = r_rel / (dist + 1e-9)
        
        # 3. Range Rate (Projection)
        # Sum over coordinate dimension 'c'
        # "b c, b c -> b"
        range_rate = (v_rel * u_los).sum(dim=1)
        
        # 4. Doppler
        # f_d = - (rr / c) * f_c
        doppler_pred = -(range_rate / self.c_light) * self.center_freq
        
        return doppler_pred

class BiasLayer(nn.Module):
    def __init__(self, bias_group: BiasGroup) -> None:
        super().__init__()
        self.bias_group = bias_group

    def forward(self, predictions, x_state) -> torch.Tensor:
        return apply_linear_bias(predictions=predictions, x_state=x_state, bias_group=self.bias_group)
    
class OrbitDeterminationModel(nn.Module):
    def __init__(self, sgp4_layer, station_layer, physics_layer, bias_layer) -> None:
        super().__init__()
        self.sgp4 = sgp4_layer
        self.station = station_layer
        self.physics = physics_layer
        self.bias = bias_layer
        
    def forward(self, x) -> torch.Tensor:
        """
        The Differentiable Pipeline.
        Input: State Vector x
        Output: Biased Residuals (or Predictions)
        """
        
        # 1. Orbit Propagation (Uses x part A)
        pos, vel = self.sgp4(x)
        
        # 2. Station Geometry (Static Context)
        # Returns tuple of (sat, st) states
        geometry_inputs = self.station(pos, vel)
        
        # 3. Physics Projection (Doppler/Range)
        # Pure physics, no learnable params usually
        raw_preds = self.physics(geometry_inputs)
        
        # 4. Bias Correction (Uses x part B)
        final_preds = self.bias(raw_preds, x)
        
        return final_preds