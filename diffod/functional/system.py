import torch
import torch.nn as nn
from diffod.functional.mlsgp4 import FunctionalMLdSGP4
from diffod.functional.sgp4 import sgp4_propagate
from diffod.physics import apply_linear_bias, compute_doppler
from diffod.state import CalibrationSSV

class SGP4(nn.Module):
    """
    Generates ephemeris (position, velocity) from SGP4 state variables.
    """
    def __init__(
        self,
        ssv,
        dtype=torch.float64, # High precision standard for orbital mechanics
        device=torch.device("cpu"),
        use_pretrained_model: bool = False,
        surrogate_weights_path: str = "models/mldsgp4_example_model.pth",
    ) -> None:
        super().__init__()
        self.ssv = ssv
        self.use_pretrained_model = use_pretrained_model
        
        if self.use_pretrained_model:
            self.surrogate_model = FunctionalMLdSGP4(dtype=dtype, device=device)
            self.surrogate_model.load_model()
            # self.surrogate_model.requires_grad_(requires_grad=True)
            self.model = self.surrogate_model
        else:
            self.model = sgp4_propagate

    def forward(self, x: torch.Tensor, tsince: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sgp4_args = self.ssv.get_functional_args(x)
        sat_pos, sat_vel = self.model(tsince=tsince, **sgp4_args)
        return sat_pos, sat_vel


class MeasurementPipeline(nn.Module):
    """
    A unified pipeline chaining an ephemeris propagator directly into any measurement model.
    """
    def __init__(self, propagator: nn.Module, measurement_model: nn.Module):
        super().__init__()
        self.propagator = propagator
        self.measurement_model = measurement_model

    def forward(
        self,
        x: torch.Tensor,
        tsince: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        
        # 1. Source (Get coordinates)
        sat_pos, sat_vel = self.propagator(x, tsince)
        
        # 2. Sink (Measure)
        # Pass the state, ephemeris, and any measurement-specific arguments
        return self.measurement_model(
            x=x, 
            sat_pos=sat_pos, 
            sat_vel=sat_vel, 
            **kwargs
        )

class DopplerMeasurement(nn.Module):
    """
    Computes Doppler observables and applies biases given arbitrary ephemeris.
    """
    def __init__(self, ssv, bias_group=None) -> None:
        super().__init__()
        self.ssv = ssv
        self.bias_group = bias_group

    def forward(
        self,
        x: torch.Tensor,
        sat_pos: torch.Tensor,
        sat_vel: torch.Tensor,
        st_pos: torch.Tensor,
        st_vel: torch.Tensor,
        center_freq: torch.Tensor,
    ) -> torch.Tensor:
        
        # 1. Physics Projection
        raw_doppler = compute_doppler(
            sat_pos=sat_pos,
            sat_vel=sat_vel,
            st_pos=st_pos,
            st_vel=st_vel,
            center_freq=center_freq,
        )

        # 2. Bias Correction
        if self.bias_group is not None:
            return apply_linear_bias(
                predictions=raw_doppler, x_state=x, bias_group=self.bias_group
            )

        return raw_doppler

class CartesianMeasurement(nn.Module):
    """
    Formats Cartesian ephemeris into a flat 1D observation vector for OD solvers.
    """
    def __init__(
        self, 
        ssv, 
        normalization_r: float = 6378.137,     # Earth Radii in km
        normalization_v: float = 7.905366      # ER/min (approximate)
    ) -> None:
        super().__init__()
        self.ssv = ssv
        self.normalization_r = normalization_r
        self.normalization_v = normalization_v

    def stack(self, x, y) -> torch.Tensor:
        return torch.cat([x, y])

    def forward(
        self,
        x: torch.Tensor,
        sat_pos: torch.Tensor,
        sat_vel: torch.Tensor,
    ) -> torch.Tensor:
        
        # 1. Flatten the (N, 3) tensors into 1D (3N,) arrays and normalize
        pos_flat = sat_pos.flatten() / self.normalization_r
        vel_flat = sat_vel.flatten() / self.normalization_v

        # 2. Concatenate into a single (6N,) observation vector
        return self.stack(pos_flat, vel_flat)

    def format_gps_observations(
        self,
        r_gps: torch.Tensor, 
        v_gps: torch.Tensor, 
    ) -> torch.Tensor:
        
        pos_flat = r_gps.flatten() / self.normalization_r
        vel_flat = v_gps.flatten() / self.normalization_v
        
        return self.stack(pos_flat, vel_flat)
    

class GPSInterpolator(nn.Module):
    """
    Acts as an empirical propagator. Interpolates GPS data at dynamically shifted times.
    """
    def __init__(self, ssv: CalibrationSSV, t_gps_ref: torch.Tensor, r_gps_ref: torch.Tensor, v_gps_ref: torch.Tensor):
        super().__init__()
        self.ssv = ssv
        # Ensure reference data is stored as float64 for precision
        self.t_ref = t_gps_ref.to(torch.float64)
        self.r_ref = r_gps_ref.to(torch.float64)
        self.v_ref = v_gps_ref.to(torch.float64)

    def forward(self, x: torch.Tensor, tsince: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Extract the time offset and shift the evaluation times


        args = self.ssv.get_functional_args(x)
        # print(args["time_offset"])
        t_eval = tsince + args["time_offset"]

        # 2. Find the bounding indices for interpolation
        # searchsorted returns the index of the first element >= t_eval
        idx = torch.searchsorted(sorted_sequence=self.t_ref, input=t_eval)
        
        # Clamp to avoid out-of-bounds if time_offset pushes t_eval past the array ends
        idx = torch.clamp(input=idx, min=1, max=len(self.t_ref) - 1)

        # 3. Gather the bounding points
        t0 = self.t_ref[idx - 1]
        t1 = self.t_ref[idx]
        
        r0 = self.r_ref[idx - 1]
        r1 = self.r_ref[idx]
        
        v0 = self.v_ref[idx - 1]
        v1 = self.v_ref[idx]

        # 4. Compute the continuous interpolation weights
        # weight = (t - t0) / (t1 - t0)
        weight = (t_eval - t0) / (t1 - t0)
        weight = weight.unsqueeze(-1) # Shape (N, 1) for broadcasting over (X, Y, Z)

        # print(t0, t1, t_eval)
        # print(weight)

        # 5. Differentiable Linear Interpolation
        r_interp = r0 + weight * (r1 - r0)
        v_interp = v0 + weight * (v1 - v0)

        print(r_interp, v_interp)

        return r_interp, v_interp