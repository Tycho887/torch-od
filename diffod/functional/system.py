import torch
import torch.nn as nn
import math
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
        center_freq: torch.Tensor | float,
    ) -> torch.Tensor:
        
        args = self.ssv.get_functional_args(x)

        # if self.ssv.active_flags["time_offset"]:
        f_c = center_freq + (args["freq_offset"] * 1e6)
        # 1. Physics Projection
        raw_doppler = compute_doppler(
            sat_pos=sat_pos,
            sat_vel=sat_vel,
            st_pos=st_pos,
            st_vel=st_vel,
            center_freq=f_c,
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
    
class DifferentiableStation(torch.nn.Module):
    """
    A PyTorch-native Ground Station model.
    Transforms WGS84 Geodetic to ECEF, and dynamically rotates to TEME 
    to preserve gradients through time shifts.
    """
    def __init__(
        self, 
        lat_deg: float, 
        lon_deg: float, 
        alt_m: float, 
        ref_unix: float, 
        ref_gmst_rad: float,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.omega_earth = 7.292115146706979e-5  # rad/s (Nominal WGS84 rotation)
        
        # Store Reference Anchors
        self.ref_unix = torch.tensor(ref_unix, dtype=torch.float64, device=device)
        self.ref_gmst_rad = torch.tensor(ref_gmst_rad, dtype=torch.float64, device=device)
        
        # 1. WGS84 Geodetic to ECEF
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        alt_km = alt_m / 1000.0
        
        a = 6378.137  # km
        e2 = 0.00669437999014
        
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        x = (N + alt_km) * math.cos(lat) * math.cos(lon)
        y = (N + alt_km) * math.cos(lat) * math.sin(lon)
        z = (N * (1 - e2) + alt_km) * math.sin(lat)
        
        # Store static ECEF position
        self.r_ecef = torch.tensor([x, y, z], dtype=torch.float64, device=device)
        
    def forward(self, t_unix_shifted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes shifted timestamps and rotates the station in a fully differentiable manner.
        """
        # 1. Compute Differentiable Earth Rotation Angle (theta)
        dt = t_unix_shifted - self.ref_unix
        theta = self.ref_gmst_rad + self.omega_earth * dt
        
        # 2. Construct Rotation Matrix Rz(theta)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)
        
        # Shape: (N, 3, 3)
        Rz = torch.stack([
            torch.stack([cos_t, -sin_t, zeros], dim=-1),
            torch.stack([sin_t,  cos_t, zeros], dim=-1),
            torch.stack([zeros,  zeros,  ones], dim=-1)
        ], dim=-2)
        
        # 3. Rotate ECEF to Inertial (TEME)
        # Broadcasting matrix multiplication: (N, 3, 3) @ (3,) -> (N, 3)
        r_teme = torch.matmul(Rz, self.r_ecef)
        
        # 4. Explicit Kinematic Velocity: v = omega x r
        omega_vec = torch.zeros_like(r_teme)
        omega_vec[:, 2] = self.omega_earth
        v_teme = torch.cross(omega_vec, r_teme, dim=-1)
        
        return r_teme, v_teme