import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import EarthLocation, TEME
from astropy.time import Time

def station_teme_preprocessor(
    times_s: np.ndarray,
    station_ids: np.ndarray,
    id_to_station: dict[int, np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes TEME position via Astropy, but calculates velocity explicitly 
    using the Earth's angular velocity vector to guarantee rotational kinematics.
    """
    # 1. Map station IDs to coordinates
    unique_stations, inverse_indices = np.unique(station_ids, return_inverse=True)
    lookup_array = np.array([id_to_station[station] for station in unique_stations])
    coords = lookup_array[inverse_indices]

    lons_deg = coords[:, 0]
    lats_deg = coords[:, 1]
    alts_m = coords[:, 2]

    # 2. Astropy Time and Location
    location = EarthLocation.from_geodetic(
        lon=lons_deg * u.deg, lat=lats_deg * u.deg, height=alts_m * u.m
    )
    t = Time(times_s, format="unix", scale="utc")

    # 3. Get Position in TEME
    # We only ask Astropy for the position, ensuring the GMST angle is applied correctly.
    itrs_pos = location.get_itrs(obstime=t)
    teme_pos = itrs_pos.transform_to(TEME(obstime=t))
    r_teme_km = teme_pos.cartesian.xyz.to(u.km).value.T  # Shape: (N, 3)

    # 4. Explicitly Compute Inertial Velocity
    # v_inertial = w_earth x r_inertial
    # WGS84 nominal Earth rotation rate in radians per second
    OMEGA_EARTH = 7.292115146706979e-5 
    
    # Create the rotation vector [0, 0, w] for all N points
    omega_vec = np.zeros_like(r_teme_km)
    omega_vec[:, 2] = OMEGA_EARTH

    # Vectorized Cross Product
    v_teme_km_s = np.cross(omega_vec, r_teme_km)

    # 5. Convert to PyTorch Tensors
    pos_tensor = torch.tensor(data=r_teme_km, dtype=dtype, device=device)
    vel_tensor = torch.tensor(data=v_teme_km_s, dtype=dtype, device=device)

    return pos_tensor, vel_tensor

def compute_station_teme(
    t_unix_shifted: torch.Tensor, 
    ref_unix: torch.Tensor, 
    ref_gmst_rad: torch.Tensor, 
    r_ecef: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure functional transformation from ECEF to TEME."""
    omega_earth = 7.292115146706979e-5
    dt = t_unix_shifted - ref_unix
    theta = ref_gmst_rad + omega_earth * dt
    
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    
    Rz = torch.stack([
        torch.stack([cos_t, -sin_t, zeros], dim=-1),
        torch.stack([sin_t,  cos_t, zeros], dim=-1),
        torch.stack([zeros,  zeros,  ones], dim=-1)
    ], dim=-2)
    
    r_teme = torch.matmul(Rz, r_ecef)
    
    omega_vec = torch.zeros_like(r_teme)
    omega_vec[:, 2] = omega_earth
    v_teme = torch.cross(omega_vec, r_teme, dim=-1)
    
    return r_teme, v_teme