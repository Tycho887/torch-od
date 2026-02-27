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

# import numpy as np
# import torch
# from astropy import units as u
# from astropy.coordinates import ITRS, TEME, CartesianDifferential, EarthLocation
# from astropy.time import Time


# def station_teme_preprocessor(
#     times_s: np.ndarray,
#     station_ids: np.ndarray,
#     id_to_station: dict[int, np.ndarray],
#     device: torch.device,
#     dtype: torch.dtype,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Vectorized computation of TEME states for moving or multiple ground platforms.
#     All input arrays must be of the same length (N).
#     """

#     # 1. Map station IDs to coordinates vectorially
#     # np.unique finds the unique strings and returns the integer indices
#     # needed to reconstruct the original station_ids array.
#     unique_stations, inverse_indices = np.unique(station_ids, return_inverse=True)

#     # Build a dense lookup array for just the unique stations (M x 3)
#     lookup_array = np.array([id_to_station[station] for station in unique_stations])

#     # Broadcast the lookup array back to the original N length using advanced indexing
#     # This happens instantly at the C-level, yielding an (N x 3) array.
#     coords = lookup_array[inverse_indices]

#     # Extract the individual columns
#     lons_deg = coords[:, 0]
#     lats_deg = coords[:, 1]
#     alts_m = coords[:, 2]

#     # 2. EarthLocation accepts arrays natively
#     location = EarthLocation.from_geodetic(
#         lon=lons_deg * u.deg, lat=lats_deg * u.deg, height=alts_m * u.m
#     )

#     # 3. Time accepts arrays natively
#     t = Time(times_s, format="unix", scale="utc")

#     # 4. Get ITRS positions for all N points
#     itrs_pos = location.get_itrs(obstime=t).cartesian

#     # 5. Create an N-length array of velocity differentials
#     zeros = np.zeros_like(itrs_pos.x.value)
#     itrs_vel = CartesianDifferential(
#         d_x=zeros * (u.km / u.s), d_y=zeros * (u.km / u.s), d_z=zeros * (u.km / u.s)
#     )

#     # 6. Combine and transform
#     itrs_pos_with_vel = itrs_pos.with_differentials(itrs_vel)
#     itrs_geo = ITRS(itrs_pos_with_vel, obstime=t)

#     # Astropy handles the vectorized transformation for all N points at once
#     teme_state = itrs_geo.transform_to(TEME(obstime=t))

#     # Returns (N, 3) arrays for position and velocity
#     pos, vel = (
#         teme_state.cartesian.xyz.to(u.km).value.T,
#         teme_state.velocity.d_xyz.to(u.km / u.s).value.T,
#     )

#     return torch.tensor(data=pos, dtype=dtype, device=device), torch.tensor(
#         data=vel, dtype=dtype, device=device
#     )