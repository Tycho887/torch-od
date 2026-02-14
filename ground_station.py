import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, TEME, ITRS, CartesianDifferential
from astropy.time import Time
import torch

def station_teme_preprocessor(times_s: np.ndarray, station_ids: np.ndarray, id_to_station: dict[int, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized computation of TEME states for moving or multiple ground platforms.
    All input arrays must be of the same length (N).
    """

    # 1. Map station IDs to coordinates vectorially
    # np.unique finds the unique strings and returns the integer indices 
    # needed to reconstruct the original station_ids array.
    unique_stations, inverse_indices = np.unique(station_ids, return_inverse=True)
    
    # Build a dense lookup array for just the unique stations (M x 3)
    lookup_array = np.array([id_to_station[station] for station in unique_stations])
    
    # Broadcast the lookup array back to the original N length using advanced indexing
    # This happens instantly at the C-level, yielding an (N x 3) array.
    coords = lookup_array[inverse_indices]
    
    # Extract the individual columns
    lons_deg = coords[:, 0]
    lats_deg = coords[:, 1]
    alts_m   = coords[:, 2]

    # 2. EarthLocation accepts arrays natively
    location = EarthLocation.from_geodetic(
        lon=lons_deg * u.deg, 
        lat=lats_deg * u.deg, 
        height=alts_m * u.m
    )
    
    # 3. Time accepts arrays natively
    t = Time(times_s, format="unix", scale="utc")
    
    # 4. Get ITRS positions for all N points
    itrs_pos = location.get_itrs(obstime=t).cartesian
    
    # 5. Create an N-length array of velocity differentials
    zeros = np.zeros_like(itrs_pos.x.value)
    itrs_vel = CartesianDifferential(
        d_x=zeros * (u.km / u.s),
        d_y=zeros * (u.km / u.s),
        d_z=zeros * (u.km / u.s)
    )
    
    # 6. Combine and transform
    itrs_pos_with_vel = itrs_pos.with_differentials(itrs_vel)
    itrs_geo = ITRS(itrs_pos_with_vel, obstime=t)
    
    # Astropy handles the vectorized transformation for all N points at once
    teme_state = itrs_geo.transform_to(TEME(obstime=t))
    
    # Returns (N, 3) arrays for position and velocity
    pos, vel = teme_state.cartesian.xyz.to(u.km).value.T, teme_state.velocity.d_xyz.to(u.km / u.s).value.T

    return torch.tensor(data=pos), torch.tensor(data=vel)