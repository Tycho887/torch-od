import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, TEME, ITRS, CartesianDifferential
from astropy.time import Time

def compute_station_state_analytical(times_s: np.ndarray, lat_deg: float, lon_deg: float, alt_m: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Ground station TEME position and velocity analytically.
    """
    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    t = Time(times_s, format="unix", scale="utc")
    
    # 1. Get the ITRS Cartesian representation (position only)
    itrs_pos = location.get_itrs(obstime=t).cartesian
    
    # 2. Create an array of zero velocities matching the shape of the position
    # This represents 0 km/s velocity relative to the Earth's surface
    zeros = np.zeros_like(itrs_pos.x.value)
    itrs_vel = CartesianDifferential(
        d_x=zeros * (u.km / u.s), 
        d_y=zeros * (u.km / u.s), 
        d_z=zeros * (u.km / u.s)
    )
    
    # 3. Attach the zero velocity to the position
    itrs_pos_with_vel = itrs_pos.with_differentials(itrs_vel)
    
    # 4. Create the ITRS frame with the combined position and velocity
    itrs_geo = ITRS(itrs_pos_with_vel, obstime=t)
    
    # 5. Transform to TEME
    # Astropy will now compute the Earth's rotational velocity automatically
    teme_state = itrs_geo.transform_to(TEME(obstime=t))
    
    # 6. Extract position and velocity arrays
    pos_km = teme_state.cartesian.xyz.to(u.km).value.T
    vel_km_s = teme_state.velocity.d_xyz.to(u.km / u.s).value.T
    
    return pos_km, vel_km_s