from dataclasses import dataclass

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import EarthLocation


@dataclass
class GroundStation:
    name: str
    lat: float  # Degrees
    lon: float  # Degrees
    alt: float  # km

    def __post_init__(self):
        # Precompute ECEF position (using Astropy for precision)
        loc = EarthLocation(
            lat=self.lat * u.deg, lon=self.lon * u.deg, height=self.alt * u.km
        )
        # Store as a (3,) float64 tensor
        self.pos_ecef = torch.tensor(
            [loc.x.to(u.km).value, loc.y.to(u.km).value, loc.z.to(u.km).value],
            dtype=torch.float64,
        )


def propagate_stations(
    stations: list[GroundStation],
    t_tai: torch.Tensor,
    station_indices: torch.Tensor,
    epoch_tai: float = 0.0,
):
    """
    Batched propagation of multiple ground stations to TEME frame.

    Args:
        stations: List of GroundStation objects.
        t_tai: (N,) tensor of timestamps (TAI seconds).
        station_indices: (N,) integer tensor mapping each timestamp to a station in the list.
        epoch_tai: Reference epoch for Earth rotation (GMST calculation).

    Returns:
        pos_teme: (N, 3) Position of the specific station at the specific time.
        vel_teme: (N, 3) Velocity of the station (due to Earth rotation).
    """
    # 1. Stack all station ECEF vectors into a single lookup table
    # Shape: (K_stations, 3)
    station_lookup = torch.stack([s.pos_ecef for s in stations]).to(t_tai.device)

    # 2. Select the correct initial position for each measurement
    # Shape: (N, 3)
    pos_ecef_selected = station_lookup[station_indices]

    # 3. Earth Rotation Physics (SGP4/TEME constants)
    w_earth = 7.2921151467e-5  # rad/s

    # Calculate GMST angle (Theta)
    # Note: For high precision, you would calculate GMST0 at epoch_tai using astropy
    # Here we assume t_tai is seconds since the epoch where GMST was 0 (simplified)
    # In production: theta = gmst0 + w * (t - t_epoch)
    theta = w_earth * (t_tai - epoch_tai)

    c = torch.cos(theta)
    s = torch.sin(theta)

    # 4. Rotation (Vectorized for N measurements)
    x_ec = pos_ecef_selected[:, 0]
    y_ec = pos_ecef_selected[:, 1]
    z_ec = pos_ecef_selected[:, 2]

    # Position Rotation
    x_teme = x_ec * c - y_ec * s
    y_teme = x_ec * s + y_ec * c
    z_teme = z_ec

    pos_teme = torch.stack([x_teme, y_teme, z_teme], dim=1)

    # Velocity Rotation (Cross Product: w x r)
    vx_teme = -w_earth * y_teme
    vy_teme = w_earth * x_teme
    vz_teme = torch.zeros_like(z_teme)

    vel_teme = torch.stack([vx_teme, vy_teme, vz_teme], dim=1)

    return pos_teme, vel_teme
