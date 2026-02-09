import numpy as np
import torch
import torch.nn as nn

# We assume astropy is available for the high-precision setup phase
try:
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


class GroundStation(nn.Module):
    """
    Differentiable Ground Station Model.

    Initialization:
        Uses high-precision libraries (Astropy) to convert Geodetic (Lat/Lon/Alt)
        to Earth-Centered Earth-Fixed (ECEF) coordinates.

    Forward:
        Computes the station's position and velocity in the TEME frame
        (True Equator Mean Equinox) at given timestamps.
        This operation is fully differentiable and GPU-optimized.
    """

    def __init__(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
        epoch_tai: float,
        station_id: str,
    ) -> None:
        super().__init__()
        self.station_id = station_id

        if not ASTROPY_AVAILABLE:
            raise ImportError("Astropy is required for GroundStation initialization.")

        # 1. High-Precision ECEF Conversion (Static)
        # This runs once on CPU during setup
        loc = EarthLocation(
            lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_km * u.km
        )

        # Convert to km and store as a registered buffer (moves to GPU automatically)
        xyz = np.array(
            object=[loc.x.to(u.km).value, loc.y.to(u.km).value, loc.z.to(u.km).value]
        )
        self.register_buffer(
            name="pos_ecef", tensor=torch.tensor(data=xyz, dtype=torch.float64)
        )

        # 2. Earth Rotation Rate
        # Consistent with SGP4/WGS-84 definitions (rad/s)
        self.w_earth = 7.2921151467e-5

        # 3. Epoch GMST (Greenwich Mean Sidereal Time)
        # We need the rotation angle of the Earth at the TLE epoch.
        self.set_epoch(epoch_tai=epoch_tai)
        self.gmst0 = 0.0

    def set_epoch(self, epoch_tai: float) -> None:
        """
        Sets the reference epoch (TAI seconds) for Earth rotation.
        """
        self.epoch_tai = epoch_tai

        # Calculate GMST at this epoch using Astropy
        # Convert TAI seconds (since 1970) to MJD TAI
        mjd_tai = 40587.0 + (epoch_tai / 86400.0)
        t = Time(mjd_tai, format="mjd", scale="tai")

        # SGP4 TEME is aligned with GMST
        self.gmst0 = t.sidereal_time("mean", "greenwich").to(u.rad).value

    def forward(self, t_tai: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (Pos, Vel) in TEME frame at t_tai (TAI seconds).

        Args:
            t_tai: Tensor (N,) of TAI seconds.

        Returns:
            pos: Tensor (N, 3) in km
            vel: Tensor (N, 3) in km/s
        """
        # Time difference from station epoch
        t_diff_sec = t_tai - self.epoch_tai

        # Current Rotation Angle: theta = gmst0 + w * t
        theta = self.gmst0 + self.w_earth * t_diff_sec

        # Trigonometry
        c = torch.cos(input=theta)
        s = torch.sin(input=theta)

        # Unpack ECEF position (Scalar)
        x_ec, y_ec, z_ec = self.pos_ecef[0], self.pos_ecef[1], self.pos_ecef[2]  # pyright: ignore[reportIndexIssue]

        # 1. Position Rotation (Rz(theta) @ r_ecef)
        # x_teme = x_ec * cos - y_ec * sin
        # y_teme = x_ec * sin + y_ec * cos
        x_teme = x_ec * c - y_ec * s
        y_teme = x_ec * s + y_ec * c
        z_teme = z_ec * torch.ones_like(theta)  # Broadcast to batch size

        pos = torch.stack([x_teme, y_teme, z_teme], dim=1)

        # 2. Velocity Rotation (Time derivative)
        # Since r_ecef is fixed: v_teme = w x r_teme
        # v_x = -w * y_teme
        # v_y =  w * x_teme
        w = self.w_earth
        vx_teme = -w * y_teme
        vy_teme = w * x_teme
        vz_teme = torch.zeros_like(input=theta)

        vel = torch.stack(tensors=[vx_teme, vy_teme, vz_teme], dim=1)

        return pos, vel
