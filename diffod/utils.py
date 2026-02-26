import torch
from dsgp4.tle import TLE
from dsgp4.util import from_datetime_to_mjd
from dataclasses import dataclass
import torch
# import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime, timezone
# from dsgp4.tle import TLE

@dataclass
class BiasGroup:
    """
    Stores metadata for a specific group of bias parameters.
    """
    name: str
    indices: torch.Tensor  # (N_measurements,) Mapping: meas_idx -> local_param_idx
    global_offset: int     # Where this group starts in the state vector 'x'
    num_params: int        # How many parameters are in this group


def unix_to_tai(time_in: float | torch.Tensor) -> float | torch.Tensor:
    """
    Converts Unix seconds (UTC) to TAI seconds.
    Approximation: Adds 37.0 seconds (Current leap second offset).
    """
    # For a robust implementation, one would use astropy or IERS tables.
    # For differentiable tensor operations, a constant offset is standard for short arcs.
    return time_in + 37.0


def get_tle_epoch(tle: TLE) -> float | torch.Tensor:
    """
    Extracts the TLE epoch and converts it to TAI seconds since 1970-01-01.
    """
    tle_epoch_mjd = from_datetime_to_mjd(datetime_obj=tle._epoch)
    return (tle_epoch_mjd - 40587.0) * 86400.0

# import pandas as pd
# import torch

def load_gmat_csv_block(file_path, tle_epoch_unix, block_sec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads GMAT CSV, converts to meters, and filters data around the TLE epoch.
    """
    # GMAT space-delimited: Day Month Year Time X Y Z VX VY VZ
    df = pd.read_csv(file_path, skiprows=1, 
                     header=None, 
                     sep=r'\s+', # Replaced delim_whitespace=True
                    )
    
    # Combine date/time columns (Assuming indices 0,1,2,3 for time)
    time_str = df[0].astype(str) + " " + df[1] + " " + df[2].astype(str) + " " + df[3]
    df['dt'] = pd.to_datetime(time_str, format='%d %b %Y %H:%M:%S.%f')
    df['unix'] = df['dt'].astype('int64') // 10**6  # Seconds
    
    # print(f"Minimum: {df['unix'].min()} | Maximum: {df['unix'].max()}")

    # Filter by time window (centered or trailing, usually trailing for block_sec)
    mask = (df['unix'] >= tle_epoch_unix) & (df['unix'] <= tle_epoch_unix + block_sec)
    df_block = df[mask].copy()
    
    if df_block.empty:
        raise ValueError(f"No GMAT data found within {block_sec}s of epoch {tle_epoch_unix}. Mean is: {df['unix'].mean()}")

    t_gps_np = df_block['unix'].values
    gps_pos_np = df_block[[4, 5, 6]].values 
    gps_vel_np = df_block[[7, 8, 9]].values 


    
    # Convert NumPy arrays to PyTorch Tensors
    t_gps = torch.from_numpy(np.copy(a=t_gps_np)).to(dtype=torch.float64)
    gps_pos = torch.from_numpy(np.copy(a=gps_pos_np)).to(dtype=torch.float64)
    gps_vel = torch.from_numpy(np.copy(a=gps_vel_np)).to(dtype=torch.float64)
    return t_gps, gps_pos, gps_vel

def transform_tle_to_mee(
    n: torch.Tensor, 
    e: torch.Tensor, 
    i: torch.Tensor, 
    omega: torch.Tensor, 
    raan: torch.Tensor, 
    m: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Transforms standard SGP4 Keplerian elements to Modified Equinoctial Elements.
    """
    # 1. Eccentricity components
    omega_plus_raan = omega + raan
    f = e * torch.cos(omega_plus_raan)
    g = e * torch.sin(omega_plus_raan)
    
    # 2. Inclination components (Note: tan(i/2) becomes singular exactly at i = 180 deg)
    tan_half_i = torch.tan(i / 2.0)
    h = tan_half_i * torch.cos(raan)
    k = tan_half_i * torch.sin(raan)
    
    # 3. Mean Longitude
    L = raan + omega + m
    L = torch.fmod(L, 2.0 * np.pi)
    L = torch.where(L < 0, L + 2.0 * np.pi, L)
    
    return {
        "n": n,
        "f": f,
        "g": g,
        "h": h,
        "k": k,
        "L": L
    }

def transform_mee_to_tle(
    n: torch.Tensor, 
    f: torch.Tensor, 
    g: torch.Tensor, 
    h: torch.Tensor, 
    k: torch.Tensor, 
    L: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Transforms Modified Equinoctial Elements back to standard SGP4 Keplerian elements.
    """
    twopi = 2.0 * np.pi
    
    ecco = torch.sqrt(f**2 + g**2)
    inclo = 2.0 * torch.atan(torch.sqrt(h**2 + k**2))
    
    nodeo = torch.atan2(k, h)
    argp_plus_raan = torch.atan2(g, f)
    argpo = argp_plus_raan - nodeo
    mo = L - argp_plus_raan
    
    # Modulo arithmetic to keep angles strictly in [0, 2pi]
    nodeo = torch.fmod(nodeo, twopi)
    argpo = torch.fmod(argpo, twopi)
    mo = torch.fmod(mo, twopi)
    
    nodeo = torch.where(nodeo < 0, nodeo + twopi, nodeo)
    argpo = torch.where(argpo < 0, argpo + twopi, argpo)
    mo = torch.where(mo < 0, mo + twopi, mo)
    
    return {
        "no_kozai": n,
        "ecco": ecco,
        "inclo": inclo,
        "nodeo": nodeo,
        "argpo": argpo,
        "mo": mo
    }

def unix_to_mjd(unix_seconds: float) -> float:

    # Create a UTC datetime object
    dt = datetime.fromtimestamp(unix_seconds, tz=timezone.utc)

    # Calculate Julian Date (JD)
    jd = dt.timestamp() / 86400.0 + 2440587.5

    # Convert to Modified Julian Date (MJD)
    mjd = jd - 2400000.5

    return mjd
