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

def segment_phase_slips(
    times_unix: torch.Tensor, 
    doppler_hz: torch.Tensor, 
    base_contacts: torch.Tensor, 
    window_size: int = 11,
    n_sigmas: float = 3.0,
    min_jump_hz: float = 30.0
) -> torch.Tensor:
    """
    Scans raw Doppler data using a rolling Hampel filter to isolate jumps,
    and segments the contact indices to assign unique OD biases.
    """
    # 1. Setup the sliding window dimensions
    if window_size % 2 == 0:
        window_size += 1  # Ensure window is odd for a true center point
    half_win = window_size // 2
    
    # 2. Create vectorized rolling windows
    # Shape transitions from (N,) -> (N - window_size + 1, window_size)
    windows = doppler_hz.unfold(0, window_size, 1)
    
    # 3. Calculate rolling median and Median Absolute Deviation (MAD)
    rolling_median = torch.median(windows, dim=1).values
    abs_dev = torch.abs(windows - rolling_median.unsqueeze(1))
    rolling_mad = torch.median(abs_dev, dim=1).values
    
    # 4. Calculate Hampel threshold 
    # (1.4826 scales the MAD to estimate standard deviation for a normal distribution)
    statistical_threshold = n_sigmas * 1.4826 * rolling_mad
    
    # Clamp to a physical minimum to prevent micro-segmenting highly quiet data regions
    effective_threshold = torch.clamp(statistical_threshold, min=min_jump_hz)
    
    # 5. Identify anomalies on the center points of our windows
    center_points = doppler_hz[half_win:-half_win]
    anomaly_mask = torch.abs(center_points - rolling_median) > effective_threshold
    
    # 6. Isolate boundaries: Do not flag windows that cross between different satellite passes
    pass_windows = base_contacts.unfold(0, window_size, 1)
    center_passes = pass_windows[:, half_win].unsqueeze(1)
    # A window is valid only if all points in it belong to the center point's pass
    valid_windows = (pass_windows == center_passes).all(dim=1)
    anomaly_mask &= valid_windows
    
    # 7. Map anomalies back to the original tensor shape
    is_anomaly = torch.zeros_like(doppler_hz, dtype=torch.bool)
    is_anomaly[half_win:-half_win] = anomaly_mask
    
    # 8. Rising edge detection
    # A step-function slip might flag 3 consecutive points. We only want to increment
    # the segment counter ONCE per slip event to avoid shattering the segment.
    rising_edge = is_anomaly.clone()
    rising_edge[1:] = is_anomaly[1:] & ~is_anomaly[:-1]
    
    # 9. Create the new segmented contact array
    cumulative_jumps = torch.cumsum(rising_edge.to(torch.int32), dim=0)
    segmented_contacts = base_contacts + cumulative_jumps
    
    # 10. Remap to ensure indices are strictly contiguous from 0
    _, final_contacts = torch.unique(segmented_contacts, return_inverse=True)
    
    return final_contacts.to(torch.int32)

# import torch
# import numpy as np
# import polars as pl

def load_gmd_to_tensors(file_path, center_freq_hz, device="cpu", dtype=torch.float64):
    """
    Loads a GMAT .gmd file (DSN_TCP) and transforms it into PyTorch tensors.
    
    Args:
        file_path: Path to the .gmd file.
        center_freq_hz: The nominal frequency to subtract (e.g., 2.2e9).
        device: Torch device.
        dtype: Torch data type.
        
    Returns:
        times_unix: Tensor of timestamps (Unix seconds).
        doppler_meas: Tensor of (Raw_Freq - Center_Freq) in Hz.
        contacts: Contiguous integer tensor (0, 1, 2...) identifying discrete passes.
    """
    # 1. Read the raw text file, skipping potential header metadata
    # GMD files are space-separated. Columns: [MJD, Type, ID, {Tags}, Participant1, Participant2, Value]
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7 or parts[1] != "DSN_TCP":
                continue
            # Extract MJD (Column 0) and Raw Frequency (Column 9 in DSN_TCP format)
            data.append([float(parts[0]), float(parts[-1])])

    raw_arr = np.array(data)
    
    # 2. Convert MJD to Unix Seconds
    # GMAT TAIModJulian = MJD + 2430000.5 (JD). 
    # Standard MJD to Unix conversion: (MJD - 10587.0) * 86400
    mjd_vals = raw_arr[:, 0]
    times_unix_np = (mjd_vals - 10587.0) * 86400.0 - 37.0
    
    # 3. Calculate Doppler Shift (Observed - Nominal)
    # Note: GMD DSN_TCP values are often negative in GMAT output; 
    # we take the absolute to get the physical frequency if needed.
    raw_freqs = np.abs(raw_arr[:, 1])
    doppler_hz_np = raw_freqs - center_freq_hz * 1e6
    
    # 4. Generate "Contacts" (Pass-Splitting)
    # If delta_t > 2 seconds, it is a new pass.
    time_diffs = np.diff(times_unix_np, prepend=times_unix_np[0])
    pass_flags = time_diffs > 10.0
    contacts_np = np.cumsum(pass_flags).astype(np.int32)
    
    # 5. Convert to PyTorch Tensors
    times_unix = torch.tensor(times_unix_np, device=device, dtype=dtype)
    doppler_meas = torch.tensor(doppler_hz_np, device=device, dtype=dtype)
    contacts = torch.tensor(contacts_np, device=device, dtype=torch.int32)
    
    print(f"Loaded {len(times_unix)} points across {contacts.max().item() + 1} passes.")
    
    print(doppler_meas.min(), doppler_meas.max())

    return times_unix, doppler_meas, contacts

# Example usage within your pipeline:
# center_freq = 2.2e9 
# t_obs, d_obs, c_obs = load_gmd_to_tensors("output_dsn_tcp_biased.gmd", center_freq)