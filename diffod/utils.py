import torch
from dsgp4.tle import TLE
from dsgp4.util import from_datetime_to_mjd
from dataclasses import dataclass

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


def get_tle_epoch_tai(tle: TLE) -> float | torch.Tensor:
    """
    Extracts the TLE epoch and converts it to TAI seconds since 1970-01-01.
    """
    tle_epoch_mjd = from_datetime_to_mjd(datetime_obj=tle._epoch)
    unix_utc = (tle_epoch_mjd - 40587.0) * 86400.0
    return unix_to_tai(time_in=unix_utc)
