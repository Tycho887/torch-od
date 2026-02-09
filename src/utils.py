from dsgp4.tle import TLE
from dsgp4.util import from_datetime_to_mjd
import torch

def extract_orbit_params(tle: TLE) -> dict[str, torch.Tensor | float | int]:
    """
    Helper to convert a TLE object into a dictionary of parameters.
    This is used to populate 'static_tle_params' and can be reused in unpacking.
    """
    return {
        'satellite_catalog_number': tle.satellite_catalog_number,
        'epoch_year': tle.epoch_year,
        'epoch_days': tle.epoch_days,
        'b_star': tle._bstar,
        'mean_motion': tle.mean_motion, 
        'eccentricity': tle.eccentricity, 
        'inclination': tle.inclination, 
        'raan': tle.raan, 
        'argument_of_perigee': tle.argument_of_perigee, 
        'mean_anomaly': tle.mean_anomaly, 
        'mean_motion_first_derivative': tle.mean_motion_first_derivative,
        'mean_motion_second_derivative': tle.mean_motion_second_derivative,
        'classification': tle.classification,
        'ephemeris_type': tle.ephemeris_type,
        'international_designator': tle.international_designator,
        'revolution_number_at_epoch': tle.revolution_number_at_epoch,
        'element_number': tle.element_number
    }

def list_elements(tle: TLE) -> None:

    elements = extract_orbit_params(tle=tle)

    for key, value in elements.items():
        try:
            print(f"{key}: {float(value):.6f}")
        except:
            print(f"{key}: {value}")

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