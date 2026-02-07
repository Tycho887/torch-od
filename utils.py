from dsgp4.tle import TLE

def extract_orbit_params(tle: TLE):
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