# --- USER SCRIPT ---
# Assume these modules are imported from the files above
import dsgp4
import torch
from dsgp4.tle import TLE

import diffod.gse as gse
import diffod.physics as physics
import diffod.state as state

# 1. Setup Data
# ---------------------------------------------------------
# Load TLE
TLE_list = [
    "ISS (ZARYA)",
    "1 25544U 98067A   26038.50283897  .00012054  00000-0  23050-3 0  9996",
    "2 25544  51.6315 221.5822 0011000  74.6214 285.5989 15.48462076551652",
]
init_tle = TLE(data=TLE_list)
# Create Stations
stations = [
    gse.GroundStation(name="Tromso", lat=69.6, lon=18.9, alt=0.1),
    gse.GroundStation(name="Svalbard", lat=78.2, lon=15.4, alt=0.4),
]

# Mock Data (N=1000 measurements)
# Timestamps (seconds from epoch)
t_obs = torch.linspace(0, 6000, 1000)
# Station ID for each measurement (0=Tromso, 1=Svalbard)
st_indices = torch.zeros(1000, dtype=torch.long)
st_indices[500:] = 1
# Pass ID for biases (0=Pass1, 1=Pass2)
pass_indices = torch.zeros(1000, dtype=torch.long)
pass_indices[500:] = 1

# 2. Define State Vector
# ---------------------------------------------------------
# We want to fit Mean Motion (n) and Inclination (i)
sv_def = state.StateDefinition(["mean_motion", "inclination"], num_measurements=1000)

# Add Pass Biases
# This automatically resizes the state vector to include 2 bias parameters
sv_def.add_linear_bias("doppler_bias", pass_indices)

# Initial Guess (x0)
# We need to construct x0 based on the dimensions sv_def calculated
x0 = torch.zeros(sv_def.current_dim, requires_grad=True)
# Fill initial orbital values from TLE
x0.data[0] = init_tle.mean_motion
x0.data[1] = init_tle.inclination
# Biases start at 0.0

# Pre-compute the selection matrix (Constant during optimization)
H_bias = sv_def.get_bias_matrix("doppler_bias")


# 3. System Function (The Forward Pass)
# ---------------------------------------------------------
def system_model(x):
    # A. Update TLE
    # Create a dict of params, overriding the ones we are fitting
    current_params = sv_def.unpack_tle_params(
        x,
        defaults={
            "mean_motion": init_tle.mean_motion,
            "inclination": init_tle.inclination,
            "eccentricity": init_tle.eccentricity,
            "raan": init_tle.raan,
            "argp": init_tle.argument_of_perigee,
            "ma": init_tle.mean_anomaly,
            "bstar": init_tle.bstar,
            # ... add others
        },
    )

    # B. Propagate Satellite
    # Note: dSGP4 typically needs specific formatting, assume a helper exists
    # or dSGP4 is modified to accept the dict/tensor directly.
    # For this example, assuming dsgp4.propagate_batch accepts the dict
    sat_pos_teme, sat_vel_teme = dsgp4.propagate_batch(current_params, t_obs)

    # C. Propagate Stations
    # (Only depends on time and Earth rotation, not the satellite state)
    st_pos_teme, st_vel_teme = gse.propagate_stations(stations, t_obs, st_indices)

    # D. Compute Physics (Doppler)
    center_freq = 2.2e9  # Hz
    pred_doppler = physics.compute_doppler(
        sat_pos_teme, sat_vel_teme, st_pos_teme, st_vel_teme, center_freq
    )

    # E. Apply Biases
    # Uses the sparse matrix H_bias we built earlier
    final_pred = physics.apply_linear_bias(pred_doppler, x, H_bias)

    return final_pred


# 4. Jacobian & Optimization
# ---------------------------------------------------------
from torch.func import jacfwd

# Compute Jacobian
J = jacfwd(system_model)(x0)

print(f"Jacobian Shape: {J.shape}")
# Should be (1000, 4) -> (N_meas, [n, i, bias_pass_0, bias_pass_1])
