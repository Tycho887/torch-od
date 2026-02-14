# --- USER SCRIPT ---
# Assume these modules are imported from the files above
import torch
from dsgp4.tle import TLE
import numpy as np
from torch.func import jacfwd
import time
from diffod.layers import (
    SGP4Layer,
    BiasLayer,
    ResidualStack,
)
import diffod.state as state

from ground_station import station_teme_preprocessor

# ---------------------------------------------------------

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
stations = {
    0: np.array(object=[69.6, 18.9, 0.1]),
    1: np.array(object=[78.2, 15.4, 0.4]),
}

# Mock Data (N=1000 measurements)
# Timestamps (seconds from epoch)
t_obs = torch.linspace(0, 6000, 10000, dtype=torch.float32)
# Station ID for each measurement (0=Tromso, 1=Svalbard)

# Define string tensor

st_indices = torch.zeros(10000, dtype=torch.int32)

st_indices[5009:] = 1
# Pass ID for biases (0=Pass1, 1=Pass2)
pass_indices = torch.zeros(10000, dtype=torch.int32)
pass_indices[2000:] = 1
pass_indices[5000:] = 2
pass_indices[4000:] = 3

t0 = time.time()

station_pos, station_vel = station_teme_preprocessor(
    times_s=t_obs.numpy(),
    station_ids=st_indices.numpy(),
    id_to_station=stations,
)

t1 = time.time()

print(f"Time taken: {t1 - t0:.4f} seconds")

# 2. Define State Vector
# ---------------------------------------------------------
# We want to fit Mean Motion (n) and Inclination (i)
state_def = state.StateDefinition(
    init_tle=init_tle,
    num_measurements=1000,
    fit_ma=True,
    fit_mean_motion=True,
    fit_argp=True,
    fit_bstar=True,
    fit_eccentricity=False,
    fit_inclination=False,
    fit_raan=False,
)

# Add Pass Biases
# This automatically resizes the state vector to include 2 bias parameters
state_def.add_linear_bias(name="doppler_bias", group_indices=pass_indices)

# 1. Assemble Layers
layer_sgp4 = SGP4Layer(timestamps=t_obs, state_def=state_def)
# layer_stat = StationLayer(stations=stations, timestamps=t_obs, station_indices=st_indices)
# layer_phys = DopplerPhysicsLayer(center_freq=2.2e9)
layer_bias = BiasLayer(bias_group=state_def.get_bias_group(name="doppler_bias"))

# 2. Compile Model
model = ResidualStack(
    state_def=state_def,
    bias_group=state_def.get_bias_group(name="doppler_bias")
)

compiled_model = torch.compile(model=model, mode="max-autotune")

# 3. Functional Jacobian Calculation
# x0 is your flat tensor of params
x0 = state_def.get_initial_state()

# Define the function for Jacobian
# jacfwd expects f(x) -> y
def functional_forward(x) -> torch.Tensor:
    return compiled_model(x=x, tsince=t_obs, st_pos=station_pos, st_vel=station_vel, center_freq=2.2e9)

t_total = 0

for i in range(100):

    print(f"--- Iteration {i} ---")

    t0 = time.time()
    # Compute Jacobian (N_obs, N_params)
    # This traces the entire graph: Bias -> Physics -> Geometry -> SGP4 -> Constants -> x
    H = jacfwd(func=functional_forward)(x0)

    t1 = time.time()

    print(f"Time taken: {t1 - t0:.4f} seconds")

    t_total += t1 - t0

print(f"Jacobian H shape: {H.shape}") 
# e.g. (1000, 8) -> 1000 measurements, 6 orbital + 2 bias params

print(H[:, :5])

print(f"Average time taken: {t_total/100:.4f}")
