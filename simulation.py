# --- USER SCRIPT ---
# Assume these modules are imported from the files above
import torch
from dsgp4.newton_method import newton_method
from dsgp4.tle import TLE
# from dsgp4.mldsgp4 import mldsgp4
from torch.func import jacfwd
import time
from diffod.layers import (
    SGP4Layer,
    StationLayer,
    DopplerPhysicsLayer,
    BiasLayer,
    OrbitDeterminationModel,
)
import diffod.gse as gse
import diffod.state as state
# from diffod.functional.tle import dsgp4, mldsgp4

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
stations = [
    gse.GroundStation(name="Tromso", lat=69.6, lon=18.9, alt=0.1),
    gse.GroundStation(name="Svalbard", lat=78.2, lon=15.4, alt=0.4),
]

# Mock Data (N=1000 measurements)
# Timestamps (seconds from epoch)
t_obs = torch.linspace(0, 6000, 10000)
# Station ID for each measurement (0=Tromso, 1=Svalbard)
st_indices = torch.zeros(10000, dtype=torch.int32)
st_indices[500:] = 1
# Pass ID for biases (0=Pass1, 1=Pass2)
pass_indices = torch.zeros(10000, dtype=torch.int32)
pass_indices[2000:] = 1
pass_indices[5000:] = 2
pass_indices[4000:] = 3

# 2. Define State Vector
# ---------------------------------------------------------
# We want to fit Mean Motion (n) and Inclination (i)
state_def = state.StateDefinition(
    init_tle=init_tle,
    num_measurements=1000,
    fit_ma=True,
    fit_mean_motion=False,
    fit_argp=False,
    fit_bstar=False,
    fit_eccentricity=False,
    fit_inclination=False,
    fit_raan=False,
)

# Add Pass Biases
# This automatically resizes the state vector to include 2 bias parameters
state_def.add_linear_bias(name="doppler_bias", group_indices=pass_indices)

# 1. Assemble Layers
layer_sgp4 = SGP4Layer(init_tle, t_obs, state_def)
layer_stat = StationLayer(stations, t_obs, st_indices)
layer_phys = DopplerPhysicsLayer(center_freq=2.2e9)
layer_bias = BiasLayer(bias_group=state_def.get_bias_group("doppler_bias"))

# 2. Compile Model
model = OrbitDeterminationModel(
    layer_sgp4, 
    layer_stat, 
    layer_phys, 
    layer_bias
)

# 3. Functional Jacobian Calculation
# x0 is your flat tensor of params
x0 = state_def.get_initial_state()

# Define the function for Jacobian
# jacfwd expects f(x) -> y
def functional_forward(x):
    return model(x)
t0 = time.time()
# Compute Jacobian (N_obs, N_params)
# This traces the entire graph: Bias -> Physics -> Geometry -> SGP4 -> Constants -> x
H = jacfwd(functional_forward)(x0)

t1 = time.time()

print(f"Jacobian H shape: {H.shape}") 
# e.g. (1000, 8) -> 1000 measurements, 6 orbital + 2 bias params

print(H[:, :5])

print(f"Time taken: {t1 - t0:.4f}")
# # Initial Guess (x0)
# # We need to construct x0 based on the dimensions sv_def calculated
# x0 = sv_def.get_initial_state()

# # Pre-compute the selection matrix (Constant during optimization)
# # H_bias = sv_def.get_bias_matrix("doppler_bias")

# t0 = time.time()

# station_pos, station_vel = gse.propagate_stations(
#     stations=stations, t_tai=t_obs, station_indices=st_indices
# )

# t1 = time.time()

# print(f"Time taken: {t1 - t0:.4f}")
# expected = 1000 * torch.sin(input=t_obs)


# def objective(x):

#     t0 = time.time()

#     ml

#     pos, vel = dsgp4(tle=init_tle, timestamps=t_obs, x=x, state_def=sv_def)

#     t1 = time.time()

#     print(f"Time taken: {t1 - t0:.4f}")

#     t0 = time.time()

#     measured = physics.compute_doppler(
#         sat_pos=pos,
#         sat_vel=vel,
#         st_pos=station_pos,
#         st_vel=station_vel,
#         center_freq=2.2e9,
#     )

#     observed = physics.apply_linear_bias(
#         measured, x, sv_def.get_bias_map("doppler_bias")
#     )

#     residuals = observed - expected

#     t1 = time.time()

#     print(f"Time taken: {t1 - t0:.4f}")

#     return residuals


# t0 = time.time()

# H = jacfwd(func=objective)(x0)
# t1 = time.time()
# # H = torch.autograd.functional.jacobian(objective, x0)

# print(H)
# print(f"Time taken: {t1 - t0:.4f}")
