# --- USER SCRIPT ---
# Assume these modules are imported from the files above
import torch
from dsgp4.tle import TLE
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
t_obs = torch.linspace(0, 6000, 100000)
# Station ID for each measurement (0=Tromso, 1=Svalbard)
st_indices = torch.zeros(100000, dtype=torch.int32)
st_indices[5000:] = 1
# Pass ID for biases (0=Pass1, 1=Pass2)
pass_indices = torch.zeros(100000, dtype=torch.int32)
pass_indices[20000:] = 1
pass_indices[50000:] = 2
pass_indices[40000:] = 3

# 2. Define State Vector
# ---------------------------------------------------------
# We want to fit Mean Motion (n) and Inclination (i)
state_def = state.StateDefinition(
    init_tle=init_tle,
    num_measurements=1000,
    fit_ma=True,
    fit_mean_motion=True,
    fit_argp=True,
    fit_bstar=False,
    fit_eccentricity=False,
    fit_inclination=False,
    fit_raan=False,
)

# Add Pass Biases
# This automatically resizes the state vector to include 2 bias parameters
state_def.add_linear_bias(name="doppler_bias", group_indices=pass_indices)

# 1. Assemble Layers
layer_sgp4 = SGP4Layer(timestamps=t_obs, state_def=state_def)
layer_stat = StationLayer(stations=stations, timestamps=t_obs, station_indices=st_indices)
layer_phys = DopplerPhysicsLayer(center_freq=2.2e9)
layer_bias = BiasLayer(bias_group=state_def.get_bias_group(name="doppler_bias"))

# 2. Compile Model
model = OrbitDeterminationModel(
    sgp4_layer=layer_sgp4, 
    station_layer=layer_stat, 
    physics_layer=layer_phys, 
    bias_layer=layer_bias
)

# 3. Functional Jacobian Calculation
# x0 is your flat tensor of params
x0 = state_def.get_initial_state()

# Define the function for Jacobian
# jacfwd expects f(x) -> y
def functional_forward(x) -> torch.Tensor:
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
