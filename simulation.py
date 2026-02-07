import torch
import matplotlib.pyplot as plt
import numpy as np
from dsgp4.tle import TLE
from system import DopplerSensor, RangeSensor, MultiSensorSystem
from state_builder import StateDefinition
from modules import SGP4Layer
import dsgp4
from utils import extract_orbit_params, list_elements

# --- 1. Fake Station Data (TEME) ---
def get_dummy_station_data(t_tensor):
    # Just a placeholder for the pre-computation step
    # Rotating vector on Earth surface
    theta = t_tensor * 0.05 # Earth rotation approx
    r = 6378.0
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.zeros_like(t_tensor)
    pos = torch.stack([x, y, z], dim=1)
    vel = torch.stack([-y, x, z], dim=1) * 0.05
    return {'pos': pos, 'vel': vel}

# --- 2. Setup ---
TLE_list = ["ISS (ZARYA)",
"1 25544U 98067A   26038.50283897  .00012054  00000-0  23050-3 0  9996",
"2 25544  51.6315 221.5822 0011000  74.6214 285.5989 15.48462076551652"]

init_tle = TLE(TLE_list)

list_elements(init_tle)

# Define Simulation Time: 3 Passes
t_minutes = torch.linspace(0, 200, 200)
# Create indices: 0-50 (Pass 0), 50-100 (Pass 1), etc...
contact_indices = torch.zeros(200, dtype=torch.long)
contact_indices[70:130] = 0
contact_indices[130:] = 1

station_data = get_dummy_station_data(t_minutes)

# --- 3. Build System ---
# We want to fit Mean Motion (n), Mean Anomaly (ma) and 3 Pass Biases
state_def = StateDefinition(
    orbital_keys=['n', 'ma'], 
    sensor_config={'doppler': 2}, # 2 passes for Doppler
    init_tle=init_tle
)

st_pos = station_data['pos']
st_vel = station_data['vel']

# Sensors
sensors = {'doppler': DopplerSensor(st_pos, st_vel, 435e6)}
system = MultiSensorSystem(state_def, init_tle, sensors)

# Input Data
obs_data = {
    'doppler': (t_minutes, contact_indices) 
}

# Initial State
x0 = state_def.get_initial_state()

results = system.forward(x0, obs_data)

# x0 structure: [n, ma, bias_pass_0, bias_pass_1]

# Compute Jacobian
H = system.get_jacobian(x0, obs_data)

# --- DEBUGGING THE JACOBIAN ---
# H shape: (N_obs, 4)
# Column 0: d(Dop)/dn
# Column 1: d(Dop)/dma
# Column 2: d(Dop)/dBias0
# Column 3: d(Dop)/dBias1

print("Checking Bias Gradients...")
# Get indices where we observe Pass 0
pass0_mask = (contact_indices == 0)
bias0_grads = H[pass0_mask, 2] 

print(f"Mean Gradient for Bias 0: {bias0_grads.mean().item()}")
# Should print 1.0 exactly. 
# If it prints 0.0, check if contact_indices match the bias slice.

print(f"Jacobian shape: {H.shape}")
print(f"First 5 rows of Jacobian:\n{H[:5, :]}")