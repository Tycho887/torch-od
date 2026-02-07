import torch
import matplotlib.pyplot as plt
import numpy as np
from dsgp4.tle import TLE
from system import OrbitSystem
from state_builder import StateDefinition
from modules import SGP4Layer

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
# A dummy TLE dictionary to initialize the object
tle_dict = {
    'satellite_catalog_number': 12345,
    'epoch_year': 23, 'epoch_days': 1.0, 'b_star': 0.0,
    'mean_motion': 15.5 * np.pi / 43200, 'eccentricity': 0.001, 'inclination': 0.9,
    'raan': 0.0, 'argument_of_perigee': 0.0, 'mean_anomaly': 0.0,
    'mean_motion_first_derivative': 0.0, 'mean_motion_second_derivative': 0.0,
    'classification': 'U', 'ephemeris_type': 0, 'international_designator': '98067A',
    'revolution_number_at_epoch': 100, 'element_number': 999
}
init_tle = TLE(tle_dict)

# Define Simulation Time: 3 Passes
t_minutes = torch.linspace(0, 200, 200)
# Create indices: 0-50 (Pass 0), 50-100 (Pass 1), etc...
contact_indices = torch.zeros(200, dtype=torch.long)
contact_indices[70:130] = 1
contact_indices[130:] = 2

station_data = get_dummy_station_data(t_minutes)

# --- 3. Build System ---
# We want to fit Mean Motion (n), Mean Anomaly (ma) and 3 Pass Biases
state_def = StateDefinition(['n', 'ma'], num_passes=3, init_tle=init_tle)
model = OrbitSystem(state_def, init_tle, station_data)

# --- 4. Generate TRUTH ---
# Create a truth vector. 
# We take the initial defaults, shift 'n' slightly, and add biases.
x_truth = state_def.get_initial_state_vector()
x_truth[0] += 0.001  # Perturb Mean Motion slightly
x_truth[2] = 500.0   # Bias Pass 0: +500 Hz
x_truth[3] = -200.0  # Bias Pass 1: -200 Hz
x_truth[4] = 100.0   # Bias Pass 2: +100 Hz

print(f"Truth Vector: {x_truth}")

# Run Forward Pass (No Gradients needed for data gen)
with torch.no_grad():
    y_truth = model(x_truth, t_minutes, contact_indices)
    
# Add Noise
y_obs = y_truth + torch.randn_like(y_truth) * 10.0

# --- 5. Compute Jacobian at Initial Guess ---
x_guess = state_def.get_initial_state_vector() # Biases are 0 here
H = model.get_jacobian(x_guess, t_minutes, contact_indices)

print(f"\nJacobian Shape: {H.shape}") # Should be (200, 5)
print("Jacobian Sensitivities (First 5 rows):")
print(H[:5])

# --- 6. Quick Solve (One Step Newton-Gauss) ---
# dx = (H'H)^-1 H'(y_obs - y_guess)
with torch.no_grad():
    y_guess = model(x_guess, t_minutes, contact_indices)
    residuals = y_obs - y_guess
    
    # Normal Equations
    lhs = H.T @ H
    rhs = H.T @ residuals
    
    # Solve
    delta_x = torch.linalg.solve(lhs, rhs)
    x_updated = x_guess + delta_x

print("\n--- Results ---")
print(f"Truth Biases:  {x_truth[2:]}")
print(f"Solved Biases: {x_updated[2:]}")