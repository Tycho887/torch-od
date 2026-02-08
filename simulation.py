import torch
import matplotlib.pyplot as plt
import numpy as np
from dsgp4.tle import TLE
from src.systemObject import BaseOrbitSystem
from src.sensorLayers import DopplerSensor, RadarSensor

# --- 1. Helper: Dummy Station ---
def get_station_vectors(t_minutes):
    # Simple Earth Rotation model
    # rad/min = 2pi / 1440
    w_earth = (2 * np.pi) / 1440.0 
    theta = w_earth * t_minutes
    r = 6378.0
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.zeros_like(t_minutes)
    
    pos = torch.stack([x, y, z], dim=1)
    # Velocity (tangential)
    vx = -r * w_earth * torch.sin(theta)
    vy =  r * w_earth * torch.cos(theta)
    vz = torch.zeros_like(t_minutes)
    vel = torch.stack([vx, vy, vz], dim=1)
    
    return pos, vel

# --- 2. Setup Scenario ---
TLE_list = ["ISS (ZARYA)",
"1 25544U 98067A   26038.50283897  .00012054  00000-0  23050-3 0  9996",
"2 25544  51.6315 221.5822 0011000  74.6214 285.5989 15.48462076551652"]
init_tle = TLE(TLE_list)

# Time: 200 minutes
t_all = torch.linspace(0, 200, 200)

# Station Ephemeris
st_pos, st_vel = get_station_vectors(t_all)

# --- 3. Define Sensors ---
# Doppler: 2 passes, fit center freq
doppler_sensor = DopplerSensor(st_pos, st_vel, center_freq=435e6, num_passes=2, fit_center_freq=True)
# Radar: 2 passes (same times)
radar_sensor = RadarSensor(st_pos, num_passes=2)

sensors = {'doppler': doppler_sensor, 'radar': radar_sensor}

# --- 4. Define Observation Packets ---
# Split data: 0-100 min (Pass 0), 100-200 min (Pass 1)
indices = torch.zeros(200, dtype=torch.long)
indices[100:] = 1

obs_data = {
    'doppler': {'t': t_all, 'indices': indices},
    'radar':   {'t': t_all, 'indices': indices}
}

# --- 5. Build System ---
system = BaseOrbitSystem(init_tle, fit_keys=['n', 'i'], sensors_dict=sensors)

# --- 6. Jacobian Analysis ---
x0 = system.state_def.get_initial_state()
# x0 layout: 
# [ n, i, 
#   dop_bias_0, dop_bias_1, dop_freq_bias, 
#   rad_bias_0, rad_bias_1 ]

print("Computing Jacobian...")
H = system.get_jacobian(x0, obs_data)

# H structure: (400 meas, 7 params) 
# First 200 rows = Doppler, Next 200 rows = Radar

print(f"Jacobian shape: {H.shape}")
print(f"Jacobian sample (Doppler rows, Doppler params):\n{H[:5, :5]}")

# --- 7. Plotting ---
plt.figure(figsize=(12, 6))

# A. Plot Doppler Gradients for Pass 0 Bias
# Rows 0:100 are Pass 0 Doppler. Param Index 2 is dop_bias_0.
grad_dop_bias0 = H[0:100, 2]
plt.subplot(1, 2, 1)
plt.plot(t_all[:100], grad_dop_bias0.detach().numpy(), label='Gradient w.r.t Bias 0')
plt.title("Doppler Sensor Gradients (Bias)")
plt.ylabel("d(Doppler)/d(Bias) [Hz/Hz]")
plt.legend()
plt.grid(True)

# B. Plot Doppler Gradient for Frequency Offset
# Param Index 4 is Freq Offset
grad_dop_fc = H[:, 4]
plt.subplot(1, 2, 2)
plt.plot(t_all, grad_dop_fc.detach().numpy()[:200], color='orange', label='Gradient w.r.t Fc')
plt.title("Doppler Sensitivity to Freq Offset")
plt.ylabel("d(Doppler)/d(Fc) [Hz/Hz]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("jacobian_analysis.png")
plt.close()
# Verification
print(f"Mean Gradient for Doppler Bias 0 (should be 1.0): {grad_dop_bias0.mean().item():.4f}")
print(f"Mean Gradient for Doppler Freq Offset: {grad_dop_fc.mean().item():.4f}")