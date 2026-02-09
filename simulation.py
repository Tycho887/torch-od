import torch
import matplotlib.pyplot as plt
import numpy as np
from dsgp4.tle import TLE
from src.systemObject import BaseOrbitSystem
from src.sensorLayers import DopplerSensor, RadarSensor
from src.groundStation import GroundStation

# --- 2. Setup Scenario ---
TLE_list = ["ISS (ZARYA)",
"1 25544U 98067A   26038.50283897  .00012054  00000-0  23050-3 0  9996",
"2 25544  51.6315 221.5822 0011000  74.6214 285.5989 15.48462076551652"]
init_tle = TLE(data=TLE_list)

# Time: 2000 minutes
t_all = torch.linspace(start=1762418742, end=1762418742+86400, steps=2000)

# --- 3. Define Ground Station ---
gs_main = GroundStation(lat_deg=9.0, lon_deg=0.0, alt_km=0.0, epoch_tai = float(t_all[0]),  station_id='GS_1')
ground_stations = {'GS_1': gs_main}

# --- 4. Define Sensors ---
doppler_sensor = DopplerSensor(center_freq=435e6, num_passes=2, fit_center_freq=True, station_id='GS_1')
radar_sensor = RadarSensor(num_passes=2, station_id='GS_1')

sensors = {'doppler': doppler_sensor, 'radar': radar_sensor}

# --- 5. Define Observation Packets ---
indices = torch.zeros(2000, dtype=torch.long)
indices[1000:] = 1

obs_data = {
    'doppler': {'t': t_all, 'indices': indices},
    'radar':   {'t': t_all, 'indices': indices}
}

# --- 6. Build System ---
system = BaseOrbitSystem(init_tle=init_tle, fit_keys=['n', 'i'], sensors_dict=sensors, ground_stations=ground_stations)

# --- 7. Jacobian Analysis ---
x0 = system.state_def.get_initial_state()
# x0 layout: 
# [ n, i, 
#   dop_bias_0, dop_bias_1, dop_freq_bias, 
#   rad_bias_0, rad_bias_1 ]

print("Computing Jacobian...")
H = system.get_jacobian(state_vector=x0, observations=obs_data)

# H structure: (400 meas, 7 params) 
# First 200 rows = Doppler, Next 200 rows = Radar

print(f"Jacobian shape: {H.shape}")
print(f"Jacobian sample (Doppler rows, Doppler params):\n{H[1000:, 2:]}")

# --- 7. Plotting ---
plt.figure(figsize=(12, 6))

# A. Plot Doppler Gradients for Pass 0 Bias
# Rows 0:100 are Pass 0 Doppler. Param Index 2 is dop_bias_0.
grad_dop_bias0 = H[0:1000, 0]
plt.subplot(1, 2, 1)
plt.plot(t_all[:1000], grad_dop_bias0.detach().numpy(), label='Gradient w.r.t Bias 0')
plt.title(label="Mean motion sensitivity")
plt.ylabel(ylabel="d(Doppler)/d(Mean motion) [rad/s/Hz]")
plt.legend()
plt.grid(visible=True)

# B. Plot Doppler Gradient for Frequency Offset
# Param Index 4 is Freq Offset
grad_dop_fc = H[:, 4]
plt.subplot(1, 2, 2)
plt.plot(t_all, grad_dop_fc.detach().numpy()[2000:], color='orange', label='Gradient w.r.t Fc')
plt.title(label="Doppler Sensitivity to Freq Offset")
plt.ylabel(ylabel="d(Doppler)/d(Fc) [Hz/Hz]")
plt.legend()
plt.grid(visible=True)

plt.tight_layout()
plt.savefig("jacobian_analysis.png")
plt.close()
# Verification
print(f"Mean Gradient for Doppler Bias 0 (should be 1.0): {grad_dop_bias0.mean().item():.4f}")
print(f"Mean Gradient for Doppler Freq Offset: {grad_dop_fc.mean().item():.4f}")

# --- 8. Plotting Expected Values ---
print("Generating expected values...")
with torch.no_grad():
    preds = system(x0, obs_data)

# Split predictions based on observation counts (sorted keys: doppler, radar)
n_dop = len(obs_data['doppler']['t'])
dop_preds = preds[:n_dop].numpy()
rad_preds = preds[n_dop:].numpy()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(t_all.numpy(), dop_preds, label='Expected Doppler')
plt.title(label="Expected Doppler Signal")
plt.xlabel(xlabel="Time (min)")
plt.ylabel(ylabel="Doppler Shift (Hz)")
plt.grid(visible=True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_all.numpy(), rad_preds, color='orange', label='Expected Range')
plt.title(label="Expected Radar Range")
plt.xlabel(xlabel="Time (min)")
plt.ylabel(ylabel="Range (km)")
plt.grid(visible=True)
plt.legend()

plt.tight_layout()
plt.savefig("expected_values.png")
plt.close()
print("Saved expected_values.png")