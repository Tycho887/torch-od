import matplotlib.pyplot as plt
import numpy as np
import torch
from dsgp4.tle import TLE

# Make sure to import the modules correctly based on file structure
from src.groundStation import GroundStation
from src.sensorLayers import DopplerSensor, RadarSensor
from src.systemObject import BaseOrbitSystem

# --- 2. Setup Scenario ---
TLE_list = [
    "ISS (ZARYA)",
    "1 25544U 98067A   26038.50283897  .00012054  00000-0  23050-3 0  9996",
    "2 25544  51.6315 221.5822 0011000  74.6214 285.5989 15.48462076551652",
]
init_tle = TLE(data=TLE_list)

# Time: 2000 steps
N_STEPS = 2000
t_start = 1762418742
t_end = t_start + 86400  # 1 day
t_all = torch.linspace(start=t_start, end=t_end, steps=N_STEPS)

# --- 3. Define Ground Station ---
gs_main = GroundStation(
    lat_deg=9.0, lon_deg=0.0, alt_km=0.0, epoch_tai=float(t_all[0]), station_id="GS_1"
)
ground_stations = {"GS_1": gs_main}

# --- 4. Define Sensors ---
# Doppler: 2 passes (pass_index 0 and 1), fit center freq
doppler_sensor = DopplerSensor(
    center_freq=435e6, num_passes=2, fit_center_freq=True, station_id="GS_1"
)
# Radar: 2 passes
radar_sensor = RadarSensor(num_passes=2, station_id="GS_1")

sensors = {"doppler": doppler_sensor, "radar": radar_sensor}

# --- 5. Define Observation Packets ---
# FIX: Ensure pass_index match the time vector length (2000)
# Previously this was 3000, causing shape mismatches or broadcasting errors.
pass_index = torch.zeros(N_STEPS, dtype=torch.long)
# Simulate a pass change halfway through
pass_index[N_STEPS // 2 :] = 1

print(pass_index)

obs_data = {
    "doppler": {"t": t_all, "pass_index": pass_index},
    "radar": {"t": t_all, "pass_index": pass_index},
}

# --- 6. Build System ---
system = BaseOrbitSystem(
    init_tle=init_tle,
    fit_keys=["n", "i"],
    sensors_dict=sensors,
    ground_stations=ground_stations,
)

# --- 7. Jacobian Analysis ---
x0 = system.state_def.get_initial_state()
# Expected x0 layout (7 params):
# Orbital: [n, i]
# Doppler: [bias_0, bias_1, freq_bias]
# Radar:   [bias_0, bias_1]

print("Computing Jacobian...")
# BaseOrbitSystem handles batching.
# Output Shape should be: (Total_Measurements, Total_Params) -> (4000, 7)
H = system.get_jacobian(state_vector=x0, observations=obs_data)

print(f"Jacobian shape: {H.shape}")

# --- 8. Forward Pass for Expected Values ---
print("Computing Expected Values...")
with torch.no_grad():
    preds = system(x0, obs_data)

# Split predictions: Order is sorted keys -> doppler, radar
n_dop = len(obs_data["doppler"]["t"])
dop_preds = preds[:n_dop].numpy()
rad_preds = preds[n_dop:].numpy()

# --- 9. Plotting ---
t_plot = (t_all - t_all[0]).numpy() / 60.0  # Minutes
param_names = ["n", "i", "Dop_B0", "Dop_B1", "Dop_Fc", "Rad_B0", "Rad_B1"]

fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

# 1. Top Left: Doppler Measurements
axs[0, 0].plot(t_plot, dop_preds, label="Expected Doppler", color="b")
axs[0, 0].set_title("Doppler Signal")
axs[0, 0].set_ylabel("Frequency Shift (Hz)")
axs[0, 0].grid(True)
axs[0, 0].legend()

# 2. Top Right: Radar Measurements
axs[0, 1].plot(t_plot, rad_preds, label="Expected Range", color="orange")
axs[0, 1].set_title("Radar Range")
axs[0, 1].set_ylabel("Range (km)")
axs[0, 1].grid(True)
axs[0, 1].legend()

# 3. Bottom Left: Doppler Sensitivities (Log Scale)
# H rows 0:n_dop correspond to Doppler
H_dop = H[:n_dop, :].detach().numpy()
for i, name in enumerate(param_names):
    # Plot magnitude
    sensitivity = np.abs(H_dop[:, i])
    # Add small epsilon to avoid log(0)
    axs[1, 0].plot(t_plot, sensitivity + 1e-12, label=f"d(Dop)/d({name})")

axs[1, 0].set_yscale("log")
axs[1, 0].set_title("Doppler Sensitivities (Log Magnitude)")
axs[1, 0].set_ylabel("|Gradient|")
axs[1, 0].set_xlabel("Time (min)")
axs[1, 0].grid(True, which="both", linestyle="--", alpha=0.7)
# Place legend outside to avoid clutter
axs[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

# 4. Bottom Right: Radar Sensitivities (Log Scale)
# H rows n_dop:end correspond to Radar
H_rad = H[n_dop:, :].detach().numpy()
for i, name in enumerate(param_names):
    sensitivity = np.abs(H_rad[:, i])
    axs[1, 1].plot(t_plot, sensitivity + 1e-12, label=f"d(Rad)/d({name})")

axs[1, 1].set_yscale("log")
axs[1, 1].set_title("Radar Sensitivities (Log Magnitude)")
axs[1, 1].set_ylabel("|Gradient|")
axs[1, 1].set_xlabel("Time (min)")
axs[1, 1].grid(True, which="both", linestyle="--", alpha=0.7)
axs[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

plt.tight_layout()
plt.savefig("simulation_results.png")
print("Plot saved to simulation_results.png")
