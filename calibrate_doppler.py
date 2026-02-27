import time
import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# diffod imports
from diffod.functional.system import DopplerMeasurement, GPSInterpolator, MeasurementPipeline
from diffod.gse import station_teme_preprocessor
# from diffod.solvers.gn_svd import svd_solve as wgn_solve
from diffod.solvers.gaussNewton import wgn_solve
from diffod.utils import load_gmat_csv_block
from diffod.state import CalibrationSSV
from diffod.visualize import plot_calibrated_doppler

# ---------------------------------------------------------
# 2. Setup Data & Boundary
# ---------------------------------------------------------
print("Loading Data...")
epoch = 1762165047
base_freq = 1.8e9
target_device = torch.device("cpu")
stations = {0: np.array([15.376932, 78.228874, 463])}


# Load Doppler Telemetry
period_telemetry = pl.read_parquet("data/period_telemetry.parquet")
times_unix = torch.tensor(period_telemetry["timestamp"].to_numpy(), dtype=torch.float64, device=target_device)
doppler_obs = torch.tensor(period_telemetry["Doppler_Hz"].to_numpy(), dtype=torch.float64, device=target_device)
contacts = torch.tensor(period_telemetry["contact_index"].to_numpy(), dtype=torch.int32, device=target_device)

N_samples = len(times_unix)

print("Preprocessing Ground Station Ephemeris...")
st_indices = torch.zeros(N_samples, dtype=torch.int32, device=target_device)



# FIX: Keep in seconds, but make relative to epoch
t_obs_sec = times_unix - epoch 
N_samples = len(times_unix)

# Load Ground Truth GPS
t_gps_raw, r_gps_raw, v_gps_raw = load_gmat_csv_block(
    file_path="data/AWS_high_frequency.csv",
    tle_epoch_unix=float(torch.mean(times_unix)),
    block_sec=86400*3,
)

# ---------------------------------------------------------
# NEW: Filter observations to viable GPS duration window
# ---------------------------------------------------------
gps_start = t_gps_raw.min()
gps_end = t_gps_raw.max()

valid_mask = (times_unix >= gps_start) & (times_unix <= gps_end)

# Filter all observation-aligned tensors
times_unix = times_unix[valid_mask]
doppler_obs = doppler_obs[valid_mask]
contacts = contacts[valid_mask]
t_obs_sec = t_obs_sec[valid_mask]
st_indices = st_indices[valid_mask] 

# FIX 2: Remap contact indices to be strictly contiguous starting from 0
# This prevents the state vector from allocating "ghost" parameters for filtered data
_, remapped_contacts = torch.unique(contacts, return_inverse=True)
contacts = remapped_contacts.to(torch.int32)

# 1. Calculate the new internal mathematical epoch
T_mean = t_gps_raw.mean()

# 2. Create centered time arrays for the solver's forward pass
t_gps_centered = (t_gps_raw - T_mean).to(target_device)
t_obs_centered = (times_unix - T_mean).to(target_device)

# 3. Precompute Astropy positions using the RAW, uncentered Unix times
# Astropy strictly requires the exact Unix time to calculate Earth's rotation angle
print("Preprocessing Ground Station Ephemeris...")
station_pos_cpu, station_vel_cpu = station_teme_preprocessor(
    times_s=times_unix.numpy(), 
    station_ids=st_indices.numpy(),
    id_to_station=stations,
    dtype=torch.float64,
    device=target_device,
)
st_pos = station_pos_cpu.to(target_device)
st_vel = station_vel_cpu.to(target_device)

r_gps = r_gps_raw.to(target_device)
v_gps = v_gps_raw.to(target_device)
# FIX: Removed the second [valid_mask] slice here. The arrays are already sized correctly.

N_samples = len(times_unix)
print(f"Filtered to GPS window: {N_samples} valid samples remain.")
# ---------------------------------------------------------

# t_gps_sec = t_gps_raw.to(target_device) 
# r_gps = r_gps_raw.to(target_device) #/ 1e3
# v_gps = v_gps_raw.to(target_device) #/ 1e3

# ---------------------------------------------------------
# 3. Define State Vector & Functional Forward
# ---------------------------------------------------------
ssv = CalibrationSSV(
    num_measurements=N_samples, 
    fit_time_offset=True, 
    fit_frequency_offset=True,
    # fit_freq_drift=False
)

# Apply the contacts array to map independent biases to each pass
ssv.add_linear_bias(name="pass_bias", group_indices=contacts)

# The empirical propagator interpolating over the GPS track
interpolator = GPSInterpolator(
    ssv=ssv, 
    t_gps_ref=t_gps_centered, # Now using centered times
    r_gps_ref=r_gps, 
    v_gps_ref=v_gps
)

# The modified measurement block that natively applies the drift and pass biases
doppler_model = DopplerMeasurement(
    ssv=ssv, 
    # base_freq_hz=base_freq,
    bias_group=ssv.get_bias_group("pass_bias")
)

# The standard agnostic pipeline
model = MeasurementPipeline(
    propagator=interpolator, 
    measurement_model=doppler_model
)

def functional_forward(x) -> torch.Tensor:
    return model(
        x=x, 
        st_pos=st_pos, 
        st_vel=st_vel, 
        tsince=t_obs_centered, # Pass the centered observation times
        center_freq=base_freq    # Note: I corrected a small typo here; it was 1.707e6 in your snippet
    )
# ---------------------------------------------------------
# 4. Execute Solver
# ---------------------------------------------------------
x_in = ssv.get_initial_state(device=target_device)
estimate_map = ssv.get_active_map(device=target_device)

print(f"\nExecuting Calibration Solver... (Total Params: {len(x_in)})")
t0 = time.perf_counter()

x_out, P_out = wgn_solve(
    x_init=x_in,
    y_obs_fixed=doppler_obs,
    forward_fn=functional_forward,
    sigma_obs=10.0,
    estimate_mask=estimate_map,
    num_steps=10, 
)

t1 = time.perf_counter()
print(f"Calibration finished in {(t1 - t0) * 1000:.2f} ms")

# ---------------------------------------------------------
# 5. Extract Results & Plot Random Walk
# ---------------------------------------------------------
calibrated_params = ssv.export(x_out)
print("\n--- Calibration Results ---")
print(f"Time Offset (seconds): {calibrated_params['time_offset']:.6e}")
print(f"Freq (Hz): {base_freq + 1e6*calibrated_params['freq_offset']:.6e}")

biases = calibrated_params.get("pass_biases", [])
print(f"Estimated biases: {biases}")
pass_indices = list(range(len(biases)))

# Plotting the random walk characteristics 

plt.figure(figsize=(10, 5))
plt.plot(pass_indices, biases, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
plt.title("Per-Pass Doppler Frequency Bias (Random Walk)")
plt.xlabel("Contact Pass Index")
plt.ylabel("Frequency Bias (Hz)")
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. Plot Doppler Curves & Residuals
# ---------------------------------------------------------
print("\nEvaluating Calibrated Model...")

# Run the optimized state through the forward pipeline to get predicted measurements
with torch.no_grad():
    doppler_pred = functional_forward(x_out)

print("Generating Doppler curve plots...")

# Call the visualization function
plot_calibrated_doppler(
    t_obs=times_unix, 
    doppler_obs=doppler_obs, 
    doppler_pred=doppler_pred, 
    contacts=contacts
)