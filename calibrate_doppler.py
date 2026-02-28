import time
import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# diffod imports
from diffod.functional.system import DopplerMeasurement, GPSInterpolator, MeasurementPipeline, DifferentiableStation
from diffod.solvers.newton import newton_solve
from diffod.utils import load_gmat_csv_block
from diffod.state import CalibrationSSV
from diffod.visualize import plot_calibrated_doppler
from astropy.time import Time

# ---------------------------------------------------------
# 1. Setup Data & Initial Parameters
# ---------------------------------------------------------
print("Loading Data...")
base_freq = 1.8e9
target_device = torch.device("cpu")
stations = {0: np.array([15.376932, 78.228874, 463])}

# Load Doppler Telemetry
period_telemetry = pl.read_parquet("data/period_telemetry.parquet")
times_unix = torch.tensor(period_telemetry["timestamp"].to_numpy(), dtype=torch.float64, device=target_device)
doppler_obs = torch.tensor(period_telemetry["Doppler_Hz"].to_numpy(), dtype=torch.float64, device=target_device)
contacts = torch.tensor(period_telemetry["contact_index"].to_numpy(), dtype=torch.int32, device=target_device)

# Load Ground Truth GPS
t_gps_raw, r_gps_raw, v_gps_raw = load_gmat_csv_block(
    file_path="data/AWS_high_frequency.csv",
    tle_epoch_unix=float(torch.mean(times_unix)),
    block_sec=86400 * 3,
)

# ---------------------------------------------------------
# 2. Filter & Center Data (Epoch Architecture)
# ---------------------------------------------------------
gps_start, gps_end = t_gps_raw.min(), t_gps_raw.max()
valid_mask = (times_unix >= gps_start) & (times_unix <= gps_end)

times_unix = times_unix[valid_mask]
doppler_obs = doppler_obs[valid_mask]
contacts = contacts[valid_mask]

# Remap contact indices to be strictly contiguous starting from 0
_, remapped_contacts = torch.unique(contacts, return_inverse=True)
contacts = remapped_contacts.to(torch.int32)
N_samples = len(times_unix)
print(f"Filtered to GPS window: {N_samples} valid samples remain.")

# Calculate the central epoch for numerical stability
T_mean = float(t_gps_raw.mean())

# Create centered time arrays (tsince) for the solver
t_gps_centered = (t_gps_raw - T_mean).to(target_device)
t_obs_centered = (times_unix - T_mean).to(target_device)

r_gps = r_gps_raw.to(target_device)
v_gps = v_gps_raw.to(target_device)

# ---------------------------------------------------------
# 3. Model Initialization
# ---------------------------------------------------------
print("Initializing Differentiable Station Model...")
t_ref = Time(T_mean, format="unix", scale="utc")
ref_gmst_rad = t_ref.sidereal_time('mean', 'greenwich').radian

station_model = DifferentiableStation(
    lat_deg=78.228874, 
    lon_deg=15.376932, 
    alt_m=463.0, 
    ref_unix=T_mean, # Station rotates relative to T_mean
    ref_gmst_rad=ref_gmst_rad,
    device=target_device
)

# Define State Vector
ssv = CalibrationSSV(
    num_measurements=N_samples, 
    fit_time_offset=True, 
    fit_frequency_offset=True,
)

# Add per-pass bias groups
ssv.add_linear_bias(name="pass_freq_bias", group_indices=contacts)
ssv.add_linear_bias(name="pass_time_bias", group_indices=contacts)

# Initialize Empirical Propagator (now aware of time biases)
interpolator = GPSInterpolator(
    ssv=ssv, 
    t_gps_ref=t_gps_centered, 
    r_gps_ref=r_gps, 
    v_gps_ref=v_gps,
    time_bias_group=ssv.get_bias_group("pass_time_bias")
)

# Initialize Measurement Model
doppler_model = DopplerMeasurement(
    ssv=ssv, 
    station_model=station_model,
    freq_bias_group=ssv.get_bias_group("pass_freq_bias"),
    time_bias_group=ssv.get_bias_group("pass_time_bias")
)

model = MeasurementPipeline(propagator=interpolator, measurement_model=doppler_model)

def functional_forward(x) -> torch.Tensor:
    return model(
        x=x, 
        tsince=t_obs_centered, 
        epoch=T_mean, # Pass the unified epoch down the pipeline
        center_freq=base_freq
    )

# ---------------------------------------------------------
# 4. Execute Solver
# ---------------------------------------------------------
x_in = ssv.get_initial_state(device=target_device)
estimate_map = ssv.get_active_map(device=target_device)

print(f"\nExecuting Calibration Solver... (Total Params: {len(x_in)})")
t0 = time.perf_counter()

x_out, P_out = newton_solve(
    x_init=x_in,
    y_obs_fixed=doppler_obs,
    forward_fn=functional_forward,
    sigma_obs=10.0,
    estimate_mask=estimate_map,
    num_steps=5, 
)

t1 = time.perf_counter()
print(f"Calibration finished in {(t1 - t0) * 1000:.2f} ms")

# ---------------------------------------------------------
# 5. Extract Results & Plot
# ---------------------------------------------------------
calibrated_params = ssv.export(x_out)
print("\n--- Calibration Results ---")
print(f"Global Time Offset (seconds): {calibrated_params['time_offset']:.6f}")
print(f"Global Freq (Hz): {base_freq + calibrated_params['freq_offset']:.6f}") 

freq_biases = calibrated_params.get("pass_freq_bias", [])
time_biases = calibrated_params.get("pass_time_bias", [])

# Plotting the frequency random walk
if freq_biases:
    pass_indices = list(range(len(freq_biases)))
    plt.figure(figsize=(10, 5))
    plt.plot(pass_indices, freq_biases, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    plt.title("Per-Pass Doppler Frequency Bias (Random Walk)")
    plt.xlabel("Contact Pass Index")
    plt.ylabel("Frequency Bias (Hz)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

print("\nEvaluating Calibrated Model...")
with torch.no_grad():
    doppler_pred = functional_forward(x_out)

print("Generating Doppler curve plots...")
plot_calibrated_doppler(
    t_obs=times_unix, 
    doppler_obs=doppler_obs, 
    doppler_pred=doppler_pred, 
    contacts=contacts
)