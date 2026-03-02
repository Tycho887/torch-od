import time
import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import dsgp4
from dsgp4.tle import TLE
from astropy.time import Time

import diffod.state as state
import diffod.functional.system as system
from diffod.utils import load_gmat_csv_block_legacy, unix_to_mjd
from diffod.solvers.gn_svd import svd_solve

# ---------------------------------------------------------
# Custom Wrapper to bridge SGP4 and Time Biases
# ---------------------------------------------------------
class CalibratedSGP4(torch.nn.Module):
    def __init__(self, ssv, time_bias_group=None):
        super().__init__()
        self.sgp4 = system.SGP4(ssv, use_pretrained_model=False)
        self.time_bias_group = time_bias_group
        self.ssv = ssv

    def forward(self, x, tsince_sec):
        args = self.ssv.get_functional_args(x)
        total_time_offset = args.get("time_offset", 0.0)

        if self.time_bias_group is not None:
            mask = self.time_bias_group.indices >= 0
            valid_indices = self.time_bias_group.indices[mask]
            start = self.time_bias_group.global_offset
            end = start + self.time_bias_group.num_params
            
            pass_offsets = torch.zeros_like(tsince_sec)
            pass_offsets[mask] = x[start:end][valid_indices]
            total_time_offset = total_time_offset + pass_offsets

        t_eval_sec = tsince_sec + total_time_offset / 60
        t_eval_min = t_eval_sec
        
        return self.sgp4(x, t_eval_min)


# ---------------------------------------------------------
# 1. Setup Data & Initial Parameters
# ---------------------------------------------------------
print("Loading Data...")
target_device = torch.device("cpu")
dtype = torch.float64
base_freq = 1.707e3

TLE_list = [
    "AWS",
    "1 60543U 24149CD  25307.42878472  .00000000  00000-0  11979-3 0    11",
    "2 60543  97.7067  19.0341 0003458 123.1215 316.4897 14.89807169 65809",
]
epoch_unix = 1762207191
tle0_base = TLE(data=TLE_list)

# Load Ground Truth GPS
t_gps_raw, r_gps_raw, v_gps_raw = load_gmat_csv_block_legacy(
    file_path="data/AWS_full_long_period.csv",
    tle_epoch_unix=epoch_unix,
    block_sec=86400 * 2,
)

T_mean = float(torch.mean(t_gps_raw))
init_tle, _ = dsgp4.newton_method(tle0_base, unix_to_mjd(unix_seconds=T_mean))    

t_gps = t_gps_raw.to(device=target_device, dtype=dtype)
r_gps = r_gps_raw.to(device=target_device, dtype=dtype)
v_gps = v_gps_raw.to(device=target_device, dtype=dtype)

# Load Doppler Telemetry
period_telemetry = pl.read_parquet("data/period_telemetry.parquet")
bad_indices = period_telemetry.filter(pl.col("Doppler_Hz").abs() > 50000)["contact_index"].unique()
period_telemetry = period_telemetry.filter(~pl.col("contact_index").is_in(bad_indices))

times_unix = torch.tensor(period_telemetry["timestamp"].to_numpy(), dtype=dtype, device=target_device)
doppler_obs = torch.tensor(period_telemetry["Doppler_Hz"].to_numpy(), dtype=dtype, device=target_device)
contacts = torch.tensor(period_telemetry["contact_index"].to_numpy(), dtype=torch.int32, device=target_device)

# Filter Doppler data
valid_mask = (times_unix >= t_gps.min()) & (times_unix <= t_gps.max())
times_unix = times_unix[valid_mask]
doppler_obs = doppler_obs[valid_mask]
contacts = contacts[valid_mask]

_, remapped_contacts = torch.unique(contacts, return_inverse=True)
contacts = remapped_contacts.to(torch.int32)
N_doppler_samples = len(times_unix)


# ---------------------------------------------------------
# PHASE 1: Orbit Determination
# ---------------------------------------------------------
print("\n--- PHASE 1: Orbit Determination ---")
ssv_orbit = state.MEE_SSV(
    init_tle=init_tle,
    num_measurements=len(t_gps),
    fit_mean_motion=True, fit_f=True, fit_g=True,
    fit_h=True, fit_k=True, fit_L=True, fit_bstar=True,
)

propagator_orbit = system.SGP4(ssv=ssv_orbit, use_pretrained_model=False)
meas_model_orbit = system.CartesianMeasurement(ssv=ssv_orbit)
pipeline_orbit = system.MeasurementPipeline(propagator=propagator_orbit, measurement_model=meas_model_orbit)

gps_obs_1d = meas_model_orbit.format_gps_observations(r_gps=r_gps, v_gps=v_gps)
t_since_mins_gps = (t_gps - T_mean)

x_orbit_out, P_orbit_out = svd_solve(
    x_init=ssv_orbit.get_initial_state(device=target_device),
    y_obs_fixed=gps_obs_1d,
    forward_fn=lambda x: pipeline_orbit(x=x, tsince=t_since_mins_gps),
    sigma_obs=1.0, 
    estimate_mask=ssv_orbit.get_active_map(device=target_device),
    num_steps=5,
)
opt_tle = ssv_orbit.export(x_orbit_out)


# ---------------------------------------------------------
# PHASE 2: System Calibration 
# ---------------------------------------------------------
print("\n--- PHASE 2: System Calibration ---")
t_obs_centered = (times_unix - T_mean)

ssv_calib = state.MEE_SSV(
    init_tle=opt_tle,
    num_measurements=N_doppler_samples,
    fit_mean_motion=False, fit_f=False, fit_g=False,
    fit_h=False, fit_k=False, fit_L=False, fit_bstar=False,
)

ssv_calib.add_linear_bias(name="pass_freq_bias", group_indices=contacts)
ssv_calib.add_linear_bias(name="pass_time_bias", group_indices=contacts)

t_ref = Time(T_mean, format="unix", scale="utc")
station_model = system.DifferentiableStation(
    lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, 
    ref_unix=T_mean, ref_gmst_rad=t_ref.sidereal_time('mean', 'greenwich').radian,
    device=target_device
)

propagator_calib = CalibratedSGP4(ssv=ssv_calib, time_bias_group=ssv_calib.get_bias_group("pass_time_bias"))
doppler_model = system.DopplerMeasurement(
    ssv=ssv_calib, station_model=station_model,
    freq_bias_group=ssv_calib.get_bias_group("pass_freq_bias"),
    time_bias_group=ssv_calib.get_bias_group("pass_time_bias")
)

pipeline_calib = system.MeasurementPipeline(propagator=propagator_calib, measurement_model=doppler_model)

x_calib_out, _ = svd_solve(
    x_init=ssv_calib.get_initial_state(device=target_device),
    y_obs_fixed=doppler_obs,
    forward_fn=lambda x: pipeline_calib(x=x, tsince=t_obs_centered, epoch=T_mean, center_freq=base_freq),
    sigma_obs=50.0,
    estimate_mask=ssv_calib.get_active_map(device=target_device),
    num_steps=5, 
)

# ---------------------------------------------------------
# PHASE 3: Synthetic Dataset Generation & Verification
# ---------------------------------------------------------
print("\n--- PHASE 3: Generating Synthetic Doppler & Verification ---")

# 1. Extract solved state and create a new simulation state vector
x_sim = x_calib_out.clone()
bg_time = ssv_calib.get_bias_group("pass_time_bias")

# 2. Calculate the global mean time offset and overwrite the per-pass biases
time_biases_sec = x_sim[bg_time.global_offset : bg_time.global_offset + bg_time.num_params]
mean_time_bias = torch.mean(time_biases_sec)
x_sim[bg_time.global_offset : bg_time.global_offset + bg_time.num_params] = mean_time_bias

print(f"Applied Global Mean Time Bias: {mean_time_bias.item():.4f} seconds")

# 3. Setup the GPS Interpolator as the propagator
interpolator = system.GPSInterpolator(
    ssv=ssv_calib,
    t_gps_ref=(t_gps - T_mean), 
    r_gps_ref=r_gps,
    v_gps_ref=v_gps,
    time_bias_group=bg_time 
)

# 4. Construct the simulation pipeline using the interpolator
pipeline_sim = system.MeasurementPipeline(propagator=interpolator, measurement_model=doppler_model)

with torch.no_grad():
    # Synthetic Doppler (GPS Interpolated + Global Time Bias + Pass Freq Bias)
    synthetic_doppler = pipeline_sim(
        x=x_sim, 
        tsince=t_obs_centered, 
        epoch=T_mean, 
        center_freq=base_freq
    )
    
    # TLE-Based Doppler (Calibrated SGP4 + Original Calibrated Biases)
    tle_doppler = pipeline_calib(
        x=x_calib_out, 
        tsince=t_obs_centered, 
        epoch=T_mean, 
        center_freq=base_freq
    )

# --- Verification Plot ---
print("Plotting verification residuals...")
residuals_synthetic = doppler_obs - synthetic_doppler
residuals_tle = doppler_obs - tle_doppler

plt.figure(figsize=(12, 6))
plt.scatter(times_unix.cpu().numpy(), residuals_tle.cpu().numpy(), 
            s=4, alpha=0.6, label='TLE SGP4 Residuals (True - SGP4)', color='blue')
plt.scatter(times_unix.cpu().numpy(), residuals_synthetic.cpu().numpy(), 
            s=4, alpha=0.6, label='Synthetic GPS Residuals (True - GPS Synthetic)', color='orange')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Doppler Residuals Verification: TLE vs. Synthetic GPS Dataset")
plt.xlabel("Unix Time (s)")
plt.ylabel("Doppler Residual (Hz)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# PHASE 4: Export to Parquet
# ---------------------------------------------------------
print("\n--- PHASE 4: Exporting Dataset ---")

synthetic_df = pl.DataFrame({
    "timestamp": times_unix.cpu().numpy(),
    "Doppler_Hz": synthetic_doppler.cpu().numpy(),
    "contact_index": contacts.cpu().numpy()
})

output_path = "data/synthetic_period_telemetry.parquet"
synthetic_df.write_parquet(output_path)
print(f"Successfully saved synthetic dataset to: {output_path}")