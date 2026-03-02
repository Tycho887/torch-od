import time
import torch
import dsgp4
from dsgp4.tle import TLE
from astropy.time import Time

import diffod.state as state
import diffod.functional.system as system
from diffod.utils import load_gmat_csv_block, unix_to_mjd, load_gmd_to_tensors
from diffod.visualize import plot_calibrated_doppler, plot_pass_biases
from diffod.solvers.gn_svd import svd_solve


# ---------------------------------------------------------
# Custom Wrapper to bridge SGP4 and Time Biases
# ---------------------------------------------------------
class CalibratedSGP4(torch.nn.Module):
    """
    Wraps the standard SGP4 model to handle time in seconds and natively
    inject the per-pass time biases into the satellite's ephemeris evaluation.
    """
    def __init__(self, ssv, time_bias_group=None) -> None:
        super().__init__()
        self.sgp4 = system.SGP4(ssv, use_pretrained_model=False)
        self.time_bias_group = time_bias_group
        self.ssv = ssv

    def forward(self, x, tsince_sec) -> tuple[torch.Tensor, torch.Tensor]:
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

        # Apply bias and convert to minutes for SGP4
        t_eval_sec = tsince_sec + total_time_offset #/ 60
        t_eval_min = t_eval_sec
        
        return self.sgp4(x, t_eval_min)


# ---------------------------------------------------------
# 1. Setup Data & Initial Parameters
# ---------------------------------------------------------
print("Loading Simulated Data...")
target_device = torch.device("cpu")
dtype = torch.float64
base_freq = 2200.0  # MHz base frequency from GMAT script 

# Initial TLE
TLE_list = [
    "AWS",
    "1 60543U 24149CD  25309.00045718  .00000000  00000-0  30929-3 0    12",
    "2 60543  97.8060   3.3204 0011868 284.2452 333.3414 14.91418332 65809",
]
# tle_base = TLE(data=TLE_list)
epoch_unix = 1735862400.0 # 01 Jan 2025 12:00:00.000 [cite: 2]
tle0_base = TLE(data=TLE_list)

# Load Ground Truth GPS from Simulator
t_gps_raw, r_gps_raw, v_gps_raw = load_gmat_csv_block(
    file_path="data/AWS_ideal_simulated.csv", # [cite: 28]
    tle_epoch_unix=epoch_unix,
    block_sec=86400 * 1,
)

T_mean = float(torch.mean(t_gps_raw))
print(f"Central Epoch (T_mean): {T_mean}")

init_tle, _ = dsgp4.newton_method(tle0_base, unix_to_mjd(unix_seconds=T_mean))    

t_gps = t_gps_raw.to(device=target_device, dtype=dtype)
r_gps = r_gps_raw.to(device=target_device, dtype=dtype)
v_gps = v_gps_raw.to(device=target_device, dtype=dtype)

# Load Simulated Doppler Telemetry
times_unix, doppler_obs, contacts = load_gmd_to_tensors(
    file_path="data/AWS_Svalbard_ideal_sim.gmd", # [cite: 20]
    center_freq_hz=base_freq, 
    device="cpu"
)

# Filter Doppler data to match GPS window
valid_mask = (times_unix >= t_gps.min()) & (times_unix <= t_gps.max())
times_unix = times_unix[valid_mask]
doppler_obs = doppler_obs[valid_mask]
contacts = contacts[valid_mask]

# Remap contact indices to be strictly contiguous starting from 0
_, remapped_contacts = torch.unique(contacts, return_inverse=True)
contacts = remapped_contacts.to(torch.int32)
N_doppler_samples = len(times_unix)


# ---------------------------------------------------------
# PHASE 1: Orbit Determination (Solving for TLE using GPS)
# ---------------------------------------------------------
print("\n--- PHASE 1: Orbit Determination (GPS) ---")
ssv_orbit = state.MEE_SSV(
    init_tle=init_tle,
    num_measurements=len(t_gps),
    fit_mean_motion=True, fit_f=True, fit_g=True,
    fit_h=True, fit_k=True, fit_L=True, fit_bstar=False,
)

propagator_orbit = system.SGP4(ssv=ssv_orbit, use_pretrained_model=False)
meas_model_orbit = system.CartesianMeasurement(ssv=ssv_orbit)
pipeline_orbit = system.MeasurementPipeline(propagator=propagator_orbit, measurement_model=meas_model_orbit)

gps_obs_1d = meas_model_orbit.format_gps_observations(r_gps=r_gps, v_gps=v_gps)
t_since_mins_gps = (t_gps - T_mean)

def functional_forward_orbit(x) -> torch.Tensor:
    return pipeline_orbit(x=x, tsince=t_since_mins_gps)

x_orbit_in = ssv_orbit.get_initial_state(device=target_device)

t0 = time.perf_counter()
x_orbit_out, P_orbit_out = svd_solve(
    x_init=x_orbit_in,
    y_obs_fixed=gps_obs_1d,
    forward_fn=functional_forward_orbit,
    sigma_obs=1.0, 
    estimate_mask=ssv_orbit.get_active_map(device=target_device),
    num_steps=5,
)
t1 = time.perf_counter()
print(f"OD finished in {(t1 - t0) * 1000:.2f} ms")

opt_tle = ssv_orbit.export(x_orbit_out)


# ---------------------------------------------------------
# PHASE 2: System Calibration (Estimating Biases using Doppler)
# ---------------------------------------------------------
print("\n--- PHASE 2: System Calibration (Doppler) ---")
t_obs_centered = (times_unix - T_mean)

# Re-initialize the MEE_SSV with the optimized TLE, freezing the orbit
ssv_calib = state.MEE_SSV(
    init_tle=opt_tle,
    num_measurements=N_doppler_samples,
    fit_mean_motion=False, fit_f=False, fit_g=False,
    fit_h=False, fit_k=False, fit_L=False, fit_bstar=False,
)

ssv_calib.add_linear_bias(name="pass_freq_bias", group_indices=contacts)
ssv_calib.add_linear_bias(name="pass_time_bias", group_indices=contacts)

t_ref = Time(T_mean, format="unix", scale="utc")

# Svalbard coordinates from simulator.script
station_model = system.DifferentiableStation(
    lat_deg=78.23, # [cite: 11]
    lon_deg=15.39, # [cite: 11]
    alt_m=450,   # 0.45 km [cite: 11, 12]
    ref_unix=T_mean, 
    ref_gmst_rad=t_ref.sidereal_time('mean', 'greenwich').radian,
    device=target_device
)

propagator_calib = CalibratedSGP4(
    ssv=ssv_calib, 
    time_bias_group=ssv_calib.get_bias_group("pass_time_bias")
)

doppler_model = system.DopplerMeasurement(
    ssv=ssv_calib, 
    station_model=station_model,
    freq_bias_group=ssv_calib.get_bias_group("pass_freq_bias"),
    time_bias_group=ssv_calib.get_bias_group("pass_time_bias")
)

pipeline_calib = system.MeasurementPipeline(propagator=propagator_calib, measurement_model=doppler_model)

def functional_forward_calib(x) -> torch.Tensor:
    return pipeline_calib(
        x=x, 
        tsince=t_obs_centered, 
        epoch=T_mean, 
        center_freq=base_freq
    )

x_calib_in = ssv_calib.get_initial_state(device=target_device)
estimate_map_calib = ssv_calib.get_active_map(device=target_device)

print(f"Executing Calibration Solver... (Total Params: {len(x_calib_in)})")
t0 = time.perf_counter()
x_calib_out, P_calib_out = svd_solve(
    x_init=x_calib_in,
    y_obs_fixed=doppler_obs,
    forward_fn=functional_forward_calib,
    sigma_obs=10.0, # Adjusted noise sigma
    estimate_mask=estimate_map_calib,
    num_steps=5, 
)
t1 = time.perf_counter()
print(f"Calibration finished in {(t1 - t0) * 1000:.2f} ms")


# ---------------------------------------------------------
# 3. Extract Results & Visualize
# ---------------------------------------------------------
print("\n--- Calibration Results ---")
bg_freq = ssv_calib.get_bias_group("pass_freq_bias")
bg_time = ssv_calib.get_bias_group("pass_time_bias")

freq_biases_hz = x_calib_out[bg_freq.global_offset : bg_freq.global_offset + bg_freq.num_params] * 1000.0
time_biases_sec = x_calib_out[bg_time.global_offset : bg_time.global_offset + bg_time.num_params]

for i in range(len(freq_biases_hz)):
    print(f"Pass {i:02d} | Freq Offset: {freq_biases_hz[i]*1e-3:8.2f} kHz | Time Offset: {time_biases_sec[i]:6.3f} sec")

with torch.no_grad():
    doppler_pred = functional_forward_calib(x_calib_out)

plot_calibrated_doppler(
    t_obs=times_unix, 
    doppler_obs=doppler_obs, 
    doppler_pred=doppler_pred, 
    contacts=contacts
)

plot_pass_biases(freq_biases_hz=freq_biases_hz, time_biases_sec=time_biases_sec)