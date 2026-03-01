# import time
# from scipy.stats import fit
import torch
# import numpy as np
import dsgp4
from dsgp4.tle import TLE
from astropy.time import Time

# Internal diffod modules
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import unix_to_mjd, load_gmat_csv_block
from diffod.visualize import compute_ric_residuals, plot_ric_residuals, plot_calibrated_doppler, print_ric_residual_summary
# from diffod.visualize import compute_ric_residuals, plot_ric_residuals, plot_calibrated_doppler, print_ric_residual_summary
from diffod.solvers.gn_svd import svd_solve
from diffod.solvers.gaussNewton import wgn_solve

# Use the loading function from your utils
from diffod.utils import load_gmd_to_tensors

# ---------------------------------------------------------
# 1. Configuration & Data Loading
# ---------------------------------------------------------
device = torch.device(device="cpu")
dtype = torch.float64
center_freq = 2200.0  # Matches your simulated 2200 MHz center frequency [cite: 1, 5]

# Initial TLE Guess (Standard AWS/InnoSat TLE)
TLE_list = [
    "AWS",
    "1 60543U 24149CD  25307.42878472  .00000000  00000-0  11979-3 0    11",
    "2 60543  97.7067  19.0341 0003458 123.1215 316.4897 14.89807169 65809",
]
tle_base = TLE(data=TLE_list)

print("Loading Simulated Data...")
# Load GPS Truth (From your AWS_simulated.csv)
t_gps, r_gps, v_gps = load_gmat_csv_block(
    file_path="data/AWS_ideal_simulated.csv", 
    tle_epoch_unix=1735862400.0, # Jan 01 2025 12:00:00 [cite: 1]
    block_sec=86400*0.5
)



# Load Doppler Telemetry (From your GMAT .gmd file)
t_dopp, d_dopp, c_dopp = load_gmd_to_tensors(
    file_path="data/AWS_Svalbard_ideal_sim.gmd", 
    center_freq_hz=center_freq, 
    device="cpu"
)

print(f"Mean doppler time: {torch.mean(t_dopp)}")

valid_mask = (t_dopp >= t_gps.min()) & (t_dopp <= t_gps.max())
t_dopp = t_dopp[valid_mask]
d_dopp = d_dopp[valid_mask]
c_dopp = c_dopp[valid_mask]

print(f"Filtered to GPS window: {len(t_dopp)} valid samples remain.")
print(f"Unique passes: {len(torch.unique(c_dopp))}")

# Define Central Epoch for the Batch
T_mean = float(torch.mean(t_gps))
t_ref_astropy = Time(T_mean, format="unix", scale="utc")

# ---------------------------------------------------------
# PHASE 1: GPS-Based Orbit Determination (The "Truth" Benchmark)
# ---------------------------------------------------------
print("\n--- Phase 1: Fitting TLE to GPS ---")
init_tle_gps, _ = dsgp4.newton_method(tle_base, unix_to_mjd(T_mean))

ssv_gps = state.MEE_SSV(init_tle=init_tle_gps, num_measurements=len(t_gps), fit_bstar=False)
prop_gps = system.SGP4(ssv=ssv_gps)
meas_gps = system.CartesianMeasurement(ssv=ssv_gps)
pipe_gps = system.MeasurementPipeline(propagator=prop_gps, measurement_model=meas_gps)

y_gps_1d = meas_gps.format_gps_observations(r_gps, v_gps)
t_since_gps = (t_gps - T_mean) #/ 60.0

x_gps_out, _ = svd_solve(
    x_init=ssv_gps.get_initial_state(),
    y_obs_fixed=y_gps_1d,
    forward_fn=lambda x: pipe_gps(x=x, tsince=t_since_gps),
    estimate_mask=ssv_gps.get_active_map(),
    num_steps=5,
    sigma_obs=10
)
 
tle_gps_fit = ssv_gps.export(x_gps_out)

# Create perturbed TLE:

x_perturbed = x_gps_out.clone() #+ torch.randn_like(x_gps_out) * x_gps_out * 1e-5
tle_perturbed = ssv_gps.export(x_perturbed)
# ---------------------------------------------------------
# PHASE 2: Doppler-Only Orbit Determination
# ---------------------------------------------------------
print("\n--- Phase 2: Doppler-Only OD ---")
# Use a perturbed version of the GPS-fit TLE to simulate an OD start
ssv_dopp = state.MEE_SSV(init_tle=tle_perturbed, num_measurements=len(t_dopp),
                         fit_mean_motion=False, fit_f=False, fit_g=False,
                         fit_h=False, fit_k=False, fit_L=False, fit_bstar=False)

# ssv_dopp.add_linear_bias(name="pass_freq_bias", group_indices=c_dopp)

station = system.DifferentiableStation(
    lat_deg=78.23, lon_deg=15.39, alt_m=450, # Svalbard [cite: 1]
    ref_unix=T_mean, 
    ref_gmst_rad=t_ref_astropy.sidereal_time('mean', 'greenwich').radian
)

prop_dopp = system.SGP4(ssv=ssv_dopp)
meas_dopp = system.DopplerMeasurement(ssv=ssv_dopp, station_model=station)
                                        #   freq_bias_group=ssv_dopp.get_bias_group("pass_freq_bias"))
pipe_dopp = system.MeasurementPipeline(propagator=prop_dopp, measurement_model=meas_dopp)

t_since_dopp = (t_dopp - T_mean) #/ 60.0

x_dopp_out, _ = wgn_solve(
    x_init=ssv_dopp.get_initial_state(),
    y_obs_fixed=d_dopp,
    forward_fn=lambda x: pipe_dopp(x=x, tsince=t_since_dopp, epoch=T_mean, center_freq=center_freq),
    sigma_obs=100.0, # Assume 1Hz noise
    estimate_mask=ssv_dopp.get_active_map(),
    num_steps=8
)
# ---------------------------------------------------------
# 3. Comparison & Visualization
# ---------------------------------------------------------
print("\n--- Final Results Comparison ---")

# Export and print both TLEs
print(f"GPS-Fit TLE:\n{tle_gps_fit}\n")
tle_dopp_fit = ssv_dopp.export(x_dopp_out)
print(f"Doppler-Fit TLE:\n{tle_dopp_fit}\n")


# Compute RIC Residuals against GPS Truth for BOTH fits
results_ric = {}

# We iterate over both optimized state vectors
for name, state_vec in [("GPS-Fit", x_gps_out), ("Doppler-Fit", x_dopp_out)]:
    _, pos_err, vel_err = compute_ric_residuals(
        x_state=state_vec, 
        propagator=prop_gps, # SGP4 propagator 
        t_gps=t_gps, r_gps=r_gps, v_gps=v_gps, 
        tle_epoch_unix=T_mean
    )
    results_ric[name] = (pos_err, vel_err)

# Print the RMS summary to the console
print_ric_residual_summary(results_ric)

# Plot RIC Comparison
t_plot = (t_gps - T_mean) #/ 60.0
plot_ric_residuals(t_plot, results_ric)

# Plot Doppler Curves
with torch.no_grad():
    doppler_pred = pipe_dopp(x=x_gps_out, tsince=t_since_dopp, epoch=T_mean, center_freq=center_freq)

plot_calibrated_doppler(
    t_obs=t_dopp, 
    doppler_obs=d_dopp, 
    doppler_pred=doppler_pred, 
    contacts=c_dopp
)