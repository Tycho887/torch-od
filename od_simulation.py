import torch
import polars as pl
import dsgp4
from dsgp4.tle import TLE
from astropy.time import Time

# Internal diffod modules
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import unix_to_mjd, load_gmat_csv_block_legacy
from diffod.visualize import compute_ric_residuals, plot_ric_residuals, plot_calibrated_doppler, print_ric_residual_summary
from diffod.solvers.gn_svd import svd_solve
from diffod.solvers.gaussNewton import wgn_solve

# ---------------------------------------------------------
# 1. Configuration & Data Loading
# ---------------------------------------------------------
device = torch.device(device="cpu")
dtype = torch.float64
center_freq = 1707.0  # MHz, matching the synthetic dataset base_freq

# Initial TLE Guess (Matches the generation script)
TLE_list = [
    "AWS",
    "1 60543U 24149CD  25307.42878472  .00000000  00000-0  11979-3 0    11",
    "2 60543  97.7067  19.0341 0003458 123.1215 316.4897 14.89807169 65809",
]
tle_base = TLE(data=TLE_list)
epoch_unix = 1762207191

print("Loading Synthetic Dataset & GPS Truth...")

# Load GPS Truth (From legacy CSV format used in generation)
t_gps, r_gps, v_gps = load_gmat_csv_block_legacy(
    file_path="data/AWS_full_long_period.csv", 
    tle_epoch_unix=epoch_unix,
    block_sec=86400 * 2
)

# Load Synthetic Doppler Telemetry (From Parquet)
synthetic_telemetry = pl.read_parquet("data/synthetic_period_telemetry.parquet")

t_dopp = torch.tensor(synthetic_telemetry["timestamp"].to_numpy(), dtype=dtype, device=device)
d_dopp = torch.tensor(synthetic_telemetry["Doppler_Hz"].to_numpy(), dtype=dtype, device=device)
c_dopp = torch.tensor(synthetic_telemetry["contact_index"].to_numpy(), dtype=torch.int32, device=device)

print(f"Mean doppler time: {torch.mean(t_dopp)}")

# Filter to GPS window
valid_mask = (t_dopp >= t_gps.min()) & (t_dopp <= t_gps.max())
t_dopp = t_dopp[valid_mask]
d_dopp = d_dopp[valid_mask]
c_dopp = c_dopp[valid_mask]

print(f"Filtered to GPS window: {len(t_dopp)} valid samples remain.")
print(f"Unique passes: {len(torch.unique(c_dopp))}")

# Define Central Epoch for the Batch
T_mean = float(torch.mean(t_gps))
print(f"Central Epoch (T_mean): {T_mean}")

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
t_since_gps = (t_gps - T_mean) 

x_gps_out, _ = svd_solve(
    x_init=ssv_gps.get_initial_state(),
    y_obs_fixed=y_gps_1d,
    forward_fn=lambda x: pipe_gps(x=x, tsince=t_since_gps),
    estimate_mask=ssv_gps.get_active_map(),
    num_steps=5,
    sigma_obs=1.0 # Adjusted to 1.0 based on calibration script defaults
)
 
tle_gps_fit = ssv_gps.export(x_gps_out)

# ---------------------------------------------------------
# PHASE 2: Doppler-Only Orbit Determination
# ---------------------------------------------------------
print("\n--- Phase 2: Doppler-Only OD ---")

# We use the GPS-fit TLE as the starting point, but now we set the fit flags 
# to True so the solver actually optimizes the orbit based on Doppler.
ssv_dopp = state.MEE_SSV(
    init_tle=tle_gps_fit, 
    num_measurements=len(t_dopp),
    fit_mean_motion=True, fit_f=True, fit_g=True,
    fit_h=True, fit_k=True, fit_L=True, fit_bstar=False
)

# Note: If you want to perform Joint OD + Calibration on the synthetic dataset,
# you would uncomment the bias groups here and in the DopplerMeasurement below.
# ssv_dopp.add_linear_bias(name="pass_freq_bias", group_indices=c_dopp)
# ssv_dopp.add_linear_bias(name="pass_time_bias", group_indices=c_dopp)

station_model = system.DifferentiableStation(
    lat_deg=78.228874, 
    lon_deg=15.376932, 
    alt_m=463.0, 
    ref_unix=T_mean, 
    ref_gmst_rad=t_ref_astropy.sidereal_time('mean', 'greenwich').radian,
    device=device
)

prop_dopp = system.SGP4(ssv=ssv_dopp)
meas_dopp = system.DopplerMeasurement(
    ssv=ssv_dopp, 
    station_model=station_model,
    # freq_bias_group=ssv_dopp.get_bias_group("pass_freq_bias"),
    # time_bias_group=ssv_dopp.get_bias_group("pass_time_bias")
)

pipe_dopp = system.MeasurementPipeline(propagator=prop_dopp, measurement_model=meas_dopp)

t_since_dopp = (t_dopp - T_mean)

def functional_forward_calib(x) -> torch.Tensor:
    return pipe_dopp(
        x=x, 
        tsince=t_since_dopp, 
        epoch=T_mean, 
        center_freq=center_freq
    )

x_dopp_out, _ = wgn_solve(
    x_init=ssv_dopp.get_initial_state(),
    y_obs_fixed=d_dopp,
    forward_fn=functional_forward_calib,
    sigma_obs=50.0, 
    estimate_mask=ssv_dopp.get_active_map(),
    num_steps=5
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

for name, state_vec in [("GPS-Fit", x_gps_out), ("Doppler-Fit", x_dopp_out)]:
    _, pos_err, vel_err = compute_ric_residuals(
        x_state=state_vec, 
        propagator=prop_gps,  
        t_gps=t_gps, r_gps=r_gps, v_gps=v_gps, 
        tle_epoch_unix=T_mean
    )
    results_ric[name] = (pos_err, vel_err)

# Print the RMS summary to the console
print_ric_residual_summary(results_ric)

# Plot RIC Comparison
t_plot = (t_gps - T_mean) 
plot_ric_residuals(t_plot, results_ric)

# Plot Doppler Curves (Using the Doppler-fit state)
with torch.no_grad():
    doppler_pred = functional_forward_calib(x_dopp_out)

plot_calibrated_doppler(
    t_obs=t_dopp, 
    doppler_obs=d_dopp, 
    doppler_pred=doppler_pred, 
    contacts=c_dopp
)