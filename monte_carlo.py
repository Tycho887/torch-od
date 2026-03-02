import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import dsgp4
from dsgp4.tle import TLE
from astropy.time import Time

# Internal diffod modules
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import unix_to_mjd, load_gmat_csv_block_legacy
from diffod.visualize import compute_ric_residuals, print_ric_residual_summary
from diffod.solvers.gn_svd import svd_solve

# ---------------------------------------------------------
# 1. Configuration & Data Loading
# ---------------------------------------------------------
device = torch.device(device="cpu")
dtype = torch.float64
center_freq = 1707.0  

# Parameters for the Monte Carlo Simulation
KNOWN_GLOBAL_TIME_BIAS_SEC = 0.277  # Replace with your actual bias
MC_ITERATIONS = 20
STATE_NOISE_SCALE = 1e-4          # Relative scaling for initial state perturbation
DOPPLER_NOISE_STD_HZ = 20.0       # Absolute Hz white noise added to measurements

TLE_list = [
    "AWS",
    "1 60543U 24149CD  25307.42878472  .00000000  00000-0  11979-3 0    11",
    "2 60543  97.7067  19.0341 0003458 123.1215 316.4897 14.89807169 65809",
]
tle_base = TLE(data=TLE_list)
epoch_unix = 1762207191

print("Loading Synthetic Dataset & GPS Truth...")
t_gps, r_gps, v_gps = load_gmat_csv_block_legacy(
    file_path="data/AWS_full_long_period.csv", 
    tle_epoch_unix=epoch_unix,
    block_sec=86400 * 2
)

synthetic_telemetry = pl.read_parquet("data/synthetic_period_telemetry.parquet")
t_dopp = torch.tensor(synthetic_telemetry["timestamp"].to_numpy(), dtype=dtype, device=device)
d_dopp_true = torch.tensor(synthetic_telemetry["Doppler_Hz"].to_numpy(), dtype=dtype, device=device)
c_dopp = torch.tensor(synthetic_telemetry["contact_index"].to_numpy(), dtype=torch.int32, device=device)

valid_mask = (t_dopp >= t_gps.min()) & (t_dopp <= t_gps.max())
t_dopp = t_dopp[valid_mask]
d_dopp_true = d_dopp_true[valid_mask]
c_dopp = c_dopp[valid_mask]

T_mean = float(torch.mean(t_gps))
t_ref_astropy = Time(T_mean, format="unix", scale="utc")


# ---------------------------------------------------------
# PHASE 1: GPS-Based Baseline OD
# ---------------------------------------------------------
print("\n--- Phase 1: Fitting Baseline TLE to GPS ---")
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
    sigma_obs=1.0 
)
tle_gps_fit = ssv_gps.export(x_gps_out)


# ---------------------------------------------------------
# PHASE 2: Encapsulated Doppler OD Function
# ---------------------------------------------------------
def run_doppler_od(x_guess: torch.Tensor, doppler_obs: torch.Tensor) -> torch.Tensor:
    """
    Runs the Doppler-only Orbit Determination solving only for n, L, and frequency biases.
    """
    # 1. Setup State Vector
    ssv_dopp = state.MEE_SSV(
        init_tle=tle_gps_fit, 
        num_measurements=len(t_dopp),
        fit_mean_motion=True, 
        fit_f=False, fit_g=False,  
        fit_h=False, fit_k=False,  
        fit_L=True,                
        fit_bstar=False
    )
    ssv_dopp.add_linear_bias(name="pass_freq_bias", group_indices=c_dopp)

    # 2. Setup Measurement Pipeline
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, 
        ref_unix=T_mean, ref_gmst_rad=t_ref_astropy.sidereal_time('mean', 'greenwich').radian,
        device=device
    )

    prop_dopp = system.SGP4(ssv=ssv_dopp)
    meas_dopp = system.DopplerMeasurement(
        ssv=ssv_dopp, station_model=station_model,
        freq_bias_group=ssv_dopp.get_bias_group("pass_freq_bias"), time_bias_group=None
    )
    pipe_dopp = system.MeasurementPipeline(propagator=prop_dopp, measurement_model=meas_dopp)

    t_since_dopp_calibrated = (t_dopp - T_mean) + KNOWN_GLOBAL_TIME_BIAS_SEC

    def forward_fn(x):
        return pipe_dopp(x=x, tsince=t_since_dopp_calibrated, epoch=T_mean, center_freq=center_freq)

    # 3. Solve
    x_out, _ = svd_solve(
        x_init=x_guess,
        y_obs_fixed=doppler_obs,
        forward_fn=forward_fn,
        sigma_obs=50.0, 
        estimate_mask=ssv_dopp.get_active_map(),
        num_steps=5
    )
    
    # We return the SSV object as well to extract active maps/indices later if needed
    return x_out, ssv_dopp

# ---------------------------------------------------------
# PHASE 3: Monte Carlo Simulation
# ---------------------------------------------------------
print("\n--- Phase 3: Executing Monte Carlo Simulation ---")

# First, we need the "clean" Doppler fit to establish the baseline dimensionality and expected output
# We use the initial state from MEE_SSV initialized with the GPS fit as our standard starting point
dummy_ssv = state.MEE_SSV(init_tle=tle_gps_fit, num_measurements=len(t_dopp), fit_mean_motion=True, fit_f=False, fit_g=False, fit_h=False, fit_k=False, fit_L=True, fit_bstar=False)
dummy_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_dopp)
standard_x_guess = dummy_ssv.get_initial_state()

print("Calculating clean baseline...")
x_clean_out, _ = run_doppler_od(x_guess=standard_x_guess, doppler_obs=d_dopp_true)
n_idx = dummy_ssv.map_param_to_idx["n"]
L_idx = dummy_ssv.map_param_to_idx["L"]

n_clean = x_clean_out[n_idx].item()
L_clean = x_clean_out[L_idx].item()

mc_results_n = []
mc_results_L = []
mc_states = []

for i in range(MC_ITERATIONS):
    # 1. Perturb the initial state guess 
    noise_vector = torch.randn_like(standard_x_guess) * STATE_NOISE_SCALE
    x_noisy_guess = standard_x_guess + (standard_x_guess * noise_vector)
    
    # 2. Add White Noise to the "True" Synthetic Doppler Data
    d_dopp_noisy = d_dopp_true + torch.randn_like(d_dopp_true) * DOPPLER_NOISE_STD_HZ
    
    # 3. Solve
    x_mc_out, current_ssv = run_doppler_od(x_guess=x_noisy_guess, doppler_obs=d_dopp_noisy)
    
    # 4. Store the FULL state vector for spatial evaluation
    mc_states.append(x_mc_out)
    
    if (i + 1) % 5 == 0:
        print(f"Completed MC Iteration {i + 1}/{MC_ITERATIONS}")


# ---------------------------------------------------------
# 4. Analysis & Visualization
# ---------------------------------------------------------
# ---------------------------------------------------------
# PHASE 4: Spatial Residual Evaluation (RIC)
# ---------------------------------------------------------
def evaluate_mc_ric_residuals(
    mc_states: list[torch.Tensor], 
    propagator: torch.nn.Module, 
    t_gps_truth: torch.Tensor, 
    r_gps_truth: torch.Tensor, 
    v_gps_truth: torch.Tensor, 
    epoch: float
):
    print("\n--- Phase 4: Computing Ensemble RIC Residuals ---")
    all_pos_ric = []
    
    for i, x_state in enumerate(mc_states):
        # compute_ric_residuals returns (t_mins, pos_ric_km, vel_ric_km)
        _, pos_ric, _ = compute_ric_residuals(
            x_state=x_state,
            propagator=propagator,
            t_gps=t_gps_truth,
            r_gps=r_gps_truth,
            v_gps=v_gps_truth,
            tle_epoch_unix=epoch
        )
        all_pos_ric.append(pos_ric) # Shape: (N_gps_samples, 3)

    # Stack into a single tensor of shape: (MC_Iterations, N_gps_samples, 3)
    # Dimension 2 indices: 0 = Radial, 1 = In-track (Along-track), 2 = Cross-track
    ensemble_ric = torch.stack(all_pos_ric)
    
    # Compute Grand Mean and Variance across both the MC ensemble and time domains
    # We reduce over dim 0 (Iterations) and dim 1 (Time steps)
    grand_mean = torch.sqrt(torch.mean(ensemble_ric**2, dim=(0, 1))) #/ len(ensemble_ric)
    grand_var = torch.var(ensemble_ric, dim=(0, 1))
    
    print("\n--- Ensemble Spatial Error Statistics (km) ---")
    print(f"{'Axis':<15} | {'Mean (km)':<12} | {'Variance (km^2)':<15} | {'Std Dev (km)':<12}")
    print("-" * 62)
    
    axes = ["Radial", "In-track", "Cross-track"]
    for i, axis in enumerate(axes):
        mean_val = grand_mean[i].item()
        var_val = grand_var[i].item()
        std_val = torch.sqrt(grand_var[i]).item()
        print(f"{axis:<15} | {mean_val:>12.6f} | {var_val:>15.6f} | {std_val:>12.6f}")

    return ensemble_ric, grand_mean, grand_var

# Execute the evaluation using the GPS propagator from Phase 1
ensemble_ric_tensor, g_mean, g_var = evaluate_mc_ric_residuals(
    mc_states=mc_states,
    propagator=prop_gps, # The SGP4 propagator initialized for the truth baseline
    t_gps_truth=t_gps,
    r_gps_truth=r_gps,
    v_gps_truth=v_gps,
    epoch=T_mean
)

import matplotlib.pyplot as plt
import numpy as np

def plot_mc_uncertainty_dashboard(
    ensemble_ric: torch.Tensor, 
    t_gps: torch.Tensor, 
    mc_results_n: list[float], 
    mc_results_L: list[float]
):
    print("\n--- Generating Uncertainty Dashboard ---")
    
    # ---------------------------------------------------------
    # 1. Data Processing
    # ---------------------------------------------------------
    # Time axis in hours from the start of the epoch for readability
    t_hours = (t_gps - t_gps[0]).cpu().numpy() / 3600.0
    
    # Calculate Time-Series Statistics across the MC dimension (dim=0)
    # Shapes will be (N_timepoints, 3)
    ric_mean = torch.mean(ensemble_ric, dim=0).cpu().numpy()
    ric_std = torch.std(ensemble_ric, dim=0).cpu().numpy()
    
    # Calculate Spatial RMS per MC Iteration for the Boxplots
    # Shape transitions to (MC_Iterations, 3)
    ric_rms_per_iter = torch.sqrt(torch.mean(ensemble_ric**2, dim=1)).cpu().numpy()
    
    n_array = np.array(mc_results_n)
    L_array = np.array(mc_results_L)

    # ---------------------------------------------------------
    # 2. Figure Setup
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, wspace=0.3, hspace=0.4)
    
    # --- Plot A: 3D Spatial Confidence Bounds Over Time ---
    ax_rad = fig.add_subplot(gs[0, :2])
    ax_int = fig.add_subplot(gs[1, :2])
    ax_cross = fig.add_subplot(gs[2, :2])
    
    time_axes = [ax_rad, ax_int, ax_cross]
    labels = ['Radial (km)', 'In-track (km)', 'Cross-track (km)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i in range(3):
        ax = time_axes[i]
        mean_val = ric_mean[:, i]
        std_val = ric_std[:, i]
        
        # Plot the ensemble mean
        ax.plot(t_hours, mean_val, color=colors[i], label='Ensemble Mean')
        
        # Plot the 3-sigma uncertainty blob
        ax.fill_between(
            t_hours, 
            mean_val - 3 * std_val, 
            mean_val + 3 * std_val, 
            color=colors[i], alpha=0.2, label=r'$\pm 3\sigma$ Bound'
        )
        
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Spatial Uncertainty Envelopes Over Time")
            ax.legend(loc='upper right')
        if i == 2:
            ax.set_xlabel("Time since start of trajectory (Hours)")
            
    # --- Plot B: Spatial RMS Boxplots ---
    ax_box_spat = fig.add_subplot(gs[0:2, 2:])
    ax_box_spat.boxplot(
        [ric_rms_per_iter[:, 0], ric_rms_per_iter[:, 1], ric_rms_per_iter[:, 2]],
        labels=['Radial', 'In-track', 'Cross-track'],
        patch_artist=True,
        boxprops=dict(color='lightgray', alpha=0.6),
        medianprops=dict(color='red', linewidth=2)
    )
    ax_box_spat.set_title("Distribution of Spatial RMS Errors\n(Across Monte Carlo Iterations)")
    ax_box_spat.set_ylabel("RMS Error (km)")
    ax_box_spat.grid(True, axis='y', alpha=0.3)

    # --- Plot C: Parameter Boxplots ---
    # Because n and L have completely different scales and physical units, 
    # it's best to split them into two separate side-by-side boxplots.
    ax_box_n = fig.add_subplot(gs[2, 2])
    ax_box_n.boxplot(n_array, labels=['Mean Motion ($n$)'], widths=0.5,
                     boxprops=dict(color='#1f77b4', alpha=0.6))
    ax_box_n.set_ylabel("Revs / Day")
    ax_box_n.grid(True, axis='y', alpha=0.3)
    
    ax_box_L = fig.add_subplot(gs[2, 3])
    ax_box_L.boxplot(L_array, labels=['Mean Longitude ($L$)'], widths=0.5,
                     boxprops=dict(color='#ff7f0e', alpha=0.6))
    ax_box_L.set_ylabel("Radians")
    ax_box_L.grid(True, axis='y', alpha=0.3)

    plt.suptitle("Monte Carlo Orbit Determination Analysis", fontsize=16, y=0.95)
    plt.show()

# ---------------------------------------------------------
# Execute the Dashboard
# ---------------------------------------------------------
# Assuming ensemble_ric_tensor is returned from the previous Phase 4 step
plot_mc_uncertainty_dashboard(
    ensemble_ric=ensemble_ric_tensor,
    t_gps=t_gps,
    mc_results_n=mc_results_n,
    mc_results_L=mc_results_L
)