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
from diffod.utils import unix_to_mjd, load_gmat_csv_block_legacy, parameter_covariance_propagation, extract_cca_priors
from diffod.visualize import compute_ric_residuals, plot_segment_ric_residuals, plot_median_forecast_trends, plot_observability_growth
from diffod.solvers.gn_svd import svd_solve
from diffod.solvers.cca import cca_solve

# =========================================================
# 1. Configuration 
# =========================================================
device = torch.device(device="cpu")
dtype = torch.float64
center_freq = 1707.0  

KNOWN_GLOBAL_TIME_BIAS_SEC = 0.277  
MC_ITERATIONS = 10
STATE_NOISE_SCALE = 1e-4       
DOPPLER_NOISE_STD_HZ = 5.0       

TLE_list = [
    "AWS",
    "1 60543U 24149CD  25307.42878472  .00000000  00000-0  11979-3 0    11",
    "2 60543  97.7067  19.0341 0003458 123.1215 316.4897 14.89807169 65809",
]
tle_base = TLE(data=TLE_list)
epoch_unix = 1762207191

# =========================================================
# 2. Function Definitions
# =========================================================
def create_pass_chunks(t, d, c, passes_per_chunk=6, num_chunks=4):
    """
    Splits the dataset into independent chunks and remaps contact 
    indices to be contiguous (e.g., [4,4,5,5] -> [0,0,1,1]).
    """
    unique_passes = torch.unique(c)
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * passes_per_chunk
        end_idx = start_idx + passes_per_chunk
        
        if end_idx > len(unique_passes):
            print(f"Warning: Only enough data for {i} full chunks of {passes_per_chunk} passes.")
            break
            
        # 1. Mask the original data
        chunk_passes = unique_passes[start_idx:end_idx]
        mask = torch.isin(c, chunk_passes)
        
        c_masked = c[mask]
        
        # 2. Remap contact indices to a 0-based contiguous sequence
        # original_unique will hold [4, 5, 6...], c_remapped will hold [0, 0... 1, 1...]
        original_unique, c_remapped = torch.unique(c_masked, return_inverse=True)
        
        chunks.append({
            "chunk_id": i,
            "t": t[mask],
            "d_true": d[mask],
            "c": c_remapped, 
            "pass_ids": torch.unique(c_remapped),   # Use the new 0, 1, 2... IDs for the inner loop
            "original_pass_ids": original_unique    # Keep the global IDs just in case you need them for logging
        })
        
    return chunks

def run_doppler_od(
    x_guess: torch.Tensor, 
    t_obs: torch.Tensor, 
    d_obs: torch.Tensor, 
    c_obs: torch.Tensor,
    ref_unix: float,
    base_tle: dsgp4.TLE,
    center_freq: float = 1707.0,
    time_bias_sec: float = 0.277
) -> tuple[torch.Tensor, torch.Tensor, object, object]:
    """Runs Doppler-only OD and returns the state, covariance, SSV, and propagator."""
    ssv_dopp = state.MEE_SSV(
        init_tle=base_tle, 
        num_measurements=len(t_obs),
        fit_mean_motion=True, fit_f=False, fit_g=True,  
        fit_h=False, fit_k=False, fit_L=True, fit_bstar=False
    )
    ssv_dopp.add_linear_bias(name="pass_freq_bias", group_indices=c_obs)

    t_ref_astropy = Time(ref_unix, format="unix", scale="utc")
    station_model = system.DifferentiableStation(
        lat_deg=78.228874, lon_deg=15.376932, alt_m=463.0, 
        ref_unix=ref_unix, 
        ref_gmst_rad=t_ref_astropy.sidereal_time('mean', 'greenwich').radian,
        device=device
    )

    prop_dopp = system.SGP4(ssv=ssv_dopp)
    meas_dopp = system.DopplerMeasurement(
        ssv=ssv_dopp, station_model=station_model,
        freq_bias_group=ssv_dopp.get_bias_group("pass_freq_bias"), 
        time_bias_group=None
    )
    pipe_dopp = system.MeasurementPipeline(propagator=prop_dopp, measurement_model=meas_dopp)

    t_since_dopp_calibrated = (t_obs - ref_unix) + time_bias_sec

    def forward_fn(x):
        return pipe_dopp(x=x, tsince=t_since_dopp_calibrated, epoch=ref_unix, center_freq=center_freq)

    # Note: ensure svd_solve returns the covariance matrix as the second argument
    x_out, cov = svd_solve(
        x_init=x_guess, y_obs_fixed=d_obs, forward_fn=forward_fn,
        sigma_obs=20.0, estimate_mask=ssv_dopp.get_active_map(), num_steps=10
    )
    
    return x_out, cov, ssv_dopp, prop_dopp


def evaluate_forecast_metrics(
    x_state: torch.Tensor, 
    propagator: torch.nn.Module, 
    t_gps: torch.Tensor, 
    r_gps: torch.Tensor, 
    v_gps: torch.Tensor, 
    epoch_unix: float, 
    t_end_obs: float
) -> dict[str, float]:
    """Computes 3D and decomposed RIC RMSE for prediction horizons."""
    metrics = {}
    horizons = [(1, '1h'), (6, '6h'), (24, '24h')]
    
    for hours, label in horizons:
        mask = (t_gps > t_end_obs) & (t_gps <= t_end_obs + hours * 3600)
        
        if mask.sum() == 0:
            metrics[label] = np.nan
            metrics[f'{label}_R'] = np.nan
            metrics[f'{label}_I'] = np.nan
            metrics[f'{label}_C'] = np.nan
            continue
            
        _, pos_ric, _ = compute_ric_residuals(
            x_state=x_state, propagator=propagator,
            t_gps=t_gps[mask], r_gps=r_gps[mask], v_gps=v_gps[mask], 
            tle_epoch_unix=epoch_unix
        )
        
        # Calculate full 3D RMSE
        metrics[label] = torch.sqrt(torch.mean(torch.sum(pos_ric**2, dim=1))).item()
        
        # Decompose into Radial (0), In-track (1), and Cross-track (2) RMSE
        rms_ric = torch.sqrt(torch.mean(pos_ric**2, dim=0))
        metrics[f'{label}_R'] = rms_ric[0].item()
        metrics[f'{label}_I'] = rms_ric[1].item()
        metrics[f'{label}_C'] = rms_ric[2].item()
        
    return metrics

# =========================================================
# 3. Data Loading
# =========================================================
print("Loading Synthetic Dataset & GPS Truth...")
t_gps, r_gps, v_gps = load_gmat_csv_block_legacy(
    file_path="data/AWS_full_long_period.csv", 
    tle_epoch_unix=epoch_unix, block_sec=86400 * 2
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

# =========================================================
# PHASE 1: GPS-Based Baseline OD
# =========================================================
print("\n--- Phase 1: Fitting Baseline TLE to GPS ---")
init_tle_gps, _ = dsgp4.newton_method(tle_base, unix_to_mjd(T_mean))

ssv_gps = state.MEE_SSV(init_tle=init_tle_gps, num_measurements=len(t_gps), fit_bstar=False)
prop_gps = system.SGP4(ssv=ssv_gps)
meas_gps = system.CartesianMeasurement(ssv=ssv_gps)
pipe_gps = system.MeasurementPipeline(propagator=prop_gps, measurement_model=meas_gps)

y_gps_1d = meas_gps.format_gps_observations(r_gps, v_gps)
t_since_gps = (t_gps - T_mean) 

x_gps_out, cov_gps_epoch = svd_solve(
    x_init=ssv_gps.get_initial_state(), y_obs_fixed=y_gps_1d,
    forward_fn=lambda x: pipe_gps(x=x, tsince=t_since_gps),
    estimate_mask=ssv_gps.get_active_map(), num_steps=5, sigma_obs=1.0 
)
tle_gps_fit = ssv_gps.export(x_gps_out)



# =========================================================
# PHASE 2: Multi-Dimensional Monte Carlo Simulation
# =========================================================
print("\n--- Phase 2: Multi-Dimensional Epoch-Centered Simulation ---")

data_chunks = create_pass_chunks(t_dopp, d_dopp_true, c_dopp, passes_per_chunk=6, num_chunks=4)
passes_to_test = [0, 1, 2, 3, 4, 5, 6]  # Added 0 to the test list
experiment_results = []
segment_ric_trajectories = {}
for chunk in data_chunks:
    print(f"\nProcessing Dataset Chunk {chunk['chunk_id']}...")
    
    for num_passes in passes_to_test:
        
        # ==========================================
        # BASELINE (0-PASS) LOGIC
        # ==========================================
        if num_passes == 0:
            # Use the first pass to establish a baseline epoch and end-time
            active_pass_ids = chunk["pass_ids"][:1]
            pass_mask = torch.isin(chunk["c"], active_pass_ids)
            t_active = chunk["t"][pass_mask]
            
            t_mean_window = float(torch.mean(t_active))
            t_end_window = float(t_active[-1])
            
            # Center the global GPS TLE to this specific window
            tle_window, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
            
            # Setup a baseline SSV and propagator (no measurement tracking needed)
            baseline_ssv = state.MEE_SSV(init_tle=tle_window, num_measurements=1, fit_mean_motion=False)
            prop_baseline = system.SGP4(ssv=baseline_ssv)
            
            # Evaluate the prior state
            metrics = evaluate_forecast_metrics(
                x_state=baseline_ssv.get_initial_state(), 
                propagator=prop_baseline,
                t_gps=t_gps, r_gps=r_gps, v_gps=v_gps, 
                epoch_unix=t_mean_window, t_end_obs=t_end_window
            )
            
            experiment_results.append({
                "chunk_id": chunk["chunk_id"],
                "num_passes": 0,
                "mc_iteration": 0, # Added for schema consistency
                "1h_rmse": metrics.get('1h', np.nan),
                "6h_rmse": metrics.get('6h', np.nan),
                "24h_rmse": metrics.get('24h', np.nan),
                "24h_R": np.nanmean(metrics.get('24h_R', [np.nan])),
                "24h_I": np.nanmean(metrics.get('24h_I', [np.nan])),
                "24h_C": np.nanmean(metrics.get('24h_C', [np.nan])),
                "cov_frob": np.nan 
            })
            
            print(f"  -> Passes: 0 | 1h RMSE: {metrics.get('1h', np.nan):.2f} km | 24h RMSE: {metrics.get('24h', np.nan):.2f} km (Baseline)")
            continue

        # ==========================================
        # OD (>0 PASS) LOGIC
        # ==========================================
        active_pass_ids = chunk["pass_ids"][:num_passes]
        pass_mask = torch.isin(chunk["c"], active_pass_ids)
        
        t_active = chunk["t"][pass_mask]
        d_active_true = chunk["d_true"][pass_mask]
        c_active = chunk["c"][pass_mask]
        
        # 1. Shift the reference epoch
        t_mean_window = float(torch.mean(t_active))
        t_end_window = float(t_active[-1])
        tle_window, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_mean_window))
        
        dummy_ssv = state.MEE_SSV(init_tle=tle_window, num_measurements=len(t_active), fit_mean_motion=True, fit_L=True)
        dummy_ssv.add_linear_bias(name="pass_freq_bias", group_indices=c_active)
        standard_x_guess = dummy_ssv.get_initial_state()
        
        # mc_metrics = {'1h': [], '6h': [], '24h': [], 'frob': []}
        
        for i in range(MC_ITERATIONS):
            noise_vector = torch.randn_like(standard_x_guess) * STATE_NOISE_SCALE
            x_noisy_guess = standard_x_guess + (standard_x_guess * noise_vector)
            d_active_noisy = d_active_true + torch.randn_like(d_active_true) * DOPPLER_NOISE_STD_HZ
            
            x_mc_out, cov_mc, _, prop_mc = run_doppler_od(
                x_guess=x_noisy_guess, t_obs=t_active, d_obs=d_active_noisy, c_obs=c_active,
                ref_unix=t_mean_window, base_tle=tle_window, time_bias_sec=KNOWN_GLOBAL_TIME_BIAS_SEC
            )   

            if num_passes == 6 and i == 0:
                # Create a mask for the 24 hours immediately following this segment's epoch
                t_forecast_mask = (t_gps >= t_mean_window) & (t_gps <= t_mean_window + 24 * 3600)
                
                # Compute the residuals for this specific window
                t_mins, pos_ric, vel_ric = compute_ric_residuals(
                    x_state=x_mc_out, 
                    propagator=prop_mc,
                    t_gps=t_gps[t_forecast_mask], 
                    r_gps=r_gps[t_forecast_mask], 
                    v_gps=v_gps[t_forecast_mask],
                    tle_epoch_unix=t_mean_window
                )
                segment_ric_trajectories[f"Segment {chunk['chunk_id']}"] = (t_mins, pos_ric, vel_ric)
                        
            # Evaluate metrics for THIS iteration
            metrics = evaluate_forecast_metrics(
                x_state=x_mc_out, propagator=prop_mc,
                t_gps=t_gps, r_gps=r_gps, v_gps=v_gps, 
                epoch_unix=t_mean_window, t_end_obs=t_end_window
            )
            
            # Append immediately
            experiment_results.append({
                "chunk_id": chunk["chunk_id"],
                "num_passes": num_passes,
                "mc_iteration": i,
                "1h_rmse": metrics.get('1h', np.nan),
                "6h_rmse": metrics.get('6h', np.nan),
                "24h_rmse": metrics.get('24h', np.nan),
                "cov_frob": torch.sum(torch.diag(cov_mc)[:2]).item()
            })
            print(f"  -> Passes: {num_passes} | 1h RMSE: {np.nanmean(metrics['1h']):.2f} km | 24h RMSE: {np.nanmean(metrics['24h']):.2f} km")


            # metrics = evaluate_forecast_metrics(
            #     x_state=x_mc_out, propagator=prop_mc,
            #     t_gps=t_gps, r_gps=r_gps, v_gps=v_gps, 
            #     epoch_unix=t_mean_window, t_end_obs=t_end_window
            # )
            # for k in ['1h', '6h', '24h']:
            #     mc_metrics[k].append(metrics[k])
                
        # experiment_results.append({
        #     "chunk_id": chunk["chunk_id"],
        #     "num_passes": num_passes,
        #     "1h_rmse": np.nanmean(mc_metrics['1h']),
        #     "6h_rmse": np.nanmean(mc_metrics['6h']),
        #     "24h_rmse": np.nanmean(mc_metrics['24h']),
        #     "cov_frob": np.nanmean(mc_metrics['frob'])
        # })
        


import seaborn as sns

def plot_multipass_forecast_boxplots(results: list[dict]):
    """Generates grouped box plots for OD performance metrics across MC iterations."""
    # Convert Polars DataFrame to Pandas, as Seaborn integrates natively with Pandas
    df = pl.DataFrame(results).to_pandas()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1-Hour Forecast Distribution
    sns.boxplot(
        data=df, x="num_passes", y="1h_rmse", hue="chunk_id", 
        ax=axes[0], palette="Set2", showfliers=True
    )
    axes[0].set_title("1-Hour Forecast RMSE Distribution")
    axes[0].set_xlabel("Number of Passes")
    axes[0].set_ylabel("RMSE (km)")
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot 24-Hour Forecast Distribution
    sns.boxplot(
        data=df, x="num_passes", y="24h_rmse", hue="chunk_id", 
        ax=axes[1], palette="Set2", showfliers=True
    )
    axes[1].set_title("24-Hour Forecast RMSE Distribution")
    axes[1].set_xlabel("Number of Passes")
    axes[1].set_ylabel("RMSE (km)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Adjust legends to not overlap with data
    for ax in axes:
        ax.legend(title="Data Segments", loc='upper right')

    plt.tight_layout()
    plt.show()

# Execute plotting
# plot_cross_validation(experiment_results)
plot_multipass_forecast_boxplots(experiment_results)
plot_segment_ric_residuals(segment_ric_trajectories)
plot_median_forecast_trends(experiment_results)
plot_observability_growth(data_chunks, T_mean, tle_base, center_freq)