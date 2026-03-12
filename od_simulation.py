import torch
import numpy as np
import polars as pl
from collections import defaultdict

import dsgp4
from dsgp4.tle import TLE
from astropy.time import Time

# Internal diffod modules
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import unix_to_mjd, load_gmat_csv_block_legacy
from diffod.visualize import compute_ric_residuals, plot_ric_residuals
from diffod.solvers.gn_svd import svd_solve

# ---------------------------------------------------------
# 1. Configuration & Data Loading
# ---------------------------------------------------------
device = torch.device(device="cpu")
dtype = torch.float64

known_global_time_bias_sec = 0.277

TLE_list = [
    "AWS",
    "1 60543U 24149CD  25307.42878472  .00000000  00000-0  11979-3 0    11",
    "2 60543  97.7067  19.0341 0003458 123.1215 316.4897 14.89807169 65809",
]
tle_base = TLE(data=TLE_list)
epoch_unix = 1762207191

print("Loading GPS Truth...")
t_gps, r_gps, v_gps = load_gmat_csv_block_legacy(
    file_path="data/AWS_full_long_period.csv", 
    tle_epoch_unix=epoch_unix,
    block_sec=86400 * 7  # Expanded block size to ensure enough data for rolling windows
)

# Time constants (in seconds)
FIT_WINDOW = 2 * 3600
FORECAST_HORIZONS = {
    "1-hour": 3600,
    "6-hour": 6 * 3600,
    "24-hour": 24 * 3600
}
EVAL_TOLERANCE = 300  # Evaluate within a +/- 5 minute slice around the target horizon

# ---------------------------------------------------------
# 2. Core Fitting Function
# ---------------------------------------------------------
def fit_orbit_chunk(t_fit, r_fit, v_fit, use_ml=False):
    """Fits an orbit to a given time slice of GPS data."""
    t_chunk_mean = float(torch.mean(t_fit))
    
    # Initialize state for this specific chunk's epoch
    init_tle, _ = dsgp4.newton_method(tle_base, unix_to_mjd(t_chunk_mean))
    ssv = state.MEE_SSV(init_tle=init_tle, num_measurements=len(t_fit), fit_bstar=False)
    
    prop = system.SGP4(ssv=ssv, use_pretrained_model=use_ml)
    meas = system.CartesianMeasurement(ssv=ssv)
    pipe = system.MeasurementPipeline(propagator=prop, measurement_model=meas)

    y_obs_1d = meas.format_gps_observations(r_fit, v_fit)
    t_since_mean = t_fit - t_chunk_mean 

    x_out, _ = svd_solve(
        x_init=ssv.get_initial_state(),
        y_obs_fixed=y_obs_1d,
        forward_fn=lambda x: pipe(x=x, tsince=t_since_mean),
        estimate_mask=ssv.get_active_map(),
        num_steps=5,
        sigma_obs=1.0 
    )
    return x_out, prop, t_chunk_mean

# ---------------------------------------------------------
# 3. Rolling Window Evaluation
# ---------------------------------------------------------
print("\n--- Running Rolling Window OD & Forecasting ---")

t_start = float(t_gps[0])
t_end_total = float(t_gps[-1])
max_forecast = max(FORECAST_HORIZONS.values())

# Dictionary to store mean RIC position errors for aggregation
error_log = {
    "Base": defaultdict(list),
    "ML": defaultdict(list)
}

current_t = t_start
chunk_idx = 0

while current_t + FIT_WINDOW + max_forecast <= t_end_total:
    print(f"Processing Chunk {chunk_idx + 1}...")
    try:    
        # 1. Extract the 2-hour fit data
        fit_mask = (t_gps >= current_t) & (t_gps < current_t + FIT_WINDOW)
        t_fit, r_fit, v_fit = t_gps[fit_mask], r_gps[fit_mask], v_gps[fit_mask]
        
        if len(t_fit) < 10:
            current_t += FIT_WINDOW
            continue
            
        # 2. Fit both models
        x_base, prop_base, t_mean_base = fit_orbit_chunk(t_fit, r_fit, v_fit, use_ml=False)
        x_ml, prop_ml, t_mean_ml = fit_orbit_chunk(t_fit, r_fit, v_fit, use_ml=True)
        
        # 3. Evaluate at forecast horizons
        fit_end_time = current_t + FIT_WINDOW
        
        for label, horizon_sec in FORECAST_HORIZONS.items():
            target_t = fit_end_time + horizon_sec
            
            # Grab a small slice of data around the target horizon for a stable error metric
            eval_mask = (t_gps >= target_t - EVAL_TOLERANCE) & (t_gps <= target_t + EVAL_TOLERANCE)
            t_eval, r_eval, v_eval = t_gps[eval_mask], r_gps[eval_mask], v_gps[eval_mask]
            
            if len(t_eval) == 0:
                continue
                
            # Base Eval
            _, pos_err_base, _ = compute_ric_residuals(
                x_state=x_base, propagator=prop_base, 
                t_gps=t_eval, r_gps=r_eval, v_gps=v_eval, 
                tle_epoch_unix=t_mean_base
            )
            # Store mean position error magnitude across the evaluation slice
            err_mag_base = torch.norm(pos_err_base, dim=1).mean().item()
            error_log["Base"][label].append(err_mag_base)
            
            # ML Eval
            _, pos_err_ml, _ = compute_ric_residuals(
                x_state=x_ml, propagator=prop_ml, 
                t_gps=t_eval, r_gps=r_eval, v_gps=v_eval, 
                tle_epoch_unix=t_mean_ml
            )
            err_mag_ml = torch.norm(pos_err_ml, dim=1).mean().item()
            error_log["ML"][label].append(err_mag_ml)

        # Advance the rolling window (tumbling window)
        current_t += FIT_WINDOW
        chunk_idx += 1
    except Exception:
        print("Run failed")
        current_t += FIT_WINDOW
        chunk_idx += 1

# [Image of RIC (Radial, In-track, Cross-track) satellite coordinate frame]

# ---------------------------------------------------------
# 4. Results Aggregation
# ---------------------------------------------------------
print("\n--- Final Forecast Error Summary (Mean 3D RIC Position Error) ---")
print(f"{'Horizon':<10} | {'Base dSGP4 (m)':<15} | {'ML-dSGP4 (m)':<15}")
print("-" * 65)

for label in FORECAST_HORIZONS.keys():
    base_errors = error_log["Base"][label]
    ml_errors = error_log["ML"][label]
    
    if not base_errors:
        continue
        
    mean_base = np.mean(base_errors)
    mean_ml = np.mean(ml_errors)
    
    print(f"{label:<10} | {mean_base:<15.2f} | {mean_ml:<15.2f}")