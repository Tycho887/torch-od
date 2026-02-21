from doctest import DocTestParser
import re
from string import whitespace
import time
import dsgp4
import numpy as np
from requests import get
import torch
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd

from dsgp4.tle import TLE
from torch.utils.show_pickle import FakeClass
import diffod.state as state
from diffod.functional.system import PredictDoppler
from diffod.utils import load_gmat_csv_block, get_tle_epoch
from diffod.gse import station_teme_preprocessor

# Swapped out CCA for the WGN solver
from diffod.solvers.gaussNewton import wgn_solve
from diffod.solvers.newton import newton_solve
from diffod.solvers.cca import cca_solve


# ---------------------------------------------------------
# 1. Setup Data & Boundary
# ---------------------------------------------------------
TLE_list = [
    "AWS",
    "1 60543U 24149CD  25307.42878472  .00000000  00000-0  11979-3 0    11",
    "2 60543  97.7067  19.0341 0003458 123.1215 316.4897 14.89807169 65809",
]

epoch = 1762165047
init_tle = TLE(data=TLE_list)
stations = {
    0: np.array(object=[15.376932, 78.228874, 0.463]),
}

target_device = torch.device("cpu")

# Load real telemetry data
period_telemetry = pl.read_parquet(source="data/period_telemetry.parquet")
# gps_data = pl.read_csv(source="data/gps_data.csv")
# gps_data_pd = pd.read_fwf("data/gps_data.csv")
# gps_data = pl.from_pandas(gps_data_pd)
# print(gps_data)

"""
Sat.UTCGregorian          
Sat.TEME_Earth.X          
Sat.TEME_Earth.Y          
Sat.TEME_Earth.Z          
Sat.TEME_Earth.VX         
Sat.TEME_Earth.VY         
Sat.TEME_Earth.VZ
"""

times_unix = torch.tensor(period_telemetry["timestamp"].to_numpy(), dtype=torch.float64, device=target_device)
doppler_obs = torch.tensor(period_telemetry["Doppler_Hz"].to_numpy(), dtype=torch.float64, device=target_device)
contacts = torch.tensor(period_telemetry["contact_index"].to_numpy(), dtype=torch.int32, device=target_device)

N_samples = len(times_unix)
t_obs = (times_unix - epoch) #/ 60.0

st_indices = torch.zeros(N_samples, dtype=torch.int32, device=target_device)

print("Preprocessing Astropy arrays on CPU...")
station_pos_cpu, station_vel_cpu = station_teme_preprocessor(
    times_s=times_unix.numpy(),
    station_ids=st_indices.numpy(),
    id_to_station=stations,
    dtype=torch.float64,
    device=target_device,
)

st_pos = station_pos_cpu.to(device=target_device)
st_vel = station_vel_cpu.to(device=target_device)

# ---------------------------------------------------------
# 2. Define State Vector & Functional Forward
# ---------------------------------------------------------
state_def = state.StateDefinition(
    init_tle=init_tle,
    num_measurements=N_samples,
    fit_ma=True,
    fit_mean_motion=True,
    fit_argp=False,
    fit_bstar=False,
    fit_inclination=False,
    fit_eccentricity=False,
    fit_raan=False,
)
state_def.add_linear_bias(name="doppler_bias", group_indices=contacts)

model = PredictDoppler(
    state_def=state_def, bias_group=state_def.get_bias_group(name="doppler_bias")
)

def functional_forward(x) -> torch.Tensor:
    return model(x=x, tsince=t_obs/60.0, st_pos=st_pos, st_vel=st_vel, center_freq=1.707e9)

# ---------------------------------------------------------
# 3. Generate Perturbed Starts for Monte Carlo
# ---------------------------------------------------------
x_true = state_def.get_initial_state(device=target_device)

N_solves = 1
x_init_batch = x_true.unsqueeze(0).repeat(N_solves, 1)
x_init_batch = x_init_batch + x_init_batch * torch.randn_like(x_init_batch) * 1e-4

# ---------------------------------------------------------
# 4. WGN Matrix Configurations
# ---------------------------------------------------------
sigma_obs = 10.0  
n_total = x_true.shape[0]

active_map = state_def.get_active_map(device=target_device)
consider_map = ~active_map

n_active = int(active_map.sum().item())
n_consider = n_total - n_active

print(f"Total Params: {n_total} | Estimated: {n_active} | Considered: {n_consider}")

# Prior information matrices for the solvers
P_x_inv = torch.eye(n=n_active, dtype=torch.float64, device=target_device) # * 1e-6
P_cc = torch.eye(n=n_consider, dtype=torch.float64, device=target_device)

# ---------------------------------------------------------
# 5. Execute Sequential Iterations & Benchmark
# ---------------------------------------------------------
print("\nExecuting OD Solvers...")

# Extract the single perturbed initial state for direct comparison
x_in = x_init_batch[0]

# Dictionary to store results
results = {}

solvers = {
    "Gauss-Newton (WGN)": wgn_solve,
    "Exact Newton": newton_solve,
    "Consider Covariance (CCA)": cca_solve,
}

# Run each solver and record metrics
with torch.no_grad():
    for name, solver_fn in solvers.items():
        t0 = time.perf_counter()
        
        # Dispatch kwargs dynamically since CCA requires more specific inputs
        kwargs = {
            "x_init": x_in,
            "y_obs_fixed": doppler_obs,
            "forward_fn": functional_forward,
            "sigma_obs": sigma_obs,
            "estimate_mask": active_map,
            "num_steps": 5,
        }
        
        if name == "Consider Covariance (CCA)":
            kwargs.update({
                "consider_mask": consider_map,
                "P_cc": P_cc,
                "P_x_inv": P_x_inv,
                # "n_estimated": n_active,
            })
            
        x_out, P_out = solver_fn(**kwargs)
        
        t1 = time.perf_counter()
        exec_time = (t1 - t0) * 1000
        
        results[name] = {
            "x": x_out,
            "P": P_out,
            "time_ms": exec_time
        }
        print(f"{name:25} | Time: {exec_time:6.2f} ms | Cov Trace: {torch.trace(P_out[:7, :7]).item():.4e}")

print("\nConvergence Check (Final States):")
print(f"Initial State: {x_in[:7].detach().cpu().numpy()}")
for name, res in results.items():
    print(f"{name:25}: {res['x'][:7].detach().cpu().numpy()}")

# ---------------------------------------------------------
# 6. Plotting Results
# ---------------------------------------------------------
print("\nGenerating plots...")

# Precompute initial model fit
with torch.no_grad():
    y_init = functional_forward(x_in).cpu().numpy()

t_plot = (times_unix - times_unix[0]).cpu().numpy() / 60.0
y_obs = doppler_obs.cpu().numpy()

plt.figure(figsize=(14, 8))

# Plot observations and initial guess
plt.scatter(t_plot, y_obs, s=4, c='black', alpha=0.3, label='Observations')
plt.plot(t_plot, y_init, 'r--', linewidth=1.5, alpha=0.6, label='Initial Guess')

# Plot styling for each solver to make them distinguishable
styles = {
    "Gauss-Newton (WGN)": {"color": "blue", "linestyle": "-", "alpha": 0.8, "linewidth": 2},
    "Exact Newton": {"color": "purple", "linestyle": "--", "alpha": 0.8, "linewidth": 2.5},
    "Consider Covariance (CCA)": {"color": "green", "linestyle": ":", "alpha": 1.0, "linewidth": 3},
}

# Compute and plot fitted curves for all solvers
with torch.no_grad():
    for name, res in results.items():
        y_fit = functional_forward(res["x"]).cpu().numpy()
        
        # Calculate RMSE for the legend
        rmse = np.sqrt(np.mean((y_fit - y_obs)**2))
        
        plt.plot(
            t_plot, 
            y_fit, 
            label=f'{name} (RMSE: {rmse:.2f} Hz)', 
            **styles[name]
        )

plt.title('Doppler Shift Orbit Determination Fit Comparison', fontsize=14)
plt.xlabel('Time since start of track (Minutes)', fontsize=12)
plt.ylabel('Doppler Shift (Hz)', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.close()

tle = state_def.export(results["Gauss-Newton (WGN)"]["x"])

print(tle)

# t_gps, gps_pos, gps_vel = load_gmat_csv_block(
#     file_path="data/gps_data.csv",
#     tle_epoch_unix = get_tle_epoch(tle),
#     block_sec=24*60*60,
# )

import torch
import matplotlib.pyplot as plt
import dsgp4 # Assuming this is the module for propagation

def compute_ric_residuals(
    tle, 
    t_gps: torch.Tensor, 
    r_gps: torch.Tensor, 
    v_gps: torch.Tensor, 
    tle_epoch_unix: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes position and velocity residuals in the Radial, Along-track, 
    and Cross-track (RIC) frame.
    
    Expects r_gps and v_gps in meters (from load_gmat_csv_block). 
    Outputs residuals in kilometers to match SGP4 and plotting defaults.
    """
    # 1. Convert GPS inputs from meters back to kilometers for SGP4 math
    r_gps_km = r_gps.to(torch.float64) / 1000.0
    v_gps_km = v_gps.to(torch.float64) / 1000.0
    t_gps = t_gps.to(torch.float64)

    # 2. Propagate TLE to GPS timestamps
    t_since_mins = (t_gps - tle_epoch_unix) / 60.0
    
    # dsgp4 returns shape (N, 2, 3) where [:, 0] is pos, [:, 1] is vel
    r_tot = dsgp4.propagate(tle, t_since_mins, initialized=False)
    r_calc_km = r_tot[:, 0]
    v_calc_km = r_tot[:, 1]

    # 3. Calculate Cartesian Errors (Calculated - Truth)
    delta_r = r_calc_km - r_gps_km
    delta_v = v_calc_km - v_gps_km

    # 4. Construct RIC Basis Vectors (using GPS state as the reference)
    # Radial unit vector
    R_hat = r_gps_km / torch.norm(r_gps_km, dim=1, keepdim=True)
    
    # Cross-track (Normal) unit vector: r x v
    W_vec = torch.cross(r_gps_km, v_gps_km, dim=1)
    W_hat = W_vec / torch.norm(W_vec, dim=1, keepdim=True)
    
    # Along-track (Transverse) unit vector: w x r
    S_hat = torch.cross(W_hat, R_hat, dim=1)

    # 5. Project Cartesian errors into the RIC frame
    # Using batch dot product (summing over the coordinate dimension)
    pos_res_R = (delta_r * R_hat).sum(dim=1).unsqueeze(1)
    pos_res_A = (delta_r * S_hat).sum(dim=1).unsqueeze(1)
    pos_res_C = (delta_r * W_hat).sum(dim=1).unsqueeze(1)
    
    vel_res_R = (delta_v * R_hat).sum(dim=1).unsqueeze(1)
    vel_res_A = (delta_v * S_hat).sum(dim=1).unsqueeze(1)
    vel_res_C = (delta_v * W_hat).sum(dim=1).unsqueeze(1)

    # Shape: (N, 3) where columns are [Radial, Along-track, Cross-track]
    pos_ric = torch.cat([pos_res_R, pos_res_A, pos_res_C], dim=1)
    vel_ric = torch.cat([vel_res_R, vel_res_A, vel_res_C], dim=1)

    return t_since_mins, pos_ric, vel_ric

def plot_ric_residuals(
    t_mins: torch.Tensor, 
    results_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
):
    """
    Generates a 2x3 subplot grid comparing the RIC residuals of different models.
    
    Args:
        t_mins: Time vector in minutes since epoch (N,)
        results_dict: Dictionary mapping model name to (pos_ric, vel_ric) tensors.
                      e.g., {"Initial TLE": (pos_init, vel_init), "Optimized": (pos_opt, vel_opt)}
    """
    t_plot = t_mins.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    components = ['Radial', 'Along-track', 'Cross-track']
    
    styles = {
        "Initial TLE": {"color": "red", "linestyle": "--", "alpha": 0.7},
        "Optimized": {"color": "green", "linestyle": "-", "alpha": 0.9, "linewidth": 2}
    }

    for name, (pos_ric, vel_ric) in results_dict.items():
        pos_np = pos_ric.detach().cpu().numpy()
        vel_np = vel_ric.detach().cpu().numpy()
        
        # Default style if name not in dict
        style = styles.get(name, {"linestyle": "-", "alpha": 0.8})

        for i in range(3):
            # Position row
            axes[0, i].plot(t_plot, pos_np[:, i], label=name, **style)
            # Velocity row
            axes[1, i].plot(t_plot, vel_np[:, i], label=name, **style)

    # Formatting
    for i, comp in enumerate(components):
        axes[0, i].set_title(f'{comp} Error')
        axes[0, i].set_ylabel('Position Error (km)' if i == 0 else '')
        axes[1, i].set_ylabel('Velocity Error (km/s)' if i == 0 else '')
        axes[1, i].set_xlabel('Time Since Epoch (Minutes)')
        
        for row in range(2):
            axes[row, i].grid(True, linestyle=':', alpha=0.7)
            if i == 0 and row == 0:
                axes[row, i].legend(loc='best')

    plt.suptitle('GPS vs TLE Residuals in RIC Frame', fontsize=16)
    plt.tight_layout()
    plt.show()

# 1. Load Data
t_gps, r_gps, v_gps = load_gmat_csv_block(
    file_path="data/gps_data.csv",
    tle_epoch_unix=epoch,
    block_sec=24*60*60,
)

# 2. Compute Initial Residuals
t_mins, pos_ric_init, vel_ric_init = compute_ric_residuals(
    tle=init_tle, 
    t_gps=t_gps, 
    r_gps=r_gps, 
    v_gps=v_gps, 
    tle_epoch_unix=epoch
)

# 3. Create Dictionary & Plot
results = {
    "Initial TLE": (pos_ric_init, vel_ric_init)
    # "Optimized": (pos_ric_opt, vel_ric_opt)  <-- Add this once you convert state back to TLE
}

plot_ric_residuals(t_mins, results)