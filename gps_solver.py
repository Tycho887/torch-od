import time
from astropy.units import F
import dsgp4
import numpy as np
import torch
import matplotlib.pyplot as plt
from dsgp4.tle import TLE
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import load_gmat_csv_block

# Solvers
from diffod.solvers.gaussNewton import wgn_solve
from diffod.solvers.newton import newton_solve
from diffod.solvers.cca import cca_solve
from diffod.solvers.lbfgs import lbfgs_solve

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
target_device = torch.device("cpu")

print("Loading GPS Data...")
# Load real GPS telemetry data at the beginning of the script
t_gps_raw, r_gps_raw, v_gps_raw = load_gmat_csv_block(
    file_path="data/gps_data.csv",
    tle_epoch_unix=epoch,
    block_sec=60*60*24, # 1 hour block
)

r_gps_raw /= 1e3
v_gps_raw /= 1e3


# Apply leap second offset for SGP4 propagation
t_gps = t_gps_raw.to(target_device) #- 37.0
r_gps = r_gps_raw.to(target_device) #/ 1e3
v_gps = v_gps_raw.to(target_device) #/ 1e3

N_samples = len(t_gps)
t_since_mins = (t_gps - epoch) / 60.0

# ---------------------------------------------------------
# 2. Define State Vector & Functional Forward
# ---------------------------------------------------------
ssv = state.SSV(
    init_tle=init_tle,
    num_measurements=N_samples,
    fit_ma=True,
    fit_mean_motion=True,
    fit_argp=True,
    fit_bstar=True,
    fit_inclination=True,
    fit_eccentricity=True,
    fit_raan=True,
)

# --- CARTESIAN MODULAR PIPELINE ---
propagator = system.SGP4(ssv=ssv, use_pretrained_model=False)
measurement_model = system.CartesianMeasurement(ssv=ssv)

# Assuming you named the wrapper PropagatedCartesian
model = system.MeasurementPipeline(
    propagator=propagator, 
    measurement_model=measurement_model
)

# Format the ground truth GPS data into the 1D (6N,) observation vector
gps_obs_1d = measurement_model.format_gps_observations(r_gps=r_gps, v_gps=v_gps)

def functional_forward(x) -> torch.Tensor:
    # No station pos/vel or center_freq needed for Cartesian
    return model(x=x, tsince=t_since_mins)

# ---------------------------------------------------------
# 3. Generate Perturbed Starts for Monte Carlo
# ---------------------------------------------------------
x_true = ssv.get_initial_state(device=target_device)

N_solves = 1
x_init_batch = x_true.unsqueeze(dim=0).repeat(N_solves, 1)
x_init_batch = x_init_batch + x_init_batch * torch.randn_like(input=x_init_batch) * 1e-4
x_in = x_init_batch[0].type(dtype=torch.float64)

# ---------------------------------------------------------
# 4. WGN Matrix Configurations
# ---------------------------------------------------------
sigma_obs = 1.0  # Normalized observations, standard deviation is relative
n_total = x_true.shape[0]

active_map = ssv.get_active_map(device=target_device)
consider_map = ~active_map

n_active = int(active_map.sum().item())
n_consider = n_total - n_active

print(f"Total Params: {n_total} | Estimated: {n_active} | Considered: {n_consider}")

P_x_inv = torch.eye(n=n_active, dtype=torch.float64, device=target_device)
P_cc = torch.eye(n=n_consider, dtype=torch.float64, device=target_device)

# ---------------------------------------------------------
# 5. Execute Sequential Iterations & Benchmark
# ---------------------------------------------------------
print("\nExecuting OD Solvers...")
results = {}

solvers = {
    # "Gauss-Newton (WGN)": wgn_solve,
    "L-BFGS": lbfgs_solve,
    # "Consider Covariance (CCA)": cca_solve,
}

with torch.no_grad():
    for name, solver_fn in solvers.items():
        t0 = time.perf_counter()
        
        kwargs = {
            "x_init": x_in,
            "y_obs_fixed": gps_obs_1d, # Pass the 1D Cartesian array
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
# 6. Plotting Results (RIC Residuals)
# ---------------------------------------------------------
print("\nGenerating RIC Residual Plots...")

def compute_ric_residuals(
    tle, 
    t_gps: torch.Tensor, 
    r_gps: torch.Tensor, 
    v_gps: torch.Tensor, 
    tle_epoch_unix: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. Convert GPS inputs from meters back to kilometers for SGP4 math
    r_gps_km = r_gps.to(torch.float64) #/ 1000.0
    v_gps_km = v_gps.to(torch.float64) #/ 1000.0
    t_gps = t_gps.to(torch.float64)

    # 2. Propagate TLE to GPS timestamps
    t_since_mins = (t_gps - tle_epoch_unix) / 60.0
    
    # dsgp4 returns shape (N, 2, 3) where [:, 0] is pos, [:, 1] is vel
    r_tot = dsgp4.propagate(tle=tle, tsinces=t_since_mins, initialized=False)
    r_calc_km = r_tot[:, 0]
    v_calc_km = r_tot[:, 1]

    # 3. Calculate Cartesian Errors (Calculated - Truth)
    delta_r = r_calc_km - r_gps_km
    delta_v = v_calc_km - v_gps_km

    # 4. Construct RIC Basis Vectors (using GPS state as the reference)
    R_hat = r_gps_km / torch.norm(r_gps_km, dim=1, keepdim=True)
    W_vec = torch.cross(r_gps_km, v_gps_km, dim=1)
    W_hat = W_vec / torch.norm(W_vec, dim=1, keepdim=True)
    S_hat = torch.cross(W_hat, R_hat, dim=1)

    # 5. Project Cartesian errors into the RIC frame
    pos_res_R = (delta_r * R_hat).sum(dim=1).unsqueeze(1)
    pos_res_A = (delta_r * S_hat).sum(dim=1).unsqueeze(1)
    pos_res_C = (delta_r * W_hat).sum(dim=1).unsqueeze(1)
    
    vel_res_R = (delta_v * R_hat).sum(dim=1).unsqueeze(1)
    vel_res_A = (delta_v * S_hat).sum(dim=1).unsqueeze(1)
    vel_res_C = (delta_v * W_hat).sum(dim=1).unsqueeze(1)

    pos_ric = torch.cat([pos_res_R, pos_res_A, pos_res_C], dim=1)
    vel_ric = torch.cat([vel_res_R, vel_res_A, vel_res_C], dim=1)

    return t_since_mins, pos_ric, vel_ric

def print_ric_residual_summary(results_dict: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """
    Prints a formatted summary of RMS errors in the RIC frame for each solver result.
    """
    components = ['Radial', 'Along-track', 'Cross-track']
    header = f"{'Solver Name':<25} | {'Axis':<12} | {'Pos RMS (km)':<14} | {'Vel RMS (km/s)':<14}"
    divider = "-" * len(header)

    print("\n" + divider)
    print(header)
    print(divider)

    for name, (pos_ric, vel_ric) in results_dict.items():
        # Calculate RMS: sqrt(mean(x^2))
        pos_rms = torch.sqrt(torch.mean(pos_ric**2, dim=0))
        vel_rms = torch.sqrt(torch.mean(vel_ric**2, dim=0))

        for i, comp in enumerate(components):
            # Only print the name on the first row of each solver block for readability
            row_name = name if i == 0 else ""
            print(f"{row_name:<25} | {comp:<12} | {pos_rms[i]:<14.6f} | {vel_rms[i]:<14.6f}")
        print(divider)

# Usage in your script:
def plot_ric_residuals(
    t_mins: torch.Tensor, 
    results_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
):
    t_plot = t_mins.detach().cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    components = ['Radial', 'Along-track', 'Cross-track']
    
    styles = {
        "Initial TLE": {"color": "red", "linestyle": "--", "alpha": 0.7, "linewidth": 2},
        "Gauss-Newton (WGN)": {"color": "blue", "linestyle": "-", "alpha": 0.8, "linewidth": 2},
        "Exact Newton": {"color": "purple", "linestyle": ":", "alpha": 0.8, "linewidth": 2.5},
        "Consider Covariance (CCA)": {"color": "green", "linestyle": "-.", "alpha": 1.0, "linewidth": 2},
    }

    for name, (pos_ric, vel_ric) in results_dict.items():
        pos_np = pos_ric.detach().cpu().numpy()
        vel_np = vel_ric.detach().cpu().numpy()
        style = styles.get(name, {"linestyle": "-", "alpha": 0.8})

        for i in range(3):
            axes[0, i].plot(t_plot, pos_np[:, i], label=name, **style)
            axes[1, i].plot(t_plot, vel_np[:, i], label=name, **style)

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

# Calculate initial residuals
t_mins, pos_ric_init, vel_ric_init = compute_ric_residuals(
    tle=init_tle, 
    t_gps=t_gps_raw, 
    r_gps=r_gps_raw, 
    v_gps=v_gps_raw, 
    tle_epoch_unix=epoch
)

ric_results = {"Initial TLE": (pos_ric_init, vel_ric_init)}

print(f"Initial TLE: {init_tle}")

prior = dsgp4.propagate(tle=init_tle, tsinces=0, initialized=False)

# Calculate residuals for each optimized state
for name, res in results.items():
    opt_tle = ssv.export(res["x"])
    new = dsgp4.propagate(tle=opt_tle, tsinces=0, initialized=False)
    print(f"Difference is: {new-prior}")
    print(f"Optimized TLE: {opt_tle}")
    _, pos_ric_opt, vel_ric_opt = compute_ric_residuals(
        tle=opt_tle, 
        t_gps=t_gps_raw, 
        r_gps=r_gps_raw, 
        v_gps=v_gps_raw, 
        tle_epoch_unix=epoch
    )
    ric_results[name] = (pos_ric_opt, vel_ric_opt)

print_ric_residual_summary(ric_results)
plot_ric_residuals(t_mins, ric_results)
