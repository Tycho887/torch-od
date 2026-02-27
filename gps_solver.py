import time
import dsgp4
import torch
from dsgp4.tle import TLE
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import load_gmat_csv_block, unix_to_mjd
from diffod.visualize import compute_ric_residuals, print_ric_residual_summary, plot_ric_residuals
# Solvers
from diffod.solvers.gaussNewton import wgn_solve
from diffod.solvers.newton import newton_solve
from diffod.solvers.cca import cca_solve
from diffod.solvers.lbfgs import lbfgs_solve
from diffod.solvers.gn_svd import svd_solve

# ---------------------------------------------------------
# 1. Setup Data & Boundary
# ---------------------------------------------------------
TLE_list = [
    "AWS",
    "1 60543U 24149CD  25307.42878472  .00000000  00000-0  11979-3 0    11",
    "2 60543  97.7067  19.0341 0003458 123.1215 316.4897 14.89807169 65809",
]

dtype = torch.float64

epoch = 1762508742#1762508742
tle0_base = TLE(data=TLE_list)

target_device = torch.device("cpu")

print("Loading GPS Data...")
# Load real GPS telemetry data at the beginning of the script
t_gps_raw, r_gps_raw, v_gps_raw = load_gmat_csv_block(
    file_path="data/AWS_long_period.csv",
    tle_epoch_unix=epoch,
    block_sec=5*86400#43200, # 1 hour block
)

epoch = float(torch.mean(input=t_gps_raw))

print(f"The new epoch is: {epoch}")

init_tle, _ = dsgp4.newton_method(tle0_base, unix_to_mjd(unix_seconds=epoch))    

# Apply leap second offset for SGP4 propagation
t_gps = t_gps_raw.to(device=target_device, dtype=dtype) #- 37.0
r_gps = r_gps_raw.to(device=target_device, dtype=dtype) #/ 1e3
v_gps = v_gps_raw.to(device=target_device, dtype=dtype) #/ 1e3

N_samples = len(t_gps)
t_since_mins = (t_gps - epoch) / 60.0

ssv = state.MEE_SSV(
    init_tle=init_tle,
    num_measurements=N_samples,
    fit_mean_motion=True,  # n
    fit_f=True,            # e * cos(omega + raan)
    fit_g=True,            # e * sin(omega + raan)
    fit_h=True,            # tan(i/2) * cos(raan)
    fit_k=True,            # tan(i/2) * sin(raan)
    fit_L=True,            # raan + omega + M
    fit_bstar=True,
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
x_in = x_init_batch[0].type(dtype=dtype)

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

P_x_inv = torch.eye(n=n_active, dtype=dtype, device=target_device)
P_cc = torch.eye(n=n_consider, dtype=dtype, device=target_device)

# ---------------------------------------------------------
# 5. Execute Sequential Iterations & Benchmark
# ---------------------------------------------------------
print("\nExecuting OD Solvers...")
results = {}

solvers = {
    "Gauss-Newton (WGN)": wgn_solve,
    "L-BFGS": lbfgs_solve,
    "GN-SVD": svd_solve,
    # "Newton": newton_solve,

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

# Calculate initial residuals using the initial tensor state
t_mins, pos_ric_init, vel_ric_init = compute_ric_residuals(
    x_state=x_in,
    propagator=propagator, 
    t_gps=t_gps_raw, 
    r_gps=r_gps_raw, 
    v_gps=v_gps_raw, 
    tle_epoch_unix=epoch
)

ric_results = {}#{"Initial State": (pos_ric_init, vel_ric_init)}

print(f"Initial TLE:\n{init_tle}\n")

# Calculate residuals for each optimized state
for name, res in results.items():
    # Keep the export for human-readable logging
    opt_tle = ssv.export(res["x"])
    print(f"Optimized TLE ({name}):\n{opt_tle}\n")
    
    # Compute residuals precisely with the PyTorch model
    _, pos_ric_opt, vel_ric_opt = compute_ric_residuals(
        x_state=res["x"],
        propagator=propagator,
        t_gps=t_gps_raw, 
        r_gps=r_gps_raw, 
        v_gps=v_gps_raw, 
        tle_epoch_unix=epoch
    )
    # if name == "Initial State": 
    #     continue
    print(name)
    ric_results[name] = (pos_ric_opt, vel_ric_opt)

print_ric_residual_summary(ric_results)
plot_ric_residuals(t_mins, ric_results)