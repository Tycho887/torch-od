import time
import torch
from torch.func import jacfwd, grad, hessian
import polars as pl

import dsgp4
from dsgp4.tle import TLE
from astropy.time import Time

# Internal diffod modules
import diffod.state as state
import diffod.functional.system as system
from diffod.utils import unix_to_mjd, load_gmat_csv_block_legacy
from diffod.solvers.gn_svd import svd_solve

def run_benchmarks(pipe_model, base_state, t_full, y_full_obs):
    """
    Benchmarks Autodiff vs Finite Difference for Jacobians, and evaluates
    Gradient and Hessian computation speeds for RMSE and Huber losses.
    """
    print("\n--- Starting Autodiff vs Finite Difference Benchmarks ---")
    
    # 1. Setup Benchmark Data (1000 measurements)
    num_meas = 1000
    t_bench = t_full[:num_meas]
    # Grab 3 * num_meas assuming flattened [x1, y1, z1, x2, y2, z2...]
    y_obs_bench = y_full_obs[:6 * num_meas] 
    
    # Pad the state to 10 parameters to simulate target architecture
    pad_len = 10 - len(base_state)
    if pad_len > 0:
        x_10_init = torch.cat([base_state, torch.zeros(pad_len, dtype=base_state.dtype, device=base_state.device)])
    else:
        x_10_init = base_state[:10]
        
    x_10_init.requires_grad_(False) 

    # 2. Define Target Functions
    def forward_10param(x):
        # Route the first 6 to the propagator, ignore dummies
        return pipe_model(x=x, tsince=t_bench)

    def rmse_loss(x):
        pred = forward_10param(x)
        mse = torch.mean((pred - y_obs_bench)**2)
        return torch.sqrt(mse)

    def huber_loss(x, delta=1.0):
        pred = forward_10param(x)
        err = torch.abs(pred - y_obs_bench)
        quadratic = torch.clamp(err, max=delta)
        linear = err - quadratic
        return torch.mean(0.5 * quadratic**2 + delta * linear)

    @torch.no_grad()
    def fd_jacobian(func, x, eps=1e-5):
        """Central finite difference Jacobian"""
        y_val = func(x)
        J = torch.zeros((len(y_val), len(x)), dtype=x.dtype, device=x.device)
        for i in range(len(x)):
            x_p, x_m = x.clone(), x.clone()
            x_p[i] += eps
            x_m[i] -= eps
            J[:, i] = (func(x_p) - func(x_m)) / (2 * eps)
        return J

    # 3. Timing Utility
    def time_execution(func, *args, iterations=10):
        # Warmup (critical for JIT / PyTorch dispatch overhead)
        for _ in range(2): 
            _ = func(*args)
            
        start = time.perf_counter()
        for _ in range(iterations):
            _ = func(*args)
        end = time.perf_counter()
        return (end - start) / iterations

    # 4. Execute Benchmarks
    print(f"Configuration: {len(x_10_init)} Parameters | {num_meas} Measurements")
    
    # -- Jacobians --
    t_jac_ad = time_execution(jacfwd(forward_10param), x_10_init)
    t_jac_fd = time_execution(fd_jacobian, forward_10param, x_10_init)
    
    # Validate accuracy
    J_ad = jacfwd(forward_10param)(x_10_init)
    J_fd = fd_jacobian(forward_10param, x_10_init)
    max_diff = torch.max(torch.abs(J_ad.detach() - J_fd)).item()

    # -- Gradients --
    t_grad_rmse = time_execution(grad(rmse_loss), x_10_init)
    t_grad_huber = time_execution(grad(huber_loss), x_10_init)

    # -- Hessians --
    t_hess_rmse = time_execution(hessian(rmse_loss), x_10_init)
    t_hess_huber = time_execution(hessian(huber_loss), x_10_init)

    # 5. Report
    print("\n--- Benchmark Results (Average of 10 runs) ---")
    print(f"Jacobian (Autodiff):      {t_jac_ad:.4f} seconds")
    print(f"Jacobian (Finite Diff):   {t_jac_fd:.4f} seconds")
    print(f"  -> Speedup:             {t_jac_fd / t_jac_ad:.1f}x")
    print(f"  -> Max AD/FD Discrepancy: {max_diff:.2e}")
    print("-" * 46)
    print(f"RMSE Loss Gradient:       {t_grad_rmse:.4f} seconds")
    print(f"Huber Loss Gradient:      {t_grad_huber:.4f} seconds")
    print("-" * 46)
    print(f"RMSE Loss Hessian:        {t_hess_rmse:.4f} seconds")
    print(f"Huber Loss Hessian:       {t_hess_huber:.4f} seconds")
    print("----------------------------------------------\n")

def evaluate_jacobian_accuracy(pipe_model, base_state, t_bench, y_obs_bench):
    print("\n--- Evaluating FD vs AD Accuracy ---")
    
    # 1. Setup target function and compute AD Ground Truth
    def forward_fn(x):
        return pipe_model(x=x, tsince=t_bench)
    
    # Pad state as before
    pad_len = 10 - len(base_state)
    x_init = torch.cat([base_state, torch.zeros(pad_len, dtype=base_state.dtype, device=base_state.device)]) if pad_len > 0 else base_state[:10]
    
    J_ad = jacfwd(forward_fn)(x_init).detach()
    
    # 2. Define the epsilon sweep (from 1e-1 down to 1e-12)
    epsilons = torch.logspace(-2, -7, steps=12, dtype=x_init.dtype)
    
    results = []
    machine_eps = torch.finfo(J_ad.dtype).eps
    
    print(f"{'Epsilon':<12} | {'Max Rel Error':<15} | {'Mean Rel Error':<15} | {'Min Cos Sim':<15}")
    print("-" * 65)

    @torch.no_grad()
    def fd_jacobian(x, eps):
        y_val = forward_fn(x)
        J = torch.zeros((len(y_val), len(x)), dtype=x.dtype, device=x.device)
        for i in range(len(x)):
            x_p, x_m = x.clone(), x.clone()
            x_p[i] += eps
            x_m[i] -= eps
            J[:, i] = (forward_fn(x_p) - forward_fn(x_m)) / (2 * eps)
        return J

    # 3. Execute the sweep
    for eps in epsilons:
        eps_val = eps.item()
        J_fd = fd_jacobian(x_init, eps_val)
        
        # Relative Error Calculation
        rel_error_matrix = torch.abs(J_ad - J_fd) / (torch.abs(J_ad) + machine_eps)
        max_rel_err = torch.max(rel_error_matrix).item()
        mean_rel_err = torch.mean(rel_error_matrix).item()
        
        # Cosine Similarity (row-wise across the Jacobian)
        # J_ad and J_fd shape: [num_measurements * 3, num_params]
        cos_sims = torch.nn.functional.cosine_similarity(J_ad, J_fd, dim=1)
        min_cos_sim = torch.min(cos_sims).item()
        
        results.append((eps_val, max_rel_err, mean_rel_err, min_cos_sim))
        
        print(f"{eps_val:<12.1e} | {max_rel_err:<15.2e} | {mean_rel_err:<15.2e} | {min_cos_sim:<15.6f}")

    # Find the "best" epsilon based on Mean Relative Error
    best_eps = min(results, key=lambda r: r[2])[0]
    print("-" * 65)
    print(f"Optimal FD Epsilon found at: {best_eps:.1e}")
    
    return results

import torch
from torch.func import jacfwd

def evaluate_fp32_feasibility(pipe_model, base_state, t_bench, y_obs_bench):
    print("\n--- Evaluating FP32 AD vs FP64 AD vs FP64 FD (eps=1e-7) ---")
    
    # 1. Setup Base Tensors (Ensure FP64 baseline)
    x_64 = base_state.clone().to(torch.float64).requires_grad_(False)
    t_64 = t_bench.clone().to(torch.float64)
    
    # Setup FP32 Tensors
    x_32 = base_state.clone().to(torch.float32).requires_grad_(False)
    t_32 = t_bench.clone().to(torch.float32)
    
    # 2. Define Forward Functions for both precisions
    def forward_64(x):
        # Assumes pipe_model can handle dtype changes dynamically based on input 'x' and 'tsince'
        return pipe_model(x=x, tsince=t_64)

    def forward_32(x):
        return pipe_model(x=x, tsince=t_32)

    # 3. Compute FP64 Autodiff Ground Truth
    print("Computing FP64 Autodiff Jacobian...")
    J_ad_64 = jacfwd(forward_64)(x_64).detach()
    machine_eps_64 = torch.finfo(torch.float64).eps

    # 4. Compute FP32 Autodiff Jacobian
    print("Computing FP32 Autodiff Jacobian...")
    J_ad_32_raw = jacfwd(forward_32)(x_32).detach()
    # Cast back to FP64 to calculate the numerical discrepancies
    J_ad_32_cast = J_ad_32_raw.to(torch.float64)

    # 5. Compute FP64 Finite Difference (eps = 1e-7)
    print("Computing FP64 Finite Difference Jacobian (eps=1e-7)...")
    @torch.no_grad()
    def fd_jacobian_64(func, x, eps=1e-7):
        y_val = func(x)
        J = torch.zeros((len(y_val), len(x)), dtype=torch.float64, device=x.device)
        for i in range(len(x)):
            x_p, x_m = x.clone(), x.clone()
            x_p[i] += eps
            x_m[i] -= eps
            J[:, i] = (func(x_p) - func(x_m)) / (2 * eps)
        return J
        
    J_fd_64 = fd_jacobian_64(forward_64, x_64, eps=1e-7)

    # 6. Evaluation Metrics Helper
    def compute_metrics(J_target, J_truth, eps_val):
        abs_diff = torch.abs(J_truth - J_target)
        rel_error = abs_diff / (torch.abs(J_truth) + eps_val)
        
        max_abs = torch.max(abs_diff).item()
        mean_rel = torch.mean(rel_error).item()
        
        # Flattened cosine similarity for overall directionality
        cos_sim = torch.nn.functional.cosine_similarity(J_truth.flatten(), J_target.flatten(), dim=0).item()
        
        # Worst row-wise cosine similarity (identifies if specific state parameters gradients are destroyed)
        row_cos_sims = torch.nn.functional.cosine_similarity(J_truth, J_target, dim=1)
        min_row_cos_sim = torch.min(row_cos_sims).item()
        
        return max_abs, mean_rel, cos_sim, min_row_cos_sim

    # Calculate metrics
    metrics_32 = compute_metrics(J_ad_32_cast, J_ad_64, machine_eps_64)
    metrics_fd = compute_metrics(J_fd_64, J_ad_64, machine_eps_64)

    # 7. Report Results
    print("\n--- Accuracy Comparison against FP64 Autodiff ---")
    print(f"{'Metric':<25} | {'FP32 Autodiff':<20} | {'FP64 FD (eps=1e-7)':<20}")
    print("-" * 70)
    print(f"{'Max Absolute Error':<25} | {metrics_32[0]:<20.4e} | {metrics_fd[0]:<20.4e}")
    print(f"{'Mean Relative Error':<25} | {metrics_32[1]:<20.4e} | {metrics_fd[1]:<20.4e}")
    print(f"{'Global Cosine Sim':<25} | {metrics_32[2]:<20.6f} | {metrics_fd[2]:<20.6f}")
    print(f"{'Worst Row Cosine Sim':<25} | {metrics_32[3]:<20.6f} | {metrics_fd[3]:<20.6f}")
    print("-" * 70)
    
    return J_ad_64, J_ad_32_cast, J_fd_64

def benchmark_fp32_vs_fd_against_fp64(pipe_model, base_state, t_bench):
    print("\n--- Benchmarking FP32-AD and FP64-FD against FP64-AD Ground Truth ---")
    
    # 1. Setup Data with Explicit Dtypes
    x_64 = base_state.clone().to(torch.float64).requires_grad_(False)
    t_64 = t_bench.clone().to(torch.float64)
    
    x_32 = base_state.clone().to(torch.float32).requires_grad_(False)
    t_32 = t_bench.clone().to(torch.float32)

    # compiled_model = torch.jit.script(pipe_model)

    # 2. Define Precision-Specific Forward Passes
    def forward_64(x,tsince=t_64):
        return pipe_model(x=x, tsince=tsince)

    def forward_32(x,tsince=t_32):
        return pipe_model(x=x, tsince=tsince)

    @torch.no_grad()
    def fd_jacobian_64(x, eps=1e-3, tsince=t_64):
        y_val = forward_64(x)
        J = torch.zeros((len(y_val), len(x)), dtype=torch.float64, device=x.device)
        for i in range(len(x)):
            x_p, x_m = x.clone(), x.clone()
            x_p[i] += eps
            x_m[i] -= eps
            J[:, i] = (forward_64(x_p, tsince) - forward_64(x_m, tsince)) / (2 * eps)
        return J

    # compiled_f64 = torch.jit.script(forward_64)
    # compiled_f32 = torch.jit.script(forward_32)
    # compiled_fd = torch.jit.script(fd_jacobian_64)

    # 3. Timing Utility
    def time_execution(func, *args, iterations=10):
        # Warmup
        for _ in range(2): 
            _ = func(*args)
        torch.cuda.synchronize() if x_64.is_cuda else None
            
        start = time.perf_counter()
        for _ in range(iterations):
            _ = func(*args)
        torch.cuda.synchronize() if x_64.is_cuda else None
        end = time.perf_counter()
        return (end - start) / iterations

    # 4. Measure Execution Times
    print("Running execution time benchmarks (10 iterations)...")
    t_ad_64 = time_execution(jacfwd(forward_64), x_64)
    t_ad_32 = time_execution(jacfwd(forward_32), x_32)
    t_fd_64 = time_execution(fd_jacobian_64, x_64)

    # 5. Compute Jacobians for Accuracy Evaluation
    print("Computing Jacobians for accuracy evaluation...")
    J_ad_64 = jacfwd(forward_64)(x_64).detach()
    J_ad_32_raw = jacfwd(forward_32)(x_32).detach()
    
    # Cast FP32 result to FP64 for mathematical comparison
    J_ad_32 = J_ad_32_raw.to(torch.float64)
    J_fd_64 = fd_jacobian_64(x_64, eps=1e-7).detach()

    # 6. Evaluation Metrics Helper
    def compute_metrics(J_target, J_truth):
        machine_eps = torch.finfo(torch.float64).eps
        abs_diff = torch.abs(J_truth - J_target)
        rel_error = abs_diff / (torch.abs(J_truth) + machine_eps)
        
        max_abs = torch.max(abs_diff).item()
        mean_rel = torch.mean(rel_error).item()
        cos_sim = torch.nn.functional.cosine_similarity(J_truth.flatten(), J_target.flatten(), dim=0).item()
        min_row_cos_sim = torch.min(torch.nn.functional.cosine_similarity(J_truth, J_target, dim=1)).item()
        
        return max_abs, mean_rel, cos_sim, min_row_cos_sim

    metrics_32 = compute_metrics(J_ad_32, J_ad_64)
    metrics_fd = compute_metrics(J_fd_64, J_ad_64)

    # 7. Report Results
    print("\n" + "="*80)
    print(f"{'BENCHMARK RESULTS (vs FP64-AD Ground Truth)':^80}")
    print("="*80)
    
    print("\n--- Execution Time ---")
    print(f"{'Method':<25} | {'Time (s)':<15} | {'Relative Speed'}")
    print("-" * 60)
    print(f"{'FP64 Autodiff (Baseline)':<25} | {t_ad_64:<15.4f} | 1.00x")
    print(f"{'FP32 Autodiff':<25} | {t_ad_32:<15.4f} | {t_ad_64 / t_ad_32:<4.2f}x faster")
    print(f"{'FP64 Finite Diff (1e-7)':<25} | {t_fd_64:<15.4f} | {t_fd_64 / t_ad_64:<4.2f}x slower")

    print("\n--- Accuracy Degradation ---")
    print(f"{'Metric':<25} | {'FP32-AD Error':<20} | {'FP64-FD Error':<20}")
    print("-" * 70)
    print(f"{'Max Absolute Error':<25} | {metrics_32[0]:<20.4e} | {metrics_fd[0]:<20.4e}")
    print(f"{'Mean Relative Error':<25} | {metrics_32[1]:<20.4e} | {metrics_fd[1]:<20.4e}")
    print(f"{'Global Cosine Sim':<25} | {metrics_32[2]:<20.6f} | {metrics_fd[2]:<20.6f}")
    print(f"{'Worst Row Cosine Sim':<25} | {metrics_32[3]:<20.6f} | {metrics_fd[3]:<20.6f}")
    print("="*80 + "\n")

    return J_ad_64, J_ad_32, J_fd_64

# ---------------------------------------------------------
# Configuration & Data Loading
# ---------------------------------------------------------
device = torch.device(device="cpu")
dtype = torch.float32
center_freq = 1707.0 

known_global_time_bias_sec = 0.277

TLE_list = [
    "AWS",
    "1 60543U 24149CD  25307.42878472  .00000000  00000-0  11979-3 0    11",
    "2 60543  97.7067  19.0341 0003458 123.1215 316.4897 14.89807169 65809",
]
tle_base = TLE(data=TLE_list)
epoch_unix = 1762207191

print("Loading Synthetic Dataset & GPS Truth...")

# Load GPS Truth
t_gps, r_gps, v_gps = load_gmat_csv_block_legacy(
    file_path="data/AWS_full_long_period.csv", 
    tle_epoch_unix=epoch_unix,
    block_sec=86400 * 2
)

# Load Synthetic Doppler Telemetry
synthetic_telemetry = pl.read_parquet("data/synthetic_period_telemetry.parquet")

T_mean = float(torch.mean(t_gps))
print(f"Central Epoch (T_mean): {T_mean}")

t_ref_astropy = Time(T_mean, format="unix", scale="utc")

# ---------------------------------------------------------
# PHASE 1: GPS-Based Orbit Determination
# ---------------------------------------------------------
print("\n--- Phase 1: Fitting TLE to GPS ---")
init_tle_gps, _ = dsgp4.newton_method(tle_base, unix_to_mjd(T_mean))

ssv_gps = state.MEE_SSV(init_tle=init_tle_gps, num_measurements=len(t_gps), fit_bstar=False, dtype=dtype)
prop_gps = system.SGP4(ssv=ssv_gps, dtype=dtype)
meas_gps = system.CartesianMeasurement(ssv=ssv_gps)
pipe_gps = system.MeasurementPipeline(propagator=prop_gps, measurement_model=meas_gps)

# compiled_prop = torch.jit.script(prop_gps)
compiled_meas = torch.jit.script(meas_gps)

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

run_benchmarks(pipe_gps, x_gps_out, t_since_gps, y_gps_1d)
evaluate_jacobian_accuracy(pipe_gps, x_gps_out, t_since_gps, y_gps_1d)
benchmark_fp32_vs_fd_against_fp64(pipe_gps, x_gps_out, t_since_gps)