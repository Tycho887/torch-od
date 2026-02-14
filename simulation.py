import time

import numpy as np
import torch
from dsgp4.tle import TLE
from torch.func import jacfwd

import diffod.state as state
from diffod.functional.models import DopplerResiduals
from diffod.gse import station_teme_preprocessor
from diffod.solvers.cca import compute_cca_step

# ---------------------------------------------------------
# 1. Setup Data & GPU Boundary
# ---------------------------------------------------------
TLE_list = [
    "ISS (ZARYA)",
    "1 25544U 98067A   26038.50283897  .00012054  00000-0  23050-3 0  9996",
    "2 25544  51.6315 221.5822 0011000  74.6214 285.5989 15.48462076551652",
]
init_tle = TLE(data=TLE_list)
stations = {
    0: np.array(object=[-69.6, 18.9, 0.1]),
    1: np.array(object=[78.2, 15.4, 0.4]),
}

target_device = torch.device("cuda")

# Dialing back to 10k for out-of-the-box memory safety with 100 parallel batch solves.
# Scale this up depending on your VRAM.
N_samples = 4000
t_obs_cpu = torch.linspace(0, 20000, N_samples, dtype=torch.float64, device="cpu")

st_indices = torch.zeros(N_samples, dtype=torch.int32)
st_indices[N_samples // 2 :] = 1

pass_indices = torch.zeros(N_samples, dtype=torch.int32)
pass_indices[N_samples // 3 :] = 1
pass_indices[N_samples // 4 :] = 2
pass_indices[N_samples // 5 :] = 3

print("Preprocessing Astropy arrays on CPU...")
station_pos_cpu, station_vel_cpu = station_teme_preprocessor(
    times_s=t_obs_cpu.numpy() / 60 + 1739535000,
    station_ids=st_indices.numpy(),
    id_to_station=stations,
    dtype=torch.float64,
    device=torch.device("cpu"),
)

# Cross the boundary: Move to GPU & enforce FP32
st_pos = station_pos_cpu.to(device=target_device)
st_vel = station_vel_cpu.to(device=target_device)
t_obs_gpu = t_obs_cpu.to(device=target_device)

# ---------------------------------------------------------
# 2. Define State Vector & Functional Forward
# ---------------------------------------------------------
state_def = state.StateDefinition(
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
state_def.add_linear_bias(name="doppler_bias", group_indices=pass_indices)

model = DopplerResiduals(
    state_def=state_def, bias_group=state_def.get_bias_group(name="doppler_bias")
)


def functional_forward(x) -> torch.Tensor:
    # Computes Doppler shift predictions functionally
    return model(x=x, tsince=t_obs_gpu, st_pos=st_pos, st_vel=st_vel, center_freq=2.2e9)


# ---------------------------------------------------------
# 3. Generate Realistic "Ground Truth" and Perturbed Starts
# ---------------------------------------------------------
x_true = state_def.get_initial_state(device=target_device)

with torch.no_grad():
    y_true = functional_forward(x_true.to(torch.float64))
    # Add ~10Hz Gaussian noise to simulate realistic observations
    y_obs = y_true + torch.randn_like(y_true) * 10.0

N_solves = 10
# Duplicate ground truth 100 times, then add a small perturbation to create initial guesses
x_init_batch = x_true.unsqueeze(0).repeat(N_solves, 1)
x_init_batch = x_init_batch + x_init_batch * torch.randn_like(x_init_batch) * 1e-4


# ---------------------------------------------------------
# 4. Define the Native GPU Solver with LU & VMAP
# ---------------------------------------------------------
scale_x = torch.tensor(
    [1e2, 1e3, 1.0, 0.3, 0.25, 1e4, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=torch.float64,
    device=target_device,
)
# scale_x_inv = 1.0 / scale_x

# Define the observation scale (N_obs,)
# Typically 1.0 / observation_noise_std
# For 10Hz noise:
scale_y = torch.full((N_samples,), 0.1, dtype=torch.float64, device=target_device)


def single_gn_step_scaled(x_single_fp64, y_obs_fixed):
    """Executes a single, well-conditioned GN step mapped over a batch of states."""
    x_fp32 = x_single_fp64.to(torch.float32)

    # 1. Forward Pass & Jacobian (FP32)
    y_pred_fp32 = functional_forward(x_fp32)
    H_fp32 = jacfwd(func=functional_forward)(x_fp32)

    # 2. Cast to FP64
    H_fp64 = H_fp32.to(torch.float64)
    dy_fp64 = (y_obs_fixed - y_pred_fp32).to(torch.float64)

    # 3. Apply Scaling Transformations using Broadcasting
    # scale_x.unsqueeze(0) broadcasts across columns (parameters)
    # scale_y.unsqueeze(1) broadcasts across rows (observations)
    H_scaled = H_fp64 * scale_x.unsqueeze(0) * scale_y.unsqueeze(1)
    dy_scaled = dy_fp64 * scale_y

    # 4. Form Well-Conditioned Normal Equations
    H_T = H_scaled.T
    A_scaled = H_T @ H_scaled
    b_scaled = H_T @ dy_scaled

    # 5. Levenberg-Marquardt Damping (Now much more effective in scaled space)
    damping = (
        torch.eye(A_scaled.shape[0], dtype=torch.float64, device=target_device) * 1e-6
    )
    A_damped = A_scaled + damping

    # 6. Solve for the scaled update (du)
    du = torch.linalg.solve(A_damped, b_scaled)

    # 7. Rescale update back to physical units (dx)
    dx = du * scale_x

    return x_single_fp64 + dx


# Re-vectorize the solver
batched_gn_step = torch.vmap(single_gn_step_scaled, in_dims=(0, None))


def run_solver_loop(x_batch, y_obs_fixed, num_steps=5):
    """Executes the full batched iterative solver loop."""
    x = x_batch
    for _ in range(num_steps):
        x = batched_gn_step(x, y_obs_fixed)
    return x


# ---------------------------------------------------------
# 5. Compile the Solver Loop (Replaces Manual CUDA Graphs)
# ---------------------------------------------------------
print("Compiling solver to CUDA Graphs (this takes a moment)...")
# 'reduce-overhead' tells PyTorch to automatically use CUDA graphs internally,
# intelligently handling the cuSOLVER workspace allocations that break manual graphs.
fast_solver = torch.compile(run_solver_loop, mode="reduce-overhead")

# Warmup run to trigger compilation
_ = fast_solver(x_init_batch, y_obs)

# ---------------------------------------------------------
# 6. Execute Iterations & Benchmark
# ---------------------------------------------------------
print("Executing fast batched solve...")

# Synchronize before starting the timer to ensure accurate measurement
torch.cuda.synchronize()
t0 = time.time()

# This single line executes 100 independent GN solvers (5 steps each)
final_x_batch = fast_solver(x_init_batch, y_obs)

torch.cuda.synchronize()
t1 = time.time()

print(f"Time taken for {N_solves} simultaneous solves: {(t1 - t0) * 1000:.2f} ms")

# Validate convergence by comparing the first solve to ground truth
print("\nConvergence Check (First Solve vs Ground Truth):")
print(f"Ground Truth: {x_true[:7].detach().cpu().numpy()}")
print(f"Final Output: {final_x_batch[0, :7].detach().cpu().numpy()}")
print(f"Mean Output:  {final_x_batch.mean(dim=0)[:7].detach().cpu().numpy()}")
