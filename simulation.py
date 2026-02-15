import re
import time

import numpy as np
import torch
from dsgp4.tle import TLE
from torch.func import jacfwd

import diffod.state as state
from diffod.functional.models import DopplerResiduals
from diffod.gse import station_teme_preprocessor
from diffod.solvers.cca import cca_solve_single

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
# 4. CCA Matrix Configurations
# ---------------------------------------------------------
sigma_obs = 10.0  # 10Hz noise standard deviation
n_total = x_true.shape[0]

# Define which parameters are estimated (True) vs considered (False)
# Example: Let's consider the last two parameters (e.g., drag or specific biases)
consider_map = torch.tensor(
    [True, True, True, True, False, False, False, True, True, True, True],
    dtype=torch.bool,
    device=target_device,
)

n_est = consider_map.sum().item()
n_cons = n_total - n_est

# Define Priors
# P_cc: Uncertainty of the parameters we are NOT solving for
P_cc = torch.eye(n_cons, dtype=torch.float64, device=target_device) * 1e-4
# P_x_inv: Information matrix (inverse covariance) anchoring the estimated state
P_x_inv = torch.eye(n_est, dtype=torch.float64, device=target_device) * 1e-6


# ---------------------------------------------------------
# 5. Define VMAP Solver
# ---------------------------------------------------------
def residual_fn(state):
    return functional_forward(state)


# Vmap the single solver across the batch dimension (dim=0) for states,
# while keeping priors and observations broadcasted/fixed (None)
batched_cca_solve = torch.vmap(
    lambda x, y: cca_solve_single(
        x_init=x,
        y_obs_fixed=y,
        residual_fn=residual_fn,
        sigma_obs=sigma_obs,
        consider_map=consider_map,
        P_cc=P_cc,
        P_x_inv=P_x_inv,
        num_steps=5,
    ),
    in_dims=(0, None),
)

print("Compiling solver to CUDA Graphs (this takes a moment)...")
fast_solver = torch.compile(batched_cca_solve, mode="reduce-overhead")

# Warmup run to trigger compilation
_ = fast_solver(x_init_batch, y_obs)

# ---------------------------------------------------------
# 6. Execute Iterations & Benchmark
# ---------------------------------------------------------
print("Executing fast batched CCA solve...")
torch.cuda.synchronize()
t0 = time.time()

# Returns the batched states and batched Consider Covariances
final_x_batch, final_P_batch = fast_solver(x_init_batch, y_obs)

torch.cuda.synchronize()
t1 = time.time()

print(f"Time taken for {N_solves} simultaneous solves: {(t1 - t0) * 1000:.2f} ms")

print("\nConvergence Check (First Solve vs Ground Truth):")
print(f"Ground Truth: {x_true[:7].detach().cpu().numpy()}")
print(f"Final Output: {final_x_batch[0, :7].detach().cpu().numpy()}")
print(f"Mean Output:  {final_x_batch.mean(dim=0)[:7].detach().cpu().numpy()}")
print(f"Final Consider Covariance: {final_P_batch[0, :7, :7].detach().cpu().numpy()}")

print(
    f"Final Consider Covariance Trace: {torch.trace(final_P_batch[0, :7, :7]).detach().cpu().numpy()}"
)
