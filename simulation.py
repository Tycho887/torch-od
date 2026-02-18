from doctest import DocTestParser
import re
import time

import numpy as np
import torch
from dsgp4.tle import TLE
import polars as pl

import diffod.state as state
from diffod.functional.system import PredictDoppler
from diffod.gse import station_teme_preprocessor
from diffod.solvers.cca import cca_solve_single

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
    0: np.array(object=[78.228874, 15.376932, 463]),
}

target_device = torch.device("cpu")

# Load real telemetry data
period_telemetry = pl.read_parquet(source="data/period_telemetry.parquet")

times_unix = torch.tensor(period_telemetry["timestamp"].to_numpy(), dtype=torch.float64, device=target_device)
doppler_obs = torch.tensor(period_telemetry["Doppler_Hz"].to_numpy(), dtype=torch.float64, device=target_device)
# Cast to int32 for compatibility with diffod group indexing
contacts = torch.tensor(period_telemetry["contact_index"].to_numpy(), dtype=torch.int32, device=target_device)

N_samples = len(times_unix)
# Calculate relative time in seconds from epoch for the preprocessor
t_obs = times_unix - epoch

st_indices = torch.zeros(N_samples, dtype=torch.int32, device=target_device)

print("Preprocessing Astropy arrays on CPU...")
station_pos_cpu, station_vel_cpu = station_teme_preprocessor(
    times_s=times_unix.numpy(),# / 60 + epoch,
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
    fit_argp=True,
    fit_bstar=True,
    fit_inclination=True,
    fit_eccentricity=True,
    fit_raan=True,
)
state_def.add_linear_bias(name="doppler_bias", group_indices=contacts)

model = PredictDoppler(
    state_def=state_def, bias_group=state_def.get_bias_group(name="doppler_bias")
)

def functional_forward(x) -> torch.Tensor:
    return model(x=x, tsince=t_obs, st_pos=st_pos, st_vel=st_vel, center_freq=2.2e9)

# ---------------------------------------------------------
# 3. Generate Perturbed Starts for Monte Carlo
# ---------------------------------------------------------
x_true = state_def.get_initial_state(device=target_device)

N_solves = 1
# Create Monte Carlo initial guesses around the central state
x_init_batch = x_true.unsqueeze(0).repeat(N_solves, 1)
x_init_batch = x_init_batch + x_init_batch * torch.randn_like(x_init_batch) * 1e-4

# ---------------------------------------------------------
# 4. CCA Matrix Configurations
# ---------------------------------------------------------
sigma_obs = 10.0  
n_total = x_true.shape[0]

consider_params = ["b_star", "eccentricity"]
estimate_map = state_def.get_estimate_map(
    consider_params=consider_params, device=target_device
)

n_est = int(estimate_map.sum().item())
n_cons = n_total - n_est

P_cc = torch.eye(n_cons, dtype=torch.float64, device=target_device) #* 1e-4
P_x_inv = torch.eye(n_est, dtype=torch.float64, device=target_device) #* 1e-6

# ---------------------------------------------------------
# 5. Execute Sequential Iterations & Benchmark
# ---------------------------------------------------------
print("Executing sequential CCA solves...")
t0 = time.time()

final_x_list = []
final_P_list = []

# Standard loop for easier debugging and Jacobian introspection
for i in range(N_solves):
    x_in = x_init_batch[i]
    x_out, P_out = cca_solve_single(
        x_init=x_in,
        y_obs_fixed=doppler_obs,
        forward_fn=functional_forward,
        sigma_obs=sigma_obs,
        estimate_mask=estimate_map,
        P_cc=P_cc,
        P_x_inv=P_x_inv,
        num_steps=5,
        n_estimated=n_est,
    )
    final_x_list.append(x_out)
    final_P_list.append(P_out)

# Stack results back into batched tensors for unified analysis
final_x_batch = torch.stack(final_x_list)
final_P_batch = torch.stack(final_P_list)

t1 = time.time()

print(f"Time taken for {N_solves} sequential solves: {(t1 - t0) * 1000:.2f} ms")

print("\nConvergence Check (First Solve vs Central State):")
print(f"Central State: {x_true[:7].detach().cpu().numpy()}")
print(f"Final Output:  {final_x_batch[0, :7].detach().cpu().numpy()}")
print(f"Mean Output:   {final_x_batch.mean(dim=0)[:7].detach().cpu().numpy()}")
print(f"Final Consider Covariance Trace: {torch.trace(final_P_batch[0, :7, :7]).detach().cpu().numpy()}")