from doctest import DocTestParser
import re
import time

import numpy as np
import torch
import polars as pl
import matplotlib.pyplot as plt

from dsgp4.tle import TLE
from torch.utils.show_pickle import FakeClass
import diffod.state as state
from diffod.functional.system import PredictDoppler
from diffod.gse import station_teme_preprocessor

# Swapped out CCA for the WGN solver
from diffod.solvers.gaussNewton import wgn_solve_single

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
    return model(x=x, tsince=t_obs/60.0, st_pos=st_pos, st_vel=st_vel, center_freq=2.2e9)

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
# estimate_map = active_map
print(active_map)

# Only the prior information matrix is needed for standard WGN
P_x_inv = torch.eye(n=2, dtype=torch.float64, device=target_device) #* 1e-6

# ---------------------------------------------------------
# 5. Execute Sequential Iterations & Benchmark
# ---------------------------------------------------------
print("Executing sequential WGN solves...")
t0 = time.time()

final_x_list = []
final_P_list = []

# In wgn_test.py Section 5:
for i in range(N_solves):
    x_in = x_init_batch[i]
    x_out, P_out = wgn_solve_single(
        x_init=x_in,
        y_obs_fixed=doppler_obs,
        forward_fn=functional_forward,
        sigma_obs=sigma_obs,
        estimate_mask=active_map,  # This tells the solver which columns of J to use
        num_steps=5,
        # The following are passed but ignored by the new simplified solver
        P_x_inv=P_x_inv,
    )
    final_x_list.append(x_out)
    final_P_list.append(P_out)

final_x_batch = torch.stack(tensors=final_x_list)
final_P_batch = torch.stack(tensors=final_P_list)

t1 = time.time()

print(f"Time taken for {N_solves} sequential solves: {(t1 - t0) * 1000:.2f} ms")

print("\nConvergence Check (First Solve vs Central State):")
print(f"Central State: {x_true[:7].detach().cpu().numpy()}")
print(f"Final Output:  {final_x_batch[0, :7].detach().cpu().numpy()}")
print(f"Mean Output:   {final_x_batch.mean(dim=0)[:7].detach().cpu().numpy()}")
print(f"Final Covariance Trace: {torch.trace(final_P_batch[0, :7, :7]).detach().cpu().numpy()}")

# ---------------------------------------------------------
# 6. Plotting Results
# ---------------------------------------------------------
print("\nGenerating plots...")

# Disable gradients for inference to save memory and speed up computation
with torch.no_grad():
    y_init = functional_forward(x_init_batch[0]).cpu().numpy()
    y_fit = functional_forward(final_x_batch[0]).cpu().numpy()

# Convert time to relative minutes for readability
t_plot = (times_unix - times_unix[0]).cpu().numpy() / 60.0
y_obs = doppler_obs.cpu().numpy()



plt.figure(figsize=(12, 7))

# Plot observations (scatter plot is usually best for real noisy telemetry)
plt.scatter(t_plot, y_obs, s=4, c='black', alpha=0.5, label='Observations')

# Plot initial guess
plt.plot(t_plot, y_init, 'r--', linewidth=1.5, label='Initial Guess')

# Plot the converged WGN fit
plt.plot(t_plot, y_fit, 'g-', linewidth=2, label='WGN Fitted Curve')

plt.title('Doppler Shift Orbit Determination Fit', fontsize=14)
plt.xlabel('Time since start of track (Minutes)', fontsize=12)
plt.ylabel('Doppler Shift (Hz)', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

plt.show()