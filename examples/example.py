import numpy as np
import torch

from torch_sgp4.propagators.sgp4 import sgp4_propagate
from torch_sgp4.solvers.batchleastsquares import solve
from torch_sgp4.tle import tle_decode

# ---------------------------------------------------------
# 1. Compile the Pure Physics Engine
# ---------------------------------------------------------
# First, compile the core propagator exactly as you did in benchmark.py
compiled_propagate = torch.jit.script(sgp4_propagate)


@torch.jit.script
def compiled_forward_model(
    x: torch.Tensor,
    tsinces: torch.Tensor,
    st_pos: torch.Tensor,
    st_vel: torch.Tensor,
    center_freq: float,
    bias_indices: torch.Tensor,
    bias_global_offset: int,
    bias_num_params: int,
) -> torch.Tensor:
    """Stateless, fully typed hot-path for JIT compilation."""

    # 1. Propagate using the compiled propagator
    pos_sat, vel_sat = compiled_propagate(
        tsinces,
        bstar=x[0].unsqueeze(0),
        ndot=x[1].unsqueeze(0),
        nddot=x[2].unsqueeze(0),
        ecco=x[3].unsqueeze(0),
        argpo=x[4].unsqueeze(0),
        inclo=x[5].unsqueeze(0),
        mo=x[6].unsqueeze(0),
        no_kozai=x[7].unsqueeze(0),
        nodeo=x[8].unsqueeze(0),
    )

    # 2. Compute Doppler
    # Inlined from physics.py to guarantee JIT compatibility without function call overhead
    r_rel = pos_sat.squeeze(0) - st_pos
    v_rel = vel_sat.squeeze(0) - st_vel
    dist = torch.norm(r_rel, dim=1, keepdim=True)
    u_los = r_rel / (dist + 1e-9)
    range_rate = torch.sum(v_rel * u_los, dim=1)

    c_km_s = 299792.458
    y_raw = -(range_rate / c_km_s) * center_freq * 1e6

    # 3. Apply Biases
    # Replaced BiasGroup object with raw tensor slicing for JIT compatibility
    mask = bias_indices >= 0
    valid_indices = bias_indices[mask]
    bias_params = x[bias_global_offset : bias_global_offset + bias_num_params]
    active_biases = bias_params[valid_indices]

    y_corrected = y_raw.clone()
    y_corrected[mask] = y_corrected[mask] + active_biases

    return y_corrected


# ---------------------------------------------------------
# 2. Main Simulation & Recovery
# ---------------------------------------------------------
def simulate_and_recover_compiled():
    torch.set_default_dtype(torch.float64)

    tle_str = [
        "1 25544U 98067A   20316.40015046  .00001878  00000-0  44436-4 0  9997",
        "2 25544  51.6465 289.4354 0001961 270.2184  89.8601 15.49504104255152",
    ]
    x_base = tle_decode(tle_str)

    N_meas = 1000
    tsinces = torch.linspace(0, 120, N_meas)
    center_freq = 400.0
    st_pos = torch.zeros((N_meas, 3))
    st_pos[:, 0] = 6371.0
    st_vel = torch.zeros((N_meas, 3))

    num_passes = 2
    contact_ids = torch.cat(
        [torch.zeros(500, dtype=torch.long), torch.ones(500, dtype=torch.long)]
    )

    # Define Bias parameters as raw integers for the compiled function
    bias_offset = 9
    bias_num = num_passes

    # --- The Closure Wrapper ---
    def forward_wrapper(x: torch.Tensor) -> torch.Tensor:
        """Lightweight closure that jacfwd will digest."""
        return compiled_forward_model(
            x=x,
            tsinces=tsinces,
            st_pos=st_pos,
            st_vel=st_vel,
            center_freq=center_freq,
            bias_indices=contact_ids,
            bias_global_offset=bias_offset,
            bias_num_params=bias_num,
        )

    # Truth State
    true_mo_offset = 0.05
    true_biases = torch.tensor([35.5, -12.0], dtype=torch.float64)

    x_true = torch.cat([x_base.clone(), true_biases])
    x_true[6] += true_mo_offset

    print("Generating Synthetic Data (Compiled)...")
    y_clean = forward_wrapper(x_true)

    noise_sigma = 5.0
    torch.manual_seed(42)
    y_obs = y_clean + torch.randn_like(y_clean) * noise_sigma

    # Initial Guess
    x_init = torch.cat([x_base.clone(), torch.zeros(num_passes, dtype=torch.float64)])
    estimate_mask = torch.zeros(x_init.shape[0], dtype=torch.bool)
    estimate_mask[6] = True
    estimate_mask[9:] = True

    print("Running SVD Solver...")
    x_opt, P_cov = solve(
        x_init=x_init,
        y_obs_fixed=y_obs,
        forward_fn=forward_wrapper,
        sigma_obs=noise_sigma,
        estimate_mask=estimate_mask,
        num_steps=5,
    )

    # Evaluation
    mo_true = x_true[6].item()
    mo_init = x_init[6].item()
    mo_opt = x_opt[6].item()

    print("\n" + "=" * 40)
    print(f"{'Parameter':<15} | {'Initial':<10} | {'Truth':<10} | {'Recovered':<10}")
    print("-" * 55)
    print(
        f"{'Mean Anomaly':<15} | {mo_init:<10.5f} | {mo_true:<10.5f} | {mo_opt:<10.5f}"
    )

    for i in range(num_passes):
        b_true = x_true[9 + i].item()
        b_init = x_init[9 + i].item()
        b_opt = x_opt[9 + i].item()
        print(f"Bias Pass {i:<4} | {b_init:<10.1f} | {b_true:<10.1f} | {b_opt:<10.1f}")


if __name__ == "__main__":
    simulate_and_recover_compiled()
