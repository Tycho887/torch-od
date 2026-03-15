# Advanced Features: Observation Models and Gauss‑Newton Solver

This guide shows how to build differentiable measurement models and use the built‑in Gauss‑Newton SVD solver for orbit determination.

## Observation Models

- **Doppler:** `compute_doppler(pos_sat, vel_sat, st_pos, st_vel, center_freq)` returns one‑way Doppler shift (Hz).
- **Range:** `compute_range(pos_sat, st_pos)` returns slant range (km).
- **Biases:** Use `BiasGroup` to map per‑pass biases from a flat state vector, and `apply_linear_bias` to add them to raw measurements.

### Example: Doppler Forward Model with Biases

```python
from torch_od.physics import compute_doppler, apply_linear_bias
from torch_od.utils import BiasGroup

# Assume we have contact_ids (N,) indicating which pass each measurement belongs to
bias_group = BiasGroup(
    name="doppler_pass_bias",
    indices=contact_ids,
    global_offset=9,          # biases start after the 9 TLE parameters
    num_params=2               # two passes → two bias parameters
)

def forward_model(x: torch.Tensor) -> torch.Tensor:
    # x is flat: [9 TLE params, 2 biases]
    pos_sat, vel_sat = sgp4_propagate(tsinces, *x[:9].unbind())
    y_raw = compute_doppler(pos_sat.squeeze(0), vel_sat.squeeze(0),
                            st_pos, st_vel, center_freq)
    y_corrected = apply_linear_bias(y_raw, x, bias_group, scaling=1.0)
    return y_corrected
```

## Gauss‑Newton SVD Solver

The `svd_solve` function performs iterative least‑squares estimation using an SVD‑based pseudoinverse. It automatically computes the Jacobian of your forward model with `torch.func.jacfwd`.

```python
from torch_od.gn_svd import svd_solve

# Initial state (9 TLE params + biases)
x_init = torch.cat([tle_decode(tle_iss), torch.zeros(2)])

# Mask which parameters to estimate
estimate_mask = torch.zeros_like(x_init, dtype=bool)
estimate_mask[0] = True   # estimate BSTAR
estimate_mask[6] = True   # estimate Mean Anomaly
estimate_mask[9:] = True  # estimate biases

x_opt, P_cov = svd_solve(
    x_init=x_init,
    y_obs_fixed=y_obs,          # actual measurements
    forward_fn=forward_model,
    sigma_obs=5.0,               # measurement noise std
    estimate_mask=estimate_mask,
    num_steps=5
)
```

*See `example.py` for a complete simulation and recovery example.*
