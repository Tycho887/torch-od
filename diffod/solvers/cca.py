from typing import Callable

import torch
from torch.func import jacfwd


def compute_cca_step(
    H_total: torch.Tensor,
    residuals_fp64: torch.Tensor,
    sigma_obs: float,
    consider_map: torch.Tensor,
    P_cc: torch.Tensor,
    P_x_inv: torch.Tensor,
    x_prior_res: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Maximum Likelihood update and the Consider Covariance matrix.

    Args:
        H_total: The total Jacobian matrix of the residual function (n, n_total).
        residuals_fp64: The residual vector (n_estimated,).
        sigma_obs: The standard deviation of observation noise.
        consider_map: A boolean mask indicating which parameters are being considered.
        P_cc: The a priori covariance of consider parameters (q, q).
        P_x_inv: The a priori information matrix for estimated parameters (n_estimated, n_estimated).
        x_prior_res: The residual of current state vs prior state (n_estimated,).

    Returns:
        dx: The optimal state update vector (n,)
        P_total: The total covariance matrix including consider effects (n, n)
    """
    # 1. Computing the Jacobian of the residual function
    # y_fp64 = residual_function(x_fp64)
    # H_total = jacfwd(residual_function)(x_fp64)

    # 2. Partitioning the Jacobian
    # Extract estimated (H_x) and consider (H_c) sensitivities using the boolean mask
    H_x = H_total[:, consider_map]
    H_c = H_total[:, ~consider_map]

    # 3. Maximum Likelihood Step
    # Create the weight scalar/vector (W = R^-1).
    # For independent noise, we apply it via broadcasting.
    W = 1.0 / (sigma_obs**2)
    H_x_W = H_x.T * W

    # Accumulate Information Matrix (M_xx)
    M_xx = H_x_W @ H_x
    M_xx += P_x_inv

    # Accumulate Right-Hand Side (N_x)
    N_x = H_x_W @ residuals_fp64
    N_x += P_x_inv @ x_prior_res

    # Solve for the state update (x_hat)
    # Using torch.linalg.solve ensures we utilize optimized LU/Cholesky backends
    dx = torch.linalg.solve(M_xx, N_x.unsqueeze(-1)).squeeze(-1)

    # 4. Consider Covariance Step
    # Compute Cross-Coupling Matrix (M_xc)
    M_xc = H_x_W @ H_c

    # Compute Sensitivity Matrix (S_xc = - M_xx^-1 M_xc)
    # We solve M_xx * S_xc = -M_xc to avoid explicitly inverting M_xx
    S_xc = torch.linalg.solve(M_xx, -M_xc)

    # Compute Total Covariance
    P_noise = torch.linalg.inv(M_xx)
    P_consider = S_xc @ P_cc @ S_xc.T

    P_total = P_noise + P_consider

    return dx, P_total


def cca_solve_single(
    x_init: torch.Tensor,
    y_obs_fixed: torch.Tensor,
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma_obs: float,
    consider_map: torch.Tensor,
    P_cc: torch.Tensor,
    P_x_inv: torch.Tensor,
    num_steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Executes the CCA iterative solver for a single state.
    Designed to be vmapped across a batch.
    """
    x = x_init.to(torch.float64)
    # The a priori state anchors the regularization penalty
    x_apriori = x.clone()
    P_total = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)

    def res_fn(state):
        return residual_fn(state) - y_obs_fixed

    for _ in range(num_steps):
        # 1. Forward Pass & Jacobian (FP64)
        residuals_val = res_fn(x)
        H_val = jacfwd(res_fn)(x)

        # 2. Compute a priori residual for estimated parameters
        x_prior_res = x[consider_map] - x_apriori[consider_map]

        # 3. Compute Step and Covariance
        dx, P_total = compute_cca_step(
            H_total=H_val,
            residuals_fp64=residuals_val,
            sigma_obs=sigma_obs,
            consider_map=consider_map,
            P_cc=P_cc,
            P_x_inv=P_x_inv,
            x_prior_res=x_prior_res,
        )

        # 4. Apply update only to estimated parameters
        x[consider_map] = x[consider_map] - dx

    return x, P_total
