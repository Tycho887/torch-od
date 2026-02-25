from collections.abc import Callable
import torch
from torch.func import jacfwd


def compute_cca_step(
    H_total: torch.Tensor,
    residuals_fp64: torch.Tensor,
    sigma_obs: float,
    estimate_mask: torch.Tensor,
    consider_mask: torch.Tensor,
    P_cc: torch.Tensor,
    P_x_inv: torch.Tensor,
    x_prior_res: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Maximum Likelihood update and the Consider Covariance matrix.

    Args:
        H_total: The total Jacobian matrix of the residual function (n_obs, n_total).
        residuals_fp64: The residual vector y_calc - y_obs (n_obs,).
        sigma_obs: The standard deviation of observation noise.
        estimate_mask: Boolean mask indicating parameters being estimated.
        consider_mask: Boolean mask indicating unestimated 'consider' parameters.
        P_cc: A priori covariance of consider parameters (n_consider, n_consider).
        P_x_inv: A priori information matrix for estimated params (n_estimated, n_estimated).
        x_prior_res: Residual of current state vs prior state (n_estimated,).
    """
    # 1. Partitioning the Jacobian using explicit masks
    H_x = H_total[:, estimate_mask]
    H_c = H_total[:, consider_mask]

    # 2. Maximum Likelihood Step
    # Create the weight scalar/vector (W = R^-1).
    W = 1.0 / (sigma_obs**2)
    H_x_W = H_x.T * W

    # Accumulate Information Matrix (M_xx)
    M_xx = H_x_W @ H_x
    M_xx += P_x_inv

    # Accumulate Right-Hand Side (N_x)
    N_x = H_x_W @ residuals_fp64
    N_x += P_x_inv @ x_prior_res

    # 3. Optimize linear solves via a single Cholesky decomposition
    L = torch.linalg.cholesky(M_xx)

    # Solve for the state update (dx)
    dx = torch.cholesky_solve(N_x.unsqueeze(-1), L).squeeze(-1)

    print(f"Update Norm: {torch.linalg.norm(dx):.6e}")


    # 4. Consider Covariance Step
    # Compute Cross-Coupling Matrix (M_xc)
    M_xc = H_x_W @ H_c

    # Compute Sensitivity Matrix (S_xc = - M_xx^-1 M_xc)
    S_xc = torch.cholesky_solve(-M_xc, L)

    # Compute Total Covariance for the estimated parameters
    P_noise = torch.cholesky_inverse(L)
    P_consider = S_xc @ P_cc @ S_xc.T

    P_total = P_noise + P_consider

    return dx, P_total


def cca_solve(
    x_init: torch.Tensor,
    y_obs_fixed: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma_obs: float,
    estimate_mask: torch.Tensor,
    consider_mask: torch.Tensor,
    P_cc: torch.Tensor,
    P_x_inv: torch.Tensor,
    # n_estimated: int,
    num_steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Executes the CCA iterative solver for a single state.
    Designed to be vmapped across a batch.
    """
    x = x_init.to(torch.float64)
    # The a priori state anchors the regularization penalty
    x_apriori = x.clone()

    # Pre-allocate P_total to the correct dimension of estimated parameters
    P_total = torch.eye(len(estimate_mask)+len(consider_mask), dtype=x.dtype, device=x.device)

    def res_fn(state):
        return forward_fn(state) - y_obs_fixed

    for _ in range(num_steps):
        # 1. Forward Pass & Jacobian (FP64)
        residuals_val = res_fn(x)

        H_val = jacfwd(res_fn)(x)

        # 2. Compute a priori residual for estimated parameters
        x_prior_res = x[estimate_mask] - x_apriori[estimate_mask]

        # 3. Compute Step and Covariance
        dx, P_total = compute_cca_step(
            H_total=H_val,
            residuals_fp64=residuals_val,
            sigma_obs=sigma_obs,
            estimate_mask=estimate_mask,
            consider_mask=consider_mask,
            P_cc=P_cc,
            P_x_inv=P_x_inv,
            x_prior_res=x_prior_res,
        )

        # 4. Apply update only to estimated parameters
        x[estimate_mask] = x[estimate_mask] - dx

    return x, P_total