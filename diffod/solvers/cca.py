import torch


def compute_cca_step(
    H_total: torch.Tensor,
    y_residual: torch.Tensor,
    sigma_obs: float,
    n_estimated: int,
    P_cc: torch.Tensor,
    P_x_inv: torch.Tensor,
    x_prior_res: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Maximum Likelihood update and the Consider Covariance matrix.

    Args:
        H_total: Full Jacobian matrix (N_obs, n_est + q_consider)
        y_residual: Residual vector (N_obs,)
        sigma_obs: Standard deviation of observation noise
        n_estimated: Number of parameters being actively estimated (n)
        P_cc: A priori covariance of consider parameters (q, q)
        P_x_inv: Optional a priori information matrix for estimated params (n, n)
        x_prior_res: Optional residual of current state vs prior state (n,)

    Returns:
        dx: The optimal state update vector (n,)
        P_total: The total covariance matrix including consider effects (n, n)
    """
    # 2. Partitioning the Jacobian
    # Extract estimated (H_x) and consider (H_c) sensitivities
    H_x = H_total[:, :n_estimated]
    H_c = H_total[:, n_estimated:]

    # 3. Maximum Likelihood Step
    # Create the weight scalar/vector (W = R^-1).
    # For independent noise, we apply it via broadcasting.
    W = 1.0 / (sigma_obs**2)
    H_x_W = H_x.T * W

    # Accumulate Information Matrix (M_xx)
    M_xx = H_x_W @ H_x
    if P_x_inv is not None:
        M_xx += P_x_inv

    # Accumulate Right-Hand Side (N_x)
    N_x = H_x_W @ y_residual
    if P_x_inv is not None and x_prior_res is not None:
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
