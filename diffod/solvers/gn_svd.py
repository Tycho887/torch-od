import torch
from torch.func import jacfwd
from collections.abc import Callable

def solve_gn_step_svd(
    J: torch.Tensor, 
    y_model: torch.Tensor, 
    y_obs: torch.Tensor, 
    sqrt_w: float = 1.0,
    rcond: float = 1e-6  # Cutoff for collinear singular values
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Direct SVD-based least squares solver to handle severe collinearity.
    """
    r = y_obs - y_model
    # print(y_model)
    # print(f"Observation mean: {torch.mean(input=y_obs):.6f}, Model mean: {torch.mean(input=y_model):.6f}")
    # print(f"RMSE: {torch.sqrt(input=torch.mean(input=r**2)).detach().item():.6f}")
    
    Jw = J * sqrt_w
    rw = r * sqrt_w
    
    # Column normalization for scaling parity
    col_norms = torch.norm(Jw, dim=0) + 1e-10
    Jn = Jw / col_norms
    
    # Solve Jn * dx_tilde = rw directly using SVD
    # rcond zeroes out singular values that cause the instability
    dx_tilde = torch.linalg.lstsq(Jn, rw, rcond=rcond).solution
    
    dx = dx_tilde / col_norms

    print(f"Update Norm: {torch.linalg.norm(dx):.6e}")
    
    # Covariance estimate via pseudoinverse of the Jacobian
    # P = (J^T J)^-1  => P = V (Sigma^-2) V^T
    # This is safer than inverting Hn directly
    Jn_pinv = torch.linalg.pinv(Jn, rcond=rcond)
    P_cov = (Jn_pinv @ Jn_pinv.T) / (col_norms[:, None] @ col_norms[None, :])
    
    # print(dx)

    return dx, P_cov

def svd_solve(
    x_init: torch.Tensor,
    y_obs_fixed: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma_obs: float,
    estimate_mask: torch.Tensor,
    num_steps: int = 5,
    **kwargs # Absorbs P_x_inv and n_estimated from your test script
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative solver using the simplified Normal Equations approach.
    """
    x = x_init.detach().clone().to(torch.float64)
    sqrt_w = 1.0 #/ sigma_obs
    
    # Initialize a dummy covariance for the full state size
    n_total = x.shape[0]
    final_P = torch.zeros((n_total, n_total), dtype=torch.float64, device=x.device)

    for _ in range(num_steps):
        # 1. Compute Model and Jacobian
        y_model = forward_fn(x)

        print(x)

        # Jacobian wrt full state x
        J_full = jacfwd(forward_fn)(x)
        
        # 2. Mask Jacobian for estimated parameters only
        J_masked = J_full[:, estimate_mask]
        
        # 3. Solve Normal Equations
        dx, P_cov = solve_gn_step_svd(J_masked, y_model, y_obs_fixed, sqrt_w)
        
        # 4. Apply update to the masked indices
        x[estimate_mask] = x[estimate_mask] + dx
        
        # print(torch.linalg.norm(dx), len(dx))
        # print(x)

        # Keep track of the last covariance for the estimated block
        final_P = P_cov 

    return x, final_P