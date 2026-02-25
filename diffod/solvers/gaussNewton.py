import torch
from torch.func import jacfwd
from collections.abc import Callable

def solve_gn_step(
    J: torch.Tensor, 
    y_model: torch.Tensor, 
    y_obs: torch.Tensor, 
    sqrt_w: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard Gauss-Newton step using Normal Equations with column normalization.
    """
    # Calculate residuals
    r = y_obs - y_model
    
    # Apply weights (sqrt_w is 1/sigma)
    Jw = J * sqrt_w
    rw = r * sqrt_w
    
    # Column normalization for numerical stability
    col_norms = torch.norm(Jw, dim=0) + 1e-10
    Jn = Jw / col_norms
    
    # Form Normal Equations: (Jn^T @ Jn) @ dx_tilde = Jn^T @ rw
    Hn = Jn.T @ Jn
    bn = Jn.T @ rw

    # print(Hn)

    # print(f"cond: {torch.linalg.cond(Hn)}")
    
    # Solve for update
    dx_tilde = torch.linalg.solve(Hn, bn)
    dx = dx_tilde / col_norms

    print(f"Update Norm: {torch.linalg.norm(dx):.6e}")
    
    # Covariance estimate: (J^T W J)^-1
    # We use the normalized Hn to compute the inverse safely
    P_cov = torch.linalg.inv(Hn) / (col_norms[:, None] @ col_norms[None, :])
    
    return dx, P_cov

def wgn_solve(
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
        # Jacobian wrt full state x
        J_full = jacfwd(forward_fn)(x)
        
        # 2. Mask Jacobian for estimated parameters only
        J_masked = J_full[:, estimate_mask]
        
        # 3. Solve Normal Equations
        dx, P_cov = solve_gn_step(J_masked, y_model, y_obs_fixed, sqrt_w)
        
        # 4. Apply update to the masked indices
        x[estimate_mask] = x[estimate_mask] + dx
        
        # print(torch.linalg.norm(dx), len(dx))
        # print(x)

        # Keep track of the last covariance for the estimated block
        final_P = P_cov 

    return x, final_P