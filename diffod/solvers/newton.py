import torch
from torch.func import grad, hessian
from collections.abc import Callable

def solve_newton_step(
    H: torch.Tensor, 
    g: torch.Tensor,
    eps: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard Newton step using symmetric preconditioning for numerical stability.
    Solves: H * dx = -g
    """
    # Symmetric preconditioning (analogous to your Jacobian column normalization)
    # D = diag(1 / sqrt(diag(H) + eps))
    d = torch.sqrt(torch.abs(torch.diag(H)) + eps)
    D_inv = 1.0 / d
    
    # Scale H and g: H_scaled = D_inv * H * D_inv, g_scaled = D_inv * g
    H_scaled = D_inv[:, None] * H * D_inv[None, :]
    g_scaled = D_inv * g
    
    print(f"cond: {torch.linalg.cond(H_scaled):.2e}")
    
    # Solve the well-conditioned system: H_scaled * dx_scaled = -g_scaled
    dx_scaled = torch.linalg.solve(H_scaled, -g_scaled)
    
    # Revert the scaling on the update vector
    dx = D_inv * dx_scaled
    
    # Covariance estimate: H^-1
    # We use the scaled H to compute the inverse safely, then unscale it
    P_cov = D_inv[:, None] * torch.linalg.inv(H_scaled) * D_inv[None, :]
    
    return dx, P_cov

def newton_solve(
    x_init: torch.Tensor,
    y_obs_fixed: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma_obs: float,
    estimate_mask: torch.Tensor,
    num_steps: int = 5,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative solver using the exact exact Hessian via PyTorch functional API.
    """
    x = x_init.detach().clone().to(torch.float64)
    sqrt_w = 1.0 # / sigma_obs
    
    n_total = x.shape[0]
    final_P = torch.zeros((n_total, n_total), dtype=torch.float64, device=x.device)

    # 1. Define the scalar objective function: 1/2 * sum( (w * r)^2 )
    def objective_fn(x_state: torch.Tensor) -> torch.Tensor:
        y_model = forward_fn(x_state)
        r = y_obs_fixed - y_model
        rw = r * sqrt_w
        return 0.5 * torch.sum(rw ** 2)

    for step in range(num_steps):
        # 2. Compute exact exact Gradient and Hessian wrt full state x
        g_full = grad(objective_fn)(x)
        H_full = hessian(objective_fn)(x)
        
        # 3. Mask Gradient and Hessian for estimated parameters only
        # Gradient is a vector (1D mask), Hessian is a matrix (2D mask)
        g_masked = g_full[estimate_mask]
        H_masked = H_full[estimate_mask][:, estimate_mask]
        
        # 4. Solve the Newton step
        dx, P_cov = solve_newton_step(H_masked, g_masked)
        
        # 5. Apply update to the masked indices
        x[estimate_mask] = x[estimate_mask] + dx
        
        print(f"Step {step+1} | Update Norm: {torch.linalg.norm(dx):.6e}")
        print(f"State: {x}")

        final_P = P_cov 

    return x, final_P