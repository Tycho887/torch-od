import torch
import numpy as np
from scipy.optimize import minimize
from torch.func import jacfwd
from collections.abc import Callable

def lbfgs_solve(
    x_init: torch.Tensor,
    y_obs_fixed: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma_obs: float,
    estimate_mask: torch.Tensor,
    # bounds: list[tuple[float | None, float | None]], 
    max_iter: int = 1000
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Uses SciPy's L-BFGS-B to solve the OD problem with strict bounds,
    utilizing PyTorch Autograd for exact Jacobians/Gradients.
    
    `bounds` should be a list of (min, max) tuples matching the length 
    of the estimated parameters (where estimate_mask is True).
    Use None for no bound, e.g., (0.0, 0.999) for eccentricity.
    """
    # Assuming elements: [ecco, argpo, inclo, mo, no_kozai, nodeo]
    # eccentricity must be in [0, 1), inclination in [0, pi], others unconstrained (or 0 to 2pi)
    bounds = [
        (None, None),    # argpo (can optionally bound 0 to 2pi)
        # (None, None),    # argpo (can optionally bound 0 to 2pi)
        # (None, None),    # argpo (can optionally bound 0 to 2pi)
        (0.0, 0.9999),   # ecco
        (None, None),    # argpo (can optionally bound 0 to 2pi)
        (0.0, 3.14159),  # inclo
        (None, None),    # mo
        (0.0, None),     # no_kozai (mean motion strictly positive)
        (None, None)     # nodeo
    ]

    # Isolate the fixed state and the initial guess for active parameters
    x_full = x_init.detach().clone().to(torch.float64)
    x_active_init = x_full[estimate_mask].cpu().numpy()

    # Define the objective function for SciPy
    def cost_and_grad(x_active_np: np.ndarray) -> tuple[float, np.ndarray]:
        # 1. Convert numpy array back to an active PyTorch tensor
        x_active_torch = torch.tensor(
            x_active_np, 
            dtype=torch.float64, 
            device=x_init.device, 
            requires_grad=True
        )
        
        # 2. Reconstruct the full state dynamically
        x_current = x_full.clone()
        x_current[estimate_mask] = x_active_torch
        
        # 3. Forward pass and Loss computation (Weighted SSR)
        y_model = forward_fn(x_current)
        residual = y_obs_fixed - y_model
        loss = torch.sum((residual / sigma_obs) ** 2)
        
        # 4. Compute exact gradients via Autograd
        loss.backward()
        grad_np = x_active_torch.grad.cpu().numpy().astype(np.float64)
        
        # SciPy expects a scalar loss and a flat gradient array
        return loss.item(), grad_np

    # Execute SciPy's L-BFGS-B
    print("Handing off to SciPy L-BFGS-B...")
    result = minimize(
        fun=cost_and_grad,
        x0=x_active_init,
        method='L-BFGS-B',
        jac=True,           # Tells SciPy our function returns (loss, gradient)
        bounds=bounds,
        options={
            'maxiter': max_iter,
            'ftol': 1e-12,  # Tight tolerance for OD
            'gtol': 1e-12,
            'disp': True    # Prints convergence logs to terminal
        }
    )

    # ---------------------------------------------------------
    # Update state and recover Covariance Matrix
    # ---------------------------------------------------------
    
    # Inject the optimal bounded parameters back into the state tensor
    x_full[estimate_mask] = torch.tensor(
        result.x, 
        dtype=torch.float64, 
        device=x_init.device
    )

    with torch.no_grad():
        # Recompute the Jacobian at the final bounded state
        J_full = jacfwd(forward_fn)(x_full)
        J_final = J_full[:, estimate_mask]
        
        # Calculate covariance: P = (J^T W J)^-1
        sqrt_w = 1.0 / sigma_obs
        Jw = J_final * sqrt_w
        H_final = Jw.T @ Jw
        
        # Use pseudoinverse to safely handle remaining FNN collinearity
        P_cov = torch.linalg.pinv(H_final)

    return x_full, P_cov