import torch
import torch.nn as nn
from torch.func import jacfwd
from collections.abc import Callable

def lbfgs_solve(
    x_init: torch.Tensor,
    y_obs_fixed: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma_obs: float,
    estimate_mask: torch.Tensor,
    max_iter: int = 500,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    robust_c: float = 3.0,  # <-- NEW: Outlier threshold in sigmas (0 means use L2)
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative solver using PyTorch's native L-BFGS optimizer.
    """
    # Keep the non-estimated parameters fixed
    x_full = x_init.detach().clone().to(torch.float64)
    
    # Extract only the targeted parameters into an active Parameter tensor
    x_opt = nn.Parameter(x_full[estimate_mask].clone())
    
    # Initialize L-BFGS. 
    # The 'strong_wolfe' line search is highly recommended for ML-physics models
    # as it prevents taking excessively large steps into unstable unphysical regimes.
    optimizer = torch.optim.LBFGS(
        [x_opt], 
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe" 
    )

    def closure():
        optimizer.zero_grad()
        
        x_current = x_full.clone()
        x_current[estimate_mask] = x_opt
        
        y_model = forward_fn(x_current)
        residual = y_obs_fixed - y_model
        
        # Normalize the residual by the expected noise floor
        norm_res = residual / sigma_obs
        
        if robust_c > 0.0:
            # Cauchy Robust Loss: Ignores anything far outside the 'epsilon cone'
            loss = torch.sum((robust_c**2) * torch.log(1 + (norm_res / robust_c)**2))
        else:
            # Standard L2 Loss
            loss = torch.sum(norm_res ** 2)
        
        loss.backward()
        
        rmse = torch.sqrt(torch.mean(residual**2)).item()
        print(f"L-BFGS Step RMSE: {rmse:.6f} | Loss: {loss.item():.6e}")
        
        return loss

    # Execute the optimization
    optimizer.step(closure)
    
    # Update the full state with the optimized values
    with torch.no_grad():
        x_full[estimate_mask] = x_opt.detach()

    # --- Covariance Recovery ---
    # L-BFGS does not explicitly maintain or return the full inverse Hessian.
    # To get P_cov for OD filtering/analysis, we evaluate the Jacobian at the final optimal state.
    
    with torch.no_grad():
        # Compute final Jacobian wrt the full state, then mask
        J_full = jacfwd(forward_fn)(x_full)
        J_final = J_full[:, estimate_mask]
        
        # Calculate covariance: P = (J^T W J)^-1
        sqrt_w = 1.0 / sigma_obs
        Jw = J_final * sqrt_w
        H_final = Jw.T @ Jw
        
        # Use pseudo-inverse for safety, or add a tiny eps to the diagonal
        P_cov = torch.linalg.pinv(H_final)

    return x_full, P_cov