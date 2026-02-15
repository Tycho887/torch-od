import torch


def single_gn_step_scaled(x_single_fp64, y_obs_fixed):
    """Executes a single, well-conditioned GN step mapped over a batch of states."""
    x_fp64 = x_single_fp64.to(torch.float64)

    # 1. Forward Pass & Jacobian (FP32)
    y_pred_fp32 = functional_forward(x_fp64)
    H_fp64 = jacfwd(func=functional_forward)(x_fp64)

    # 2. Cast to FP64
    # H_fp64 = H_fp32.to(torch.float64)
    dy_fp64 = (y_obs_fixed - y_pred_fp32).to(torch.float64)

    # 3. Apply Scaling Transformations using Broadcasting
    # scale_x.unsqueeze(0) broadcasts across columns (parameters)
    # scale_y.unsqueeze(1) broadcasts across rows (observations)
    H_scaled = H_fp64 * scale_x.unsqueeze(0) * scale_y.unsqueeze(1)
    dy_scaled = dy_fp64 * scale_y

    # 4. Form Well-Conditioned Normal Equations
    H_T = H_scaled.T
    A_scaled = H_T @ H_scaled
    b_scaled = H_T @ dy_scaled

    # 5. Levenberg-Marquardt Damping (Now much more effective in scaled space)
    damping = (
        torch.eye(A_scaled.shape[0], dtype=torch.float64, device=target_device) * 1e-6
    )
    A_damped = A_scaled + damping

    # 6. Solve for the scaled update (du)
    du = torch.linalg.solve(A_damped, b_scaled)

    # 7. Rescale update back to physical units (dx)
    dx = du * scale_x

    return x_single_fp64 + dx
