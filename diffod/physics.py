import torch


def compute_doppler(sat_pos, sat_vel, st_pos, st_vel, center_freq):
    """
    Computes One-Way Doppler shift.

    Args:
        sat_pos, sat_vel: (N, 3) Satellite State (TEME)
        st_pos, st_vel:   (N, 3) Station State (TEME)
        center_freq:      float or (N,) tensor

    Returns:
        (N,) Doppler shift in Hz
    """
    # Relative State
    r_rel = sat_pos - st_pos
    v_rel = sat_vel - st_vel

    # Range (Distance)
    dist = torch.norm(r_rel, dim=1, keepdim=True)

    # Line of Sight Vector
    u_los = r_rel / (dist + 1e-9)

    # Range Rate (Projection of velocity onto LOS)
    range_rate = torch.sum(v_rel * u_los, dim=1)

    # Doppler Equation: - (range_rate / c) * f
    c_km_s = 299792.458
    return -(range_rate / c_km_s) * center_freq


def compute_range(sat_pos, st_pos):
    """Computes Slant Range in km."""
    return torch.norm(sat_pos - st_pos, dim=1)


def apply_linear_bias(observations, x_state, bias_matrix):
    """
    Applies additive bias: y_pred = y_phys + H @ x
    """
    # Sparse Matrix-Vector Multiplication
    # (N, P) @ (P, 1) -> (N, 1)
    bias_correction = torch.sparse.mm(bias_matrix, x_state.unsqueeze(1)).squeeze()
    return observations + bias_correction
