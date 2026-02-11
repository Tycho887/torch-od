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


def apply_linear_bias(observations, x_state, bias_map):
    """
    Applies additive bias using indexing (Gather).

    Args:
        observations: (N,) Tensor of computed observations.
        x_state:      (M,) The full state vector.
        bias_map:     Dict containing 'indices' (N,) and 'offset' (int).
    """
    if bias_map is None:
        return observations

    indices = bias_map["indices"]
    offset = bias_map["offset"]

    # 1. Handle "No Bias" entries (indicated by -1)
    # Create a mask of which observations actually have this bias
    mask = indices >= 0

    # 2. Calculate the correction
    # We initialize a zero-tensor so observations with index -1 get +0.0
    bias_correction = torch.zeros_like(observations)

    if mask.any():
        # A. Get the valid group IDs
        valid_group_ids = indices[mask]

        # B. Shift them to absolute positions in the state vector 'x'
        # e.g., if biases start at index 10, group 0 is at x[10]
        abs_indices = valid_group_ids + offset

        # C. Gather the values from x_state
        # This operation is fully supported by jacfwd
        bias_values = x_state[abs_indices]

        # D. Assign back to the correction vector
        bias_correction[mask] = bias_values

    return observations + bias_correction
