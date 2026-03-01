import torch
from diffod.utils import BiasGroup

def compute_doppler(sat_pos, sat_vel, st_pos, st_vel, center_freq) -> torch.Tensor:
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
    dist = torch.norm(input=r_rel, dim=1, keepdim=True)

    # Line of Sight Vector
    u_los = r_rel / (dist + 1e-9)

    # Range Rate (Projection of velocity onto LOS)
    range_rate = torch.sum(input=v_rel * u_los, dim=1)

    # Doppler Equation: - (range_rate / c) * f
    c_km_s = 299792.458
    return -(range_rate / c_km_s) * center_freq * 1e6


def compute_range(sat_pos, st_pos) -> torch.Tensor:
    """Computes Slant Range in km."""
    return torch.norm(input=(sat_pos - st_pos), dim=1)


def apply_linear_bias(
    predictions: torch.Tensor, x_state: torch.Tensor, bias_group: BiasGroup, scaling: float = 1
) -> torch.Tensor:
    if bias_group is None:
        return predictions

    # 1. Identify which measurements have a bias
    # (Assuming -1 in indices means "no bias")
    mask = bias_group.indices >= 0
    valid_indices = bias_group.indices[mask]

    # 2. Extract the relevant chunk of the state vector 'x'
    # Slice: [offset : offset + num_params]
    start = bias_group.global_offset
    end = start + bias_group.num_params
    bias_params = x_state[start:end]  # Shape (Num_Biases,)

    # 3. Gather the specific bias value for each valid measurement
    # Shape (N_valid,)
    active_biases = bias_params[valid_indices]

    # 4. Apply
    # We clone to avoid in-place modification errors in AD
    corrected = predictions.clone()
    corrected[mask] = corrected[mask] + active_biases * scaling

    return corrected
