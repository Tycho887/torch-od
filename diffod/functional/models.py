import torch
import torch.nn as nn

from diffod.functional.sgp4 import sgp4_propagate
from diffod.physics import apply_linear_bias, compute_doppler


class DopplerResiduals(nn.Module):
    """
    A strictly stateless pipeline.
    Safe to instantiate once, compile once, and share across API threads.
    """

    def __init__(self, state_def, bias_group=None) -> None:
        super().__init__()
        # Only structural metadata is kept here. No observation tensors!
        self.state_def = state_def
        self.bias_group = bias_group

    def forward(
        self,
        x: torch.Tensor,
        tsince: torch.Tensor,
        st_pos: torch.Tensor,
        st_vel: torch.Tensor,
        center_freq: torch.Tensor,
    ) -> torch.Tensor:
        """
        All observation data is injected at inference time.
        """
        # 1. Propagate Orbit
        sgp4_args = self.state_def.get_functional_args(x)
        sat_pos, sat_vel = sgp4_propagate(tsince=tsince, **sgp4_args)

        # 2. Physics Projection
        raw_doppler = compute_doppler(
            sat_pos=sat_pos,
            sat_vel=sat_vel,
            st_pos=st_pos,
            st_vel=st_vel,
            center_freq=center_freq,
        )

        # 3. Bias Correction
        if self.bias_group is not None:
            return apply_linear_bias(
                predictions=raw_doppler, x_state=x, bias_group=self.bias_group
            )
        return raw_doppler
