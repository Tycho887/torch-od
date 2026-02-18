import torch
import torch.nn as nn

from diffod.functional.mlsgp4 import FunctionalMLdSGP4
from diffod.functional.sgp4 import sgp4_propagate
from diffod.physics import apply_linear_bias, compute_doppler


class PredictDoppler(nn.Module):
    """
    A strictly stateless pipeline for inference.
    Safe to instantiate once, compile once, and share across API threads.
    """

    def __init__(
        self,
        state_def,
        bias_group=None,
        surrogate_weights_path: str = "models/mldsgp4_example_model.pth",
    ) -> None:
        super().__init__()
        # Structural metadata
        self.state_def = state_def
        self.bias_group = bias_group

        # Surrogate model initialization
        self.use_pretrained_model = surrogate_weights_path is not None
        if self.use_pretrained_model:
            self.surrogate_model = FunctionalMLdSGP4()
            self.surrogate_model.load_state_dict(torch.load(surrogate_weights_path))
            self.surrogate_model.eval()  # Ensure deterministic behavior for inference
            self.surrogate_model.requires_grad_(
                False
            )  # Enforce statelessness by freezing gradients
            self.model = self.surrogate_model
        else:
            self.model = sgp4_propagate

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

        # Branch based on initialization
        if self.use_pretrained_model:
            # The surrogate model requires the standard propagator to wrap around
            sat_pos, sat_vel = self.model(
                tsince=tsince, sgp4_propagate=sgp4_propagate, **sgp4_args
            )
        else:
            # The standard analytical propagator runs natively
            sat_pos, sat_vel = self.model(tsince=tsince, **sgp4_args)

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
