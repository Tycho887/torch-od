from typing import Callable

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.parameter import Parameter


class FunctionalMLdSGP4(nn.Module):
    def __init__(
        self,
        normalization_R=6958.137,
        normalization_V=7.947155867983262,
        hidden_size=35,
        input_correction=1e-2,
        output_correction=0.8,
    ):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 6)
        self.fc4 = nn.Linear(6, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, 6)

        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.normalization_R = normalization_R
        self.normalization_V = normalization_V
        self.input_correction = Parameter(input_correction * torch.ones(6))
        self.output_correction = Parameter(output_correction * torch.ones(6))

    def forward(
        self,
        tsince: torch.Tensor,
        sgp4_propagate: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        **sgp4_kwargs,
    ):
        """
        Forward pass using a purely functional SGP4 propagator.
        Assumes `sgp4_propagate` returns a tuple of (position, velocity) tensors.
        """
        # 1. Stack parameters (using dim=-1 supports both scalar and batched inputs seamlessly)
        x0 = torch.stack(
            (
                sgp4_kwargs["ecco"],
                sgp4_kwargs["argpo"],
                sgp4_kwargs["inclo"],
                sgp4_kwargs["mo"],
                sgp4_kwargs["no_kozai"],
                sgp4_kwargs["nodeo"],
            ),
            dim=-1,
        )

        # 2. Input Correction Pass
        x = self.leaky_relu(self.fc1(x0))
        x = self.leaky_relu(self.fc2(x))
        x_corrected = x0 * (1 + self.input_correction * self.tanh(self.fc3(x)))

        # 3. Update kwargs purely functionally (no in-place modification of original dict)
        updated_kwargs = {k: v for k, v in sgp4_kwargs.items()}
        updated_kwargs.update(
            {
                "ecco": x_corrected[..., 0],
                "argpo": x_corrected[..., 1],
                "inclo": x_corrected[..., 2],
                "mo": x_corrected[..., 3],
                "no_kozai": x_corrected[..., 4],
                "nodeo": x_corrected[..., 5],
            }
        )

        # 4. Execute custom functional propagation
        pos, vel = sgp4_propagate(tsince, **updated_kwargs)

        # 5. Output Correction Pass
        # Normalize outputs before passing to the network
        x_out = torch.cat(
            (pos / self.normalization_R, vel / self.normalization_V), dim=-1
        )

        x = self.leaky_relu(self.fc4(x_out))
        x = self.leaky_relu(self.fc5(x))
        x_final_norm = x_out * (1 + self.output_correction * self.tanh(self.fc6(x)))

        # Denormalize to return physical state vectors (km, km/s)
        pos_corrected = x_final_norm[..., :3] * self.normalization_R
        vel_corrected = x_final_norm[..., 3:] * self.normalization_V

        return pos_corrected, vel_corrected

    def load_model(
        self,
        path: str = "models/mldsgp4_example_model.pth",
        device: torch.device | str = "cpu",
    ):
        """
        Loads the pre-trained ESA weights.
        Note: The checkpoint file must have been trained with or truncated to hidden_size=35.
        """
        state_dict = torch.load(
            path, map_location=torch.device(device), weights_only=True
        )
        self.load_state_dict(state_dict)
        self.eval()
