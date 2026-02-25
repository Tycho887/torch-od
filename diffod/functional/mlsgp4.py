from typing import Any
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from diffod.functional.sgp4 import sgp4_propagate
from dsgp4.mldsgp4 import mldsgp4


class FunctionalMLdSGP4(nn.Module):
    def __init__(
        self,
        normalization_R=6958.137,
        normalization_V=7.947155867983262,
        hidden_size=35,
        input_correction=1e-2,
        output_correction=0.8,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        
        # 1. Device and Dtype Initialization

        self.dtype = dtype
        self.device = device
        
        self.fc1 = nn.Linear(in_features=6, out_features=hidden_size, dtype=dtype, device=device)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=dtype, device=device)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=6, dtype=dtype, device=device)
        self.fc4 = nn.Linear(in_features=6, out_features=hidden_size, dtype=dtype, device=device)
        self.fc5 = nn.Linear(in_features=hidden_size, out_features=hidden_size, dtype=dtype, device=device)
        self.fc6 = nn.Linear(in_features=hidden_size, out_features=6, dtype=dtype, device=device)

        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.normalization_R = normalization_R
        self.normalization_V = normalization_V
        
        self.input_correction = Parameter(data=torch.full(size=(6,), fill_value=input_correction, dtype=dtype, device=device))
        self.output_correction = Parameter(data=torch.full(size=(6,), fill_value=output_correction, dtype=dtype, device=device))

    def forward(
        self,
        tsince: torch.Tensor,
        bstar: torch.Tensor,
        ndot: torch.Tensor,
        nddot: torch.Tensor,
        ecco: torch.Tensor,
        argpo: torch.Tensor,
        inclo: torch.Tensor,
        mo: torch.Tensor,
        no_kozai: torch.Tensor,
        nodeo: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using a purely functional SGP4 propagator.
        """
        # 1. Explicitly cast all inputs to the correct device and dtype
        tsince = tsince.to(device=self.device, dtype=self.dtype)
        bstar = bstar.to(device=self.device, dtype=self.dtype)
        ndot = ndot.to(device=self.device, dtype=self.dtype)
        nddot = nddot.to(device=self.device, dtype=self.dtype)
        ecco = ecco.to(device=self.device, dtype=self.dtype)
        argpo = argpo.to(device=self.device, dtype=self.dtype)
        inclo = inclo.to(device=self.device, dtype=self.dtype)
        mo = mo.to(device=self.device, dtype=self.dtype)
        no_kozai = no_kozai.to(device=self.device, dtype=self.dtype)
        nodeo = nodeo.to(device=self.device, dtype=self.dtype)

        # 2. Stack the 6 parameters targeted for neural network correction
        # Using dim=-1 ensures it works for both batched (N, 6) and unbatched (6,) inputs
        x0 = torch.stack((ecco, argpo, inclo, mo, no_kozai, nodeo), dim=-1)

        # 3. Compute Input Corrections
        x = self.leaky_relu(self.fc1(x0))
        x = self.leaky_relu(self.fc2(x))
        x_corrected = x0 * (1 + self.input_correction * self.tanh(self.fc3(x)))

        # 4. Execute functional propagation with *corrected* parameters
        # Slicing with [..., i] preserves batch dimensions safely
        pos, vel = sgp4_propagate(
            tsince=tsince,
            bstar=bstar,
            ndot=ndot,
            nddot=nddot,
            ecco=x_corrected[0],
            argpo=x_corrected[1],
            inclo=x_corrected[2],
            mo=x_corrected[3],
            no_kozai=x_corrected[4],
            nodeo=x_corrected[5]
        )

        # 5. Compute Output Corrections
        x_out = torch.cat(
            (pos / self.normalization_R, vel / self.normalization_V), dim=-1
        )

        x = self.leaky_relu(self.fc4(x_out))
        x = self.leaky_relu(self.fc5(x))
        x_final = x_out * (1 + self.output_correction * self.tanh(self.fc6(x)))

        # 6. Denormalize to return physical state vectors (km, km/s)
        pos_corrected = x_final[:, :3] * self.normalization_R
        vel_corrected = x_final[:, 3:] * self.normalization_V

        return pos_corrected, vel_corrected

    def load_model(self, path: str = "models/mldsgp4_example_model.pth"):
        """
        Loads weights and ensures they match the model's precision.
        """
        # Load the state dict
        state_dict = torch.load(path, map_location=self.device)
        
        # Manually cast every tensor in the state_dict to the target dtype
        for key in state_dict:
            state_dict[key] = state_dict[key].to(dtype=self.dtype)
            print(state_dict[key].type())
        self.load_state_dict(state_dict)
        self.eval()