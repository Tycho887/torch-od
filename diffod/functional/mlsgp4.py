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
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.fc1 = nn.Linear(in_features=6, out_features=hidden_size, **factory_kwargs)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, **factory_kwargs)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=6, **factory_kwargs)
        self.fc4 = nn.Linear(in_features=6, out_features=hidden_size, **factory_kwargs)
        self.fc5 = nn.Linear(in_features=hidden_size, out_features=hidden_size, **factory_kwargs)
        self.fc6 = nn.Linear(in_features=hidden_size, out_features=6, **factory_kwargs)

        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.normalization_R = normalization_R
        self.normalization_V = normalization_V
        
        self.input_correction = Parameter(data=torch.full(size=(6,), fill_value=input_correction, **factory_kwargs))
        self.output_correction = Parameter(data=torch.full(size=(6,), fill_value=output_correction, **factory_kwargs))

    def forward(
        self,
        tsince: torch.Tensor,
        **sgp4_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using a purely functional SGP4 propagator.
        """
        # 2. Dynamic Memory Placement
        # Read the current device in case the model was moved via .to(device)
        current_device = self.input_correction.device
        current_dtype = self.input_correction.dtype

        # Ensure inputs match BOTH device and precision
        tsince = tsince.to(device=current_device, dtype=current_dtype)
        sgp4_kwargs = {
            k: v.to(device=current_device, dtype=current_dtype) if isinstance(v, torch.Tensor) else v 
            for k, v in sgp4_kwargs.items()
        }

        # Stack parameters
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

        x = self.leaky_relu(self.fc1(x0))
        x = self.leaky_relu(self.fc2(x))
        # nn_correction = self.tanh(self.fc3(x))
            
        x = x0 * (1+self.input_correction*self.tanh(self.fc3(x)))

        # Update kwargs purely functionally
        updated_kwargs = {k: v for k, v in sgp4_kwargs.items()}
        updated_kwargs.update(
            {
                "ecco": x[0],
                "argpo": x[1],
                "inclo": x[2],
                "mo": x[3],
                "no_kozai": x[4],
                "nodeo": x[5],
            }
        )

        # Execute custom functional propagation (Maintained in fp64)
        pos, vel = sgp4_propagate(tsince=tsince, **updated_kwargs)

        x_out = torch.cat(
            (pos / self.normalization_R, vel / self.normalization_V), dim=-1
        )

        x=self.leaky_relu(self.fc4(x_out))
        x=self.leaky_relu(self.fc5(x))
        x=x_out*(1+self.output_correction*self.tanh(self.fc6(x)))

        # Denormalize to return physical state vectors
        pos_corrected = x[:3] * self.normalization_R
        vel_corrected = x[3:] * self.normalization_V

        # print(pos_corrected, vel_corrected)

        return pos_corrected, vel_corrected

    def load_model(self, path: str = "models/mldsgp4_example_model.pth"):
        """
        Loads weights and ensures they match the model's precision.
        """
        current_device = self.input_correction.device
        current_dtype = self.input_correction.dtype # Detect if we are FP64
        
        # Load the state dict
        state_dict = torch.load(path, map_location=current_device, weights_only=True)
        
        # Manually cast every tensor in the state_dict to the target dtype
        for key in state_dict:
            state_dict[key] = state_dict[key].to(dtype=current_dtype)
        
        print(state_dict)

        self.load_state_dict(state_dict)
        self.eval()