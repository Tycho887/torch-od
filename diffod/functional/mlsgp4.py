import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from diffod.functional.sgp4 import sgp4_propagate


class FunctionalMLdSGP4(nn.Module):
    def __init__(
        self,
        normalization_R=6958.137,
        normalization_V=7.947155867983262,
        hidden_size=35,
        input_correction=1e-2,
        output_correction=0.8,
        dtype=torch.float64, # High precision standard for orbital mechanics
        device=torch.device("cpu"),
    ):
        super().__init__()
        
        # 1. Device and Dtype Initialization
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.fc1 = nn.Linear(6, hidden_size, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.fc3 = nn.Linear(hidden_size, 6, **factory_kwargs)
        self.fc4 = nn.Linear(6, hidden_size, **factory_kwargs)
        self.fc5 = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.fc6 = nn.Linear(hidden_size, 6, **factory_kwargs)

        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.normalization_R = normalization_R
        self.normalization_V = normalization_V
        
        self.input_correction = Parameter(torch.full((6,), input_correction, **factory_kwargs))
        self.output_correction = Parameter(torch.full((6,), output_correction, **factory_kwargs))

    def forward(
        self,
        tsince: torch.Tensor,
        **sgp4_kwargs,
    ):
        """
        Forward pass using a purely functional SGP4 propagator.
        """
        # 2. Dynamic Memory Placement
        # Read the current device in case the model was moved via .to(device)
        current_device = self.input_correction.device
        device_type = current_device.type if current_device.type in ['cuda', 'cpu', 'xpu'] else 'cpu'

        # Ensure inputs are on the same device as the model parameters
        tsince = tsince.to(current_device)
        sgp4_kwargs = {
            k: v.to(current_device) if isinstance(v, torch.Tensor) else v 
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

        # 3. Autocasting (Input Correction)
        # We isolate the NN in autocast to allow fp16/bf16 acceleration, 
        # while keeping the orbital state in fp64 to prevent catastrophic cancellation.
        with torch.autocast(device_type=device_type, enabled=True):
            # Pass original tensor directly; autocast handles the downcasting
            x = self.leaky_relu(self.fc1(x0))
            x = self.leaky_relu(self.fc2(x)) # Fixed: Was previously taking x0_fp32, overwriting x
            nn_correction = self.tanh(self.fc3(x))
            
        # Cast NN output back to the original high-precision dtype
        x_corrected = x0 * (1 + self.input_correction * nn_correction.to(x0.dtype))

        # Update kwargs purely functionally
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

        # Execute custom functional propagation (Maintained in fp64)
        pos, vel = sgp4_propagate(tsince, **updated_kwargs)

        # 4. Autocasting (Output Correction)
        x_out = torch.cat(
            (pos / self.normalization_R, vel / self.normalization_V), dim=-1
        )

        with torch.autocast(device_type=device_type, enabled=True):
            x = self.leaky_relu(self.fc4(x_out))
            x = self.leaky_relu(self.fc5(x))
            nn_out_correction = self.tanh(self.fc6(x))
            
        # Apply the final correction in high precision
        x_final_norm = x_out * (1 + self.output_correction * nn_out_correction.to(x_out.dtype))

        # Denormalize to return physical state vectors
        pos_corrected = x_final_norm[..., :3] * self.normalization_R
        vel_corrected = x_final_norm[..., 3:] * self.normalization_V

        return pos_corrected, vel_corrected

    def load_model(
        self,
        path: str = "models/mldsgp4_example_model.pth",
    ):
        """
        Loads the pre-trained ESA weights.
        """
        # Dynamically load to the module's current device
        current_device = self.input_correction.device
        state_dict = torch.load(
            path, map_location=current_device, weights_only=True
        )
        self.load_state_dict(state_dict)
        self.eval()