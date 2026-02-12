import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from dsgp4.sgp4init import sgp4init, sgp4
from dsgp4.util import get_gravity_constants
from diffod.functional.tle import get_tle_epoch_unix
from diffod.state import StateDefinition

class MLdSGP4(nn.Module):
    def __init__(
        self, 
        base_tle,
        normalization_R=6958.137, 
        normalization_V=7.947155867983262, 
        hidden_size=100, 
        input_correction=1e-2, 
        output_correction=0.8
    ):
        """
        Differentiable ML-dSGP4 that accepts an external State Vector.
        """
        super().__init__()
        
        # Store base TLE for static parameters (B*, catalog number, etc.)
        self.base_tle = base_tle
        
        # Neural Network Layers (Same as original)
        self.fc1 = nn.Linear(6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 6)
        
        self.fc4 = nn.Linear(6, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, 6)
        
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        # Normalization Constants
        self.norm_R = normalization_R
        self.norm_V = normalization_V
        
        # Learnable Scaling Factors
        self.input_correction = Parameter(input_correction * torch.ones((6,)))
        self.output_correction = Parameter(output_correction * torch.ones((6,)))

    def load_model(self, path="models/mldsgp4_example_model.pth", device='cpu') -> None:
        """
        This method loads a model from a file.

        Parameters:
        ----------------
        path (``str``): path to the file where the model is stored.
        device (``str``): device where the model will be loaded. Default is 'cpu'.
        """
        self.load_state_dict(state_dict=torch.load(f=path,map_location=torch.device(device=device)))
        self.eval()

    def forward(self, t_unix: torch.Tensor, x_state: torch.Tensor, state_def: StateDefinition, gravity_constant="wgs-84") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Functional Forward Pass.
        
        Args:
            t_unix: (N,) Tensor of UNIX timestamps.
            x_state: (M,) The differentiable state vector.
            state_def: The StateDefinition object mapping x_state to Keplerian elements.
            
        Returns:
            pos, vel: (N, 3) Corrected Position and Velocity in KM and KM/S.
        """
        
        # -----------------------------------------------------------
        # 1. Input Construction (Assemble x0 from State Vector)
        # -----------------------------------------------------------
        # We need the 6 Keplerian elements: [e, argp, i, M, n, raan]
        # We assume single-satellite fitting, so we create a (1, 6) tensor.
        
        def val(key, default):
            if state_def and key in state_def.map_param_to_idx:
                return x_state[state_def.map_param_to_idx[key]]
            return default

        # Extract values (preserving gradients if they are in x_state)
        e0 = val('eccentricity', self.base_tle._ecco)
        w0 = val('argument_of_perigee', self.base_tle._argpo)
        i0 = val('inclination', self.base_tle._inclo)
        M0 = val('mean_anomaly', self.base_tle._mo)
        n0 = val('mean_motion', self.base_tle._no_kozai)
        O0 = val('raan', self.base_tle._nodeo)
        
        # Stack into (1, 6) tensor for the Network
        # Ensure they are tensors for the graph
        x0 = torch.stack([
            torch.as_tensor(e0), torch.as_tensor(w0), torch.as_tensor(i0),
            torch.as_tensor(M0), torch.as_tensor(n0), torch.as_tensor(O0)
        ]).unsqueeze(0).to(x_state.device if x_state is not None else 'cpu')

        # -----------------------------------------------------------
        # 2. Input Correction (NN #1)
        # -----------------------------------------------------------
        # Network predicts multiplicative correction factors
        h = self.leaky_relu(self.fc1(x0))
        h = self.leaky_relu(self.fc2(h))
        delta_in = self.tanh(self.fc3(h))
        
        # x_corrected = x0 * (1 + scale * delta)
        x_corr = x0 * (1.0 + self.input_correction * delta_in)
        
        # Unpack corrected values for SGP4
        # Shape is (1, 6), so we take index 0
        e_c, w_c, i_c, M_c, n_c, O_c = x_corr[0]

        # -----------------------------------------------------------
        # 3. Functional Propagation (SGP4)
        # -----------------------------------------------------------
        # Initialize SGP4 constants using the CORRECTED elements
        whichconst = get_gravity_constants(gravity_constant)
        
        # We assume self.base_tle is a dSGP4 TLE object
        # Note: We must pass a TLE object to sgp4init, but we don't want to mutate self.base_tle.
        # However, dSGP4 reads from the object passed to 'satellite'. 
        # We can pass the static base_tle, but override the physics parameters via arguments.
        
        sgp4init(
            whichconst=whichconst,
            opsmode='i',
            satn=self.base_tle.satellite_catalog_number,
            epoch=(self.base_tle._jdsatepoch + self.base_tle._jdsatepochF) - 2433281.5,
            xbstar=val('b_star', self.base_tle.bstar), # B* might be in state vector too
            xndot=self.base_tle.mean_motion_first_derivative,
            xnddot=self.base_tle.mean_motion_second_derivative,
            xecco=e_c, xargpo=w_c, xinclo=i_c, xmo=M_c, xno_kozai=n_c, xnodeo=O_c,
            satellite=self.base_tle # This is used only for storage in standard dSGP4, 
                                    # but since we don't return the object, mutation doesn't matter 
                                    # as long as we don't reuse it concurrently in a way that races.
                                    # For pure functional safety, one could clone it.
        )
        
        # Time setup
        epoch_unix = get_tle_epoch_unix(self.base_tle)
        tsince_min = (t_unix - epoch_unix) / 60.0
        if tsince_min.ndim == 1:
            tsince_min = tsince_min.unsqueeze(0) # (1, Time)

        # Propagate
        state_teme = sgp4(self.base_tle, tsince_min) # (1, Time, 2, 3)
        
        # Flatten: (Time, 6) -> [Pos, Vel]
        pos_raw = state_teme[0, :, 0, :]
        vel_raw = state_teme[0, :, 1, :]
        
        # -----------------------------------------------------------
        # 4. Output Correction (NN #2)
        # -----------------------------------------------------------
        # Normalize: (Time, 6)
        x_out_norm = torch.cat([
            pos_raw / self.norm_R, 
            vel_raw / self.norm_V
        ], dim=1)
        
        h2 = self.leaky_relu(self.fc4(x_out_norm))
        h2 = self.leaky_relu(self.fc5(h2))
        delta_out = self.tanh(self.fc6(h2))
        
        # Apply correction
        x_final_norm = x_out_norm * (1.0 + self.output_correction * delta_out)
        
        # -----------------------------------------------------------
        # 5. Denormalize (Return Physical Units)
        # -----------------------------------------------------------
        # We return km and km/s so the physics layer (Doppler/Range) works correctly
        pos_final = x_final_norm[:, :3] * self.norm_R
        vel_final = x_final_norm[:, 3:] * self.norm_V
        
        return pos_final, vel_final