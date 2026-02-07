import torch
import torch.nn as nn
from torch.func import jacfwd, functional_call
from modules import SGP4Layer, DopplerSensor

class OrbitSystem(nn.Module):
    def __init__(self, state_def, init_tle, station_data):
        super().__init__()
        self.state_def = state_def
        
        # Layers
        self.propagator = SGP4Layer(init_tle)
        self.sensor = DopplerSensor(station_data['pos'], station_data['vel'], 435e6)
        
        # Truth-Tuned System Parameters (NOT in the daily state vector x)
        # These are fixed for a specific OD run, but could be trained in a separate loop.
        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, state_vector, t_minutes, contact_indices):
        # 1. Decode State Vector
        sgp4_inputs, pass_biases = self.state_def.unpack(state_vector)
        
        # 2. Propagate (Physics)
        # Note: sgp4_inputs needs to be (1, 7) for broadcasting if we want that,
        # but here we are doing single-sat OD, so (7,) is fine.
        pos, vel = self.propagator(sgp4_inputs, t_minutes)
        
        # 3. Sensor Model (Physics)
        preds = self.sensor(pos, vel)
        
        # 4. Apply State Vector Biases (Per-Pass)
        if pass_biases is not None:
            # contact_indices maps time t -> pass ID
            preds = preds + pass_biases[contact_indices]
            
        # 5. Apply System Corrections
        preds = preds + self.global_bias
        
        return preds

    def get_jacobian(self, state_vector, t_minutes, contact_indices):
        """
        Calculates d(Output)/d(StateVector) using Forward Mode AD.
        """
        # Functional wrapper: fix data args, differentiate wrt x
        def model_func(x):
            return self.forward(x, t_minutes, contact_indices)
            
        return jacfwd(model_func)(state_vector)