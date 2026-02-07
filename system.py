import torch
import torch.nn as nn
from torch.func import jacfwd
from modules import SGP4Layer, DopplerSensor, RangeSensor

class MultiSensorSystem(nn.Module):
    def __init__(self, state_def, init_tle, sensor_dict):
        """
        sensor_dict: {'doppler': doppler_instance, 'range': range_instance}
        """
        super().__init__()
        self.state_def = state_def
        self.propagator = SGP4Layer(init_tle)
        self.sensors = nn.ModuleDict(sensor_dict) # Register sensors properly

    def forward(self, state_vector, observations):
        """
        observations: Dict containing data for this batch
           {
             'doppler': (t_tensor, idx_tensor),
             'range': (t_tensor, idx_tensor)
           }
        """
        # 1. Unpack State
        sgp4_inputs, bias_dict = self.state_def.unpack(state_vector)
        
        results = []
        
        # 2. Iterate over available data types in this batch
        # (Using a sorted key list ensures deterministic output order for Jacobian)
        for name in sorted(observations.keys()):
            if name not in self.sensors:
                continue
                
            t_minutes, contact_indices = observations[name]
            
            # A. Propagate (specific to these timestamps)
            pos, vel = self.propagator(sgp4_inputs, t_minutes)
            
            # B. Sensor Prediction
            # We explicitly pass the bias slice for *this* sensor
            sensor_bias = bias_dict.get(name) 
            pred = self.sensors[name](pos, vel, sensor_bias, contact_indices)
            
            results.append(pred)
            
        # 3. Stack all measurements into one long vector
        return torch.cat(results)

    def get_jacobian(self, state_vector, observations):
        def model_func(x):
            return self.forward(x, observations)
        
        # Returns (Total_Measurements, Total_State_Dim)
        # Block structure: [ [J_dop_orbit, I_dop_bias, 0], 
        #                    [J_rng_orbit, 0, I_rng_bias] ]
        return jacfwd(model_func)(state_vector)