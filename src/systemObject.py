import torch
import torch.nn as nn
from torch.func import jacfwd
from src.physicsLayers import SGP4Layer
from src.stateDefinitionLayer import DynamicStateDefinition

class BaseOrbitSystem(nn.Module):
    def __init__(self, init_tle, fit_keys, sensors_dict) -> None:
        super().__init__()
        self.state_def = DynamicStateDefinition(orbital_keys=fit_keys, sensors_dict=sensors_dict, init_tle=init_tle)
        self.propagator = SGP4Layer(init_tle=init_tle)
        self.sensors = nn.ModuleDict(modules=sensors_dict)

    def forward(self, state_vector, observations) -> torch.Tensor:
        """
        observations: {
           'doppler': {'t': ..., 'indices': ...},
           'radar':   {'t': ..., 'indices': ...}
        }
        """
        # 1. Unpack Global State
        sgp4_inputs, sensor_param_dict = self.state_def.unpack(state_vector=state_vector)
        
        all_preds = []
        
        # 2. Iterate Sensor Data Packets
        # Sorting keys ensures deterministic order for Jacobian
        for name in sorted(observations.keys()):
            if name not in self.sensors:
                continue
            
            obs_packet = observations[name]
            sensor_module = self.sensors[name]
            
            # A. Propagate Orbit (Time-synced to this sensor)
            t_tensor = obs_packet['t']
            pos, vel = self.propagator(sgp4_inputs, t_tensor)
            
            # B. Package Geometry (Sensor will calculate derived physics)
            geo_data = {'pos': pos, 'vel': vel}
            
            # C. Sensor Forward Pass
            # Pass the specific slice of parameters for this sensor
            sensor_params = sensor_param_dict[name]
            contact_idx = obs_packet['indices']
            
            preds = sensor_module(geo_data, sensor_params, contact_idx)
            all_preds.append(preds)
            
        return torch.cat(tensors=all_preds)

    def residuals(self, state_vector, observations, measurements) -> torch.Tensor:
        preds = self.forward(state_vector=state_vector, observations=observations)
        
        if isinstance(measurements, dict):
            meas_list = []
            for name in sorted(observations.keys()):
                if name in self.sensors:
                    meas_list.append(measurements[name])
            measurements = torch.cat(tensors=meas_list)
            
        return preds - measurements

    def get_jacobian(self, state_vector, observations) -> torch.Tensor:
        def model_func(x) -> torch.Tensor:
            return self.forward(state_vector=x, observations=observations)
        return jacfwd(model_func)(state_vector)