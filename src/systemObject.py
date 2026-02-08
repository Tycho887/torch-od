import torch
import torch.nn as nn
from torch.func import jacfwd
from src.physicsLayers import SGP4Layer
from src.stateDefinitionLayer import DynamicStateDefinition

class BaseOrbitSystem(nn.Module):
    def __init__(self, init_tle, fit_keys, sensors_dict):
        super().__init__()
        self.state_def = DynamicStateDefinition(fit_keys, sensors_dict, init_tle)
        self.propagator = SGP4Layer(init_tle)
        self.sensors = nn.ModuleDict(sensors_dict)

    def forward(self, state_vector, observations):
        """
        observations: {
           'doppler': {'t': ..., 'indices': ...},
           'radar':   {'t': ..., 'indices': ...}
        }
        """
        # 1. Unpack Global State
        sgp4_inputs, sensor_param_dict = self.state_def.unpack(state_vector)
        
        all_residuals = []
        
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
            all_residuals.append(preds)
            
        return torch.cat(all_residuals)

    def get_jacobian(self, state_vector, observations):
        def model_func(x):
            return self.forward(x, observations)
        return jacfwd(model_func)(state_vector)

# import torch
# import torch.nn as nn
# from torch.func import jacfwd
# from src.physicsLayers import SGP4Layer, GeometryLayer
# from src.stateDefinitionLayer import StateDefinition, DynamicStateDefinition
# from src.sensorLayers import ComplexTelescope

# class BaseOrbitSystem(nn.Module):
#     def __init__(self, init_tle, fit_keys, sensors_dict):
#         super().__init__()
#         # Auto-wire the state vector based on the sensors provided
#         self.state_def = DynamicStateDefinition(fit_keys, sensors_dict, init_tle)
#         self.propagator = SGP4Layer(init_tle)
#         self.sensors = nn.ModuleDict(sensors_dict)
        
#         # We assume station data is embedded in the sensors or passed differently
#         # (removed SharedGeometry for clarity, can be re-added if station is shared)

#     def forward(self, state_vector, observations):
#         """
#         observations: Dict of Dicts
#         {
#            'telescope': {
#                't': tensor([0.1, 0.2...]), 
#                'indices': tensor([0, 0...]),
#                'meta': ... 
#            },
#            'doppler': {
#                't': tensor([0.1, 0.15...]), ...
#            }
#         }
#         """
#         # 1. Decode State
#         sgp4_inputs, sensor_param_dict = self.state_def.unpack(state_vector)
        
#         all_residuals = []
        
#         # 2. Process each Sensor Independently
#         for name, obs_packet in observations.items():
#             if name not in self.sensors:
#                 continue
            
#             sensor = self.sensors[name]
            
#             # A. Dynamic Propagation (Syncs to THIS sensor's time)
#             t_sensor = obs_packet['t']
#             pos, vel = self.propagator(sgp4_inputs, t_sensor)
            
#             # B. Geometry Calculation
#             # (Ideally, pass pos/vel to a geometry helper, or let sensor handle it)
#             # For generality, let's create a geometry packet:
#             geo_data = {'rel_pos': pos - sensor.station_pos, 'rel_vel': vel} 
            
#             # C. Sensor Forward Pass
#             # We pass ONLY the parameters belonging to this sensor
#             my_params = sensor_param_dict[name]
#             my_meta = obs_packet['indices'] # Or generic metadata
            
#             preds = sensor(geo_data, my_params, my_meta)
            
#             all_residuals.append(preds)
            
#         return torch.cat(all_residuals)

#     def get_jacobian(self, state_vector, observations):
#         def model_func(x):
#             return self.forward(x, observations)
        
#         # Returns (Total_Measurements, Total_State_Dim)
#         # Block structure: [ [J_dop_orbit, I_dop_bias, 0], 
#         #                    [J_rng_orbit, 0, I_rng_bias] ]
#         return jacfwd(model_func)(state_vector)
