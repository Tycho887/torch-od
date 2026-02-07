from dsgp4.tle import TLE
from modules import OrbitalParameterLayer, SGP4Layer, DopplerSensor
import torch.nn as nn

class OrbitSystem(nn.Module):
    def __init__(self, tle, station_data, num_contacts):
        super().__init__()
        # 1. The "Weights" (Orbit)
        self.orbit = OrbitalParameterLayer(tle, trainable_keys=['n', 'ma', 'bstar'])
        
        # 2. The "Physics" (Propagator)
        self.propagator = SGP4Layer(tle)
        
        # 3. The "Head" (Sensor)
        self.sensor = DopplerSensor(
            station_data['pos'], 
            station_data['vel'], 
            center_freq=435e6, 
            num_contacts=num_contacts
        )

    def forward(self, t_minutes, contact_indices):
        # A. Get State
        params = self.orbit()
        
        # B. Propagate
        pos, vel = self.propagator(params, t_minutes)
        
        # C. Measure
        preds = self.sensor(pos, vel, contact_indices)
        
        return preds