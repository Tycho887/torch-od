import torch
import torch.nn as nn
from torch.func import jacfwd

from batch_indexer import VectorizedGroundStations

# from src.OtherLayer import Residual, SensorStack
from src.physicsLayers import GeometryLayer, SGP4Layer
from src.stateDefinitionLayer import DynamicStateDefinition


class OrbitSystem(nn.Module):
    def __init__(self, init_tle, fit_keys, sensors_dict, indexer):
        super().__init__()
        self.state_def = DynamicStateDefinition(fit_keys, sensors_dict, init_tle)
        self.propagator = SGP4Layer(init_tle)

        # Extract station objects for the vectorized layer
        stations = [s for s in sensors_dict.values() if hasattr(s, "pos_ecef")]
        # Note: In real code, handle duplicates if multiple sensors share a station
        self.station_layer = VectorizedGroundStations(stations)

        self.geometry = GeometryLayer()
        self.indexer = indexer  # The BatchIndexer instance

    def forward(self, state_vector):
        # 1. Unpack State
        # sgp4_params: (7,), sensor_biases: (P_total,)
        sgp4_params, sensor_biases_flat = self.state_def.unpack_to_flat(state_vector)

        # 2. Propagate Satellite (N, 3)
        # We propagate for every timestamp in the flattened timeline
        pos_sat, vel_sat = self.propagator(sgp4_params, self.indexer.t_all)

        # 3. Propagate Stations (N, 3)
        # We pass the station indices so we get the correct station for row i
        pos_st, vel_st = self.station_layer(
            self.indexer.t_all, self.indexer.station_idx_tensor
        )

        # 4. Compute Geometry (N, 3)
        # This calculates Range AND RangeRate (and others) for ALL measurements
        geo = self.geometry(pos_sat, vel_sat, pos_st, vel_st)

        # Stack Physics options: [Range, RangeRate, ...] -> (N, K_physics)
        # Column 0 = Range, Column 1 = Doppler
        c = 299792.458
        nominal_freq = 2.2e9  # Ideally passed from indexer or normalized

        # NOTE: You may need normalization here.
        # Doppler is -(RR/c)*f. Range is km.
        # To strictly vectorize, we compute raw physics here.
        physics_stack = torch.stack(
            [
                geo["range"].squeeze(),  # Index 0
                -(geo["range_rate"].squeeze() / c) * nominal_freq,  # Index 1 (Doppler)
            ],
            dim=1,
        )

        # 5. Physics Selection (The "Boolean Selection")
        # We use gather to pick the column corresponding to the sensor type
        # phys_selector is (N, 1) indices
        base_prediction = torch.gather(
            physics_stack, 1, self.indexer.phys_selector
        ).squeeze()

        # 6. Apply Biases (Sparse Matrix Multiplication)
        # H_param is (N, P). biases is (P,).
        # Result is (N,), adding the specific bias for that specific pass/sensor
        bias_correction = torch.sparse.mm(
            self.indexer.H_param, sensor_biases_flat.unsqueeze(1)
        ).squeeze()

        total_prediction = base_prediction + bias_correction

        return total_prediction
