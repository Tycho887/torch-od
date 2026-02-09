import torch
import torch.nn as nn
from torch.func import jacfwd

from src.OtherLayer import Residual, SensorStack
from src.physicsLayers import GeometryLayer, SGP4Layer
from src.stateDefinitionLayer import DynamicStateDefinition


class BaseOrbitSystem(nn.Module):
    def __init__(self, init_tle, fit_keys, sensors_dict, ground_stations=None) -> None:
        super().__init__()
        self.state_def = DynamicStateDefinition(
            orbital_keys=fit_keys, sensors_dict=sensors_dict, init_tle=init_tle
        )
        self.propagator = SGP4Layer(init_tle=init_tle)
        self.geometry_layer = GeometryLayer()  # Stateless physics
        self.sensor_stack = SensorStack(sensors_dict)
        self.residual_layer = Residual()

        if ground_stations is not None:
            self.ground_stations = nn.ModuleDict(ground_stations)
        else:
            self.ground_stations = nn.ModuleDict({})

    def _prepare_batches(self, observations):
        """
        Flattens observation timestamps into a single global timeline.

        Returns:
            t_all: (N_total,) tensor of all timestamps.
            sensor_global_layout: Dict[sensor_name -> indices in t_all].
                                  Used to slice the Global Geometry tensors.
            station_global_layout: Dict[station_id -> indices in t_all].
                                   Used to propagate specific Ground Stations.
        """
        all_ts = []
        sensor_global_layout = {}
        station_global_layout = {}

        current_idx = 0

        for name in sorted(observations.keys()):
            if name not in self.sensor_stack.sensors:
                continue

            obs = observations[name]
            t_tensor = obs["t"]
            n_obs = t_tensor.shape[0]

            # 1. Global Layout Indices
            # (Where this sensor's data lives in the flattened vectors)
            global_indices = torch.arange(current_idx, current_idx + n_obs)
            sensor_global_layout[name] = global_indices

            # Store Time
            all_ts.append(t_tensor)

            # 2. Map Station Indices
            sensor_module = self.sensor_stack.sensors[name]
            st_id = getattr(sensor_module, "station_id", None)

            if st_id and st_id in self.ground_stations:
                if st_id not in station_global_layout:
                    station_global_layout[st_id] = []
                station_global_layout[st_id].append(global_indices)

            current_idx += n_obs

        t_all = torch.cat(all_ts)
        return t_all, sensor_global_layout, station_global_layout

    def forward(self, state_vector, observations) -> torch.Tensor:
        # A. Unpack State
        sgp4_inputs, sensor_param_dict = self.state_def.unpack(state_vector)

        # B. Batch Prep (Get layout indices)
        t_all, sensor_global_layout, station_global_layout = self._prepare_batches(
            observations
        )

        # C. Propagate Orbit (Global Time)
        pos_sat, vel_sat = self.propagator(sgp4_inputs, t_all)

        # D. Propagate Ground Stations (Mapped by Station Layout)
        pos_st = torch.zeros_like(pos_sat)
        vel_st = torch.zeros_like(vel_sat)

        for st_id, idx_list in station_global_layout.items():
            global_indices = torch.cat(idx_list)
            t_station = t_all[global_indices]

            p, v = self.ground_stations[st_id](t_station)

            pos_st.index_put_((global_indices,), p)
            vel_st.index_put_((global_indices,), v)

        # E. Geometry (Stateless)
        geo_data = self.geometry_layer(pos_sat, vel_sat, pos_st, vel_st)

        # F. Sensor Stack
        # Pass the layout map so sensors can slice their specific geometry
        preds = self.sensor_stack(
            geometry_data=geo_data,
            sensor_params_dict=sensor_param_dict,
            sensor_global_layout=sensor_global_layout,
            observations=observations,
        )

        return preds

    def get_jacobian(self, state_vector, observations) -> torch.Tensor:
        def model_func(x):
            return self.forward(x, observations)

        return jacfwd(model_func)(state_vector)
