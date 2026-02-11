import torch
import torch.nn as nn


class BatchIndexer:
    """
    Constructs the static indexing and mapping matrices
    to linearize the OD problem.
    """

    def __init__(self, sensors_dict, observations):
        self.sensors_dict = sensors_dict

        # 1. Sort and Flatten Observations
        # We need a strict global ordering. Sorting by Time is usually best for propagation continuity.
        flat_obs = []
        for sensor_name, data in observations.items():
            t = data["t"]
            n_obs = t.shape[0]

            # Extract metadata
            # We assume data has 'pass_indices' mapping meas -> pass_id (0..M)
            pass_idx = data.get("pass_indices", torch.zeros(n_obs, dtype=torch.long))

            # Store tuples: (time, sensor_name, original_val, pass_idx)
            for i in range(n_obs):
                flat_obs.append(
                    {
                        "t": t[i],
                        "val": data["data"][i] if "data" in data else 0.0,
                        "sensor": sensor_name,
                        "pass_idx": pass_idx[i],
                        "orig_idx": i,
                    }
                )

        # Sort by time
        flat_obs.sort(key=lambda x: x["t"])
        self.n_total = len(flat_obs)

        # 2. Build Tensors
        self.t_all = torch.stack([x["t"] for x in flat_obs])
        self.obs_all = torch.stack([x["val"] for x in flat_obs])

        # 3. Build Station Indexer
        # We need to know which station corresponds to row 'i'
        # We map station_ids to integers 0..K
        station_ids = sorted(list(set([s.station_id for s in sensors_dict.values()])))
        self.station_id_map = {sid: i for i, sid in enumerate(station_ids)}

        station_indices = []
        for x in flat_obs:
            s_name = x["sensor"]
            st_id = sensors_dict[s_name].station_id
            station_indices.append(self.station_id_map[st_id])

        self.station_idx_tensor = torch.tensor(station_indices, dtype=torch.long)

        # 4. Build Physics Selector (The "Boolean Matrix" concept)
        # 0: Range (km), 1: Doppler (km/s), 2: Az, 3: El ...
        # We map sensors to physics columns
        self.physics_map = {"RadarSensor": 0, "DopplerSensor": 1}

        phys_indices = []
        for x in flat_obs:
            s_obj = sensors_dict[x["sensor"]]
            # You might need a cleaner way to identify sensor type than class name
            s_type = s_obj.__class__.__name__
            phys_indices.append(self.physics_map.get(s_type, 0))

        self.phys_selector = torch.tensor(phys_indices, dtype=torch.long).unsqueeze(1)

        # 5. Build Parameter Mapping Matrix (Sparse)
        # This maps measurements to the flattened vector of sensor biases.
        # Structure: Rows = Measurements (N), Cols = Total Sensor Params (P)

        # We need to calculate the global offset for each sensor's params
        self.param_slices = {}
        current_offset = 0
        indices = []
        values = []

        # We iterate strictly in the order defined by the StateDefinition
        for name, sensor in sensors_dict.items():
            n_params = sensor.num_params
            self.param_slices[name] = (current_offset, n_params)

            # Find all measurements belonging to this sensor in our sorted list
            # (In a real impl, you'd optimize this lookup)
            for row_idx, x in enumerate(flat_obs):
                if x["sensor"] == name:
                    # Pass Index determines which parameter *within* the sensor's block
                    # e.g. Param 5 global = Offset (2) + Pass_Idx (3)
                    local_p_idx = x["pass_idx"].item()

                    # Safety: Ensure we don't exceed sensor params
                    if local_p_idx < n_params:
                        col_idx = current_offset + local_p_idx
                        indices.append([row_idx, col_idx])
                        values.append(1.0)

            current_offset += n_params

        # Create Sparse Matrix (N x P_total_biases)
        i = torch.LongTensor(indices).t()
        v = torch.FloatTensor(values)
        self.H_param = torch.sparse_coo_tensor(
            i, v, size=(self.n_total, current_offset)
        )

    def to(self, device):
        self.t_all = self.t_all.to(device)
        self.obs_all = self.obs_all.to(device)
        self.station_idx_tensor = self.station_idx_tensor.to(device)
        self.phys_selector = self.phys_selector.to(device)
        self.H_param = self.H_param.to(device)
        return self


class VectorizedGroundStations(nn.Module):
    def __init__(self, stations_list):
        super().__init__()
        # 1. Stack ECEF positions: (K_stations, 3)
        # Ensure the order matches self.station_id_map from the Indexer
        sorted_stations = sorted(stations_list, key=lambda s: s.station_id)

        ecefs = [s.pos_ecef.detach().cpu() for s in sorted_stations]  # (1,3)
        self.register_buffer("stations_ecef", torch.cat(ecefs, dim=0))  # (K, 3)

        # Assume constant Earth rotation for all
        self.w_earth = 7.2921151467e-5

        # Store GMST0s if they differ, or handle time diffs appropriately
        # For simplicity, assuming one epoch or handling t_diff internally
        # (You might need a tensor of gmst0 if stations have different epochs)
        self.gmst0 = sorted_stations[0].gmst0
        self.epoch_tai = sorted_stations[0].epoch_tai

    def forward(self, t_tai, station_indices):
        """
        Args:
            t_tai: (N,) timestamps
            station_indices: (N,) index of which station to use for each time
        """
        # 1. Select ECEF positions for all N measurements
        # (N, 3)
        pos_ecef_N = self.stations_ecef[station_indices]  # pyright: ignore[reportIndexIssue]

        # 2. Compute Rotation Matrix for all N
        t_diff = t_tai - self.epoch_tai
        theta = self.gmst0 + self.w_earth * t_diff

        c = torch.cos(theta)
        s = torch.sin(theta)

        # 3. Apply Rotation (Vectorized)
        # x_teme = x_ec * c - y_ec * s
        # y_teme = x_ec * s + y_ec * c
        x_ec, y_ec, z_ec = pos_ecef_N[:, 0], pos_ecef_N[:, 1], pos_ecef_N[:, 2]

        x_teme = x_ec * c - y_ec * s
        y_teme = x_ec * s + y_ec * c
        z_teme = z_ec  # In TEME/ECEF z is aligned

        pos_teme = torch.stack([x_teme, y_teme, z_teme], dim=1)

        # 4. Velocity (Cross product logic)
        w = self.w_earth
        vx = -w * y_teme
        vy = w * x_teme
        vz = torch.zeros_like(x_teme)

        vel_teme = torch.stack([vx, vy, vz], dim=1)

        return pos_teme, vel_teme
