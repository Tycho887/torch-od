import torch
import torch.nn as nn


class SensorStack(nn.Module):
    def __init__(self, sensors_dict: dict[str, nn.Module]) -> None:
        super().__init__()
        self.sensors = nn.ModuleDict(sensors_dict)

    def forward(
        self, geometry_data, sensor_params_dict, sensor_global_layout, observations
    ) -> torch.Tensor:
        results = []

        # Deterministic order
        for name in sorted(self.sensors.keys()):
            if name not in sensor_global_layout:
                continue

            sensor = self.sensors[name]

            # 1. Slice Global Geometry (Using the layout indices)
            global_idx = sensor_global_layout[name]

            sensor_geometry = {
                key: val[global_idx]
                for key, val in geometry_data.items()
                if torch.is_tensor(val)
                and val.shape[0] == geometry_data["range"].shape[0]
            }

            # 2. Get Sensor Parameters
            params = sensor_params_dict[name]

            # 3. Get Pass Indices (User provided metadata)
            # These are strictly 0, 1, 2... referring to the bias index
            obs_packet = observations[name]
            pass_indices = obs_packet.get("pass_indices", None)

            # --- SAFETY CHECK ---
            # Ensure the user didn't request Pass #5 for a sensor with only 2 passes
            if pass_indices is not None and hasattr(sensor, "n_passes"):
                max_pass_idx = pass_indices.max().item()
                if max_pass_idx >= sensor.n_passes:
                    raise ValueError(
                        f"Sensor '{name}' Error: Observation requests bias for Pass #{max_pass_idx}, "
                        f"but sensor is configured with num_passes={sensor.n_passes}."
                    )

            # 4. Sensor Forward
            pred = sensor(sensor_geometry, params, pass_indices)

            print(f"The predicted shape is: {pred.shape}")

            results.append(pred)

        print("The length is:", len(results))
        print("The shape is:", (results[0].shape))
        print("The shape is:", (results[1].shape))

        return torch.cat(results)


class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions, observations) -> torch.Tensor:
        meas_list = []
        # Flatten measurements in same sorted order
        for name in sorted(observations.keys()):
            # Assuming observations[name] has 'data' or similar tensor
            # Adjust key based on your exact dict structure (e.g., 'measurements', 'data', or the value itself)
            if "data" in observations[name]:
                meas_list.append(observations[name]["data"])
            # If your obs_data structure is just {'t':..., 'indices':...} you might need to pass measurements separately
            # or include them in obs_data. Assuming they are in obs_data for now:

        # If measurements aren't in obs_data, this layer needs a separate 'measurements' dict input.
        # For now, returning difference assuming measurements are extractable.
        measurements_flat = torch.cat(meas_list)
        return predictions - measurements_flat
