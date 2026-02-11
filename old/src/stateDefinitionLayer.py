import torch


class DynamicStateDefinition:
    def __init__(self, orbital_keys, sensors_dict, init_tle) -> None:
        self.orbital_keys = orbital_keys
        self.sensors_dict = sensors_dict  # Store this reference!
        self.orbit_dim = len(orbital_keys)

        # 1. Build Sensor Slices
        self.sensor_slices = {}
        current_idx = self.orbit_dim

        for name, sensor in sensors_dict.items():
            n_p = sensor.num_params
            self.sensor_slices[name] = slice(current_idx, current_idx + n_p)
            current_idx += n_p

        self.total_dim = current_idx

        # 2. Store TLE Defaults
        self.defaults = {
            "n": init_tle._no_kozai,
            "e": init_tle._ecco,
            "i": init_tle._inclo,
            "raan": init_tle._nodeo,
            "argp": init_tle._argpo,
            "ma": init_tle._mo,
            "bstar": init_tle._bstar,
        }

        # Pre-compute the index map for the orbital part
        self.idx_map = {k: i for i, k in enumerate(iterable=orbital_keys)}

    def get_initial_state(self) -> torch.Tensor:
        """
        Returns x0 populated with:
        1. Initial TLE values (for active orbital keys)
        2. Sensor-specific initial guesses (for sensor params)
        """
        x = torch.zeros(self.total_dim, dtype=torch.float64)

        # A. Populate Orbital Elements
        for key in self.orbital_keys:
            idx = self.idx_map[key]
            val = self.defaults[key]
            if torch.is_tensor(val):
                val = val.item()
            x[idx] = val

        # B. Populate Sensor Parameters
        for name, sensor in self.sensors_dict.items():
            # Get the slice for this sensor in the global vector
            sl = self.sensor_slices[name]

            # Get the sensor's preferred initial values
            init_vals = sensor.get_initial_params()

            # Assign them to the state vector
            x[sl] = init_vals

        return x

    def unpack(self, state_vector) -> tuple[torch.Tensor, dict]:
        # (Same as before...)
        current_vals = self.defaults.copy()
        for key in self.orbital_keys:
            current_vals[key] = state_vector[self.idx_map[key]]

        sgp4_tensor = torch.stack(
            [
                torch.as_tensor(current_vals["n"]),
                torch.as_tensor(current_vals["e"]),
                torch.as_tensor(current_vals["i"]),
                torch.as_tensor(current_vals["raan"]),
                torch.as_tensor(current_vals["argp"]),
                torch.as_tensor(current_vals["ma"]),
                torch.as_tensor(current_vals["bstar"]),
            ]
        )

        sensor_params = {}
        for name, sl in self.sensor_slices.items():
            sensor_params[name] = state_vector[sl]

        return sgp4_tensor, sensor_params
