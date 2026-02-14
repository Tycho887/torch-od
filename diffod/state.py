import torch
from dsgp4.tle import TLE

from diffod.utils import BiasGroup


class StateDefinition:
    def __init__(
        self,
        init_tle: TLE,
        num_measurements: int,
        fit_mean_motion: bool = False,
        fit_eccentricity: bool = False,
        fit_inclination: bool = False,
        fit_raan: bool = False,
        fit_argp: bool = False,
        fit_ma: bool = False,
        fit_bstar: bool = False,
        fit_ndot: bool = False,  # Added capability to fit drag terms if needed
        fit_nddot: bool = False,
    ):
        self.init_tle = init_tle
        self.num_measurements = num_measurements
        self.map_param_to_idx = {}
        self.current_dim = 0

        # Mapping: (Internal Name) -> (Should Fit?)
        # Order determines position in State Vector X
        orbital_flags = [
            ("mean_motion", fit_mean_motion),
            ("eccentricity", fit_eccentricity),
            ("inclination", fit_inclination),
            ("raan", fit_raan),
            ("argument_of_perigee", fit_argp),
            ("mean_anomaly", fit_ma),
            ("b_star", fit_bstar),
            ("ndot", fit_ndot),
            ("nddot", fit_nddot),
        ]

        for name, should_fit in orbital_flags:
            if should_fit:
                self.map_param_to_idx[name] = self.current_dim
                self.current_dim += 1

        self.bias_groups: dict[str, BiasGroup] = {}

        # Static mapping from our friendly names to Functional SGP4 arg names
        # and the corresponding TLE attribute for default values.
        self._func_arg_map = {
            "mean_motion": ("no_kozai", "_no_kozai"),
            "eccentricity": ("ecco", "_ecco"),
            "inclination": ("inclo", "_inclo"),
            "raan": ("nodeo", "_nodeo"),
            "argument_of_perigee": ("argpo", "_argpo"),
            "mean_anomaly": ("mo", "_mo"),
            "b_star": ("bstar", "_bstar"),
            "ndot": ("ndot", "_ndot"),
            "nddot": ("nddot", "_nddot"),
        }

    def get_initial_state(
        self, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        # Explicitly set the device for the optimization variables
        x0 = torch.zeros(
            self.current_dim, dtype=torch.float32, requires_grad=True, device=device
        )

        with torch.no_grad():
            for name, idx in self.map_param_to_idx.items():
                _, tle_attr = self._func_arg_map[name]
                val = getattr(self.init_tle, tle_attr, 0.0)
                x0[idx] = val
        return x0

    def get_functional_args(self, x_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts SGP4 parameters from x_state (active) or init_tle (static).
        Returns a dict suited for `sgp4_propagate(**kwargs)`.
        """
        args = {}
        device = x_state.device
        dtype = x_state.dtype

        # Iterate over all possible SGP4 parameters
        for friendly_name, (func_arg_name, tle_attr) in self._func_arg_map.items():
            if friendly_name in self.map_param_to_idx:
                # 1. Active Parameter: Get from State Vector
                idx = self.map_param_to_idx[friendly_name]
                val = x_state[idx]  # Keep as scalar tensor to preserve gradients
            else:
                # 2. Static Parameter: Get from initial TLE
                # We convert to tensor here so SGP4 receives uniform inputs
                float_val = getattr(self.init_tle, tle_attr, 0.0)
                val = torch.tensor(float_val, device=device, dtype=dtype)

            args[func_arg_name] = val

        return args

    def add_linear_bias(self, name: str, group_indices: torch.Tensor):
        valid_mask = group_indices >= 0
        if not valid_mask.any():
            return

        max_idx = group_indices[valid_mask].max().item()
        num_new_params = int(max_idx + 1)

        group = BiasGroup(
            name=name,
            indices=group_indices,
            global_offset=self.current_dim,
            num_params=num_new_params,
        )
        self.bias_groups[name] = group
        self.current_dim += num_new_params

    def get_bias_group(self, name: str) -> BiasGroup:
        if name in self.bias_groups:
            return self.bias_groups[name]
        else:
            raise ValueError(f"Bias group '{name}' not found.")
