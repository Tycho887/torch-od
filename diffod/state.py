import torch
from dsgp4.tle import TLE
# Assuming BiasGroup is available in diffod.utils
from diffod.functional.tle import update
from diffod.utils import BiasGroup 
import dsgp4

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
        fit_ndot: bool = False,
        fit_nddot: bool = False,
    ):
        self.init_tle = init_tle
        self.num_measurements = num_measurements
        
        # The core state vector is permanently fixed to 9 elements.
        self.core_dim = 9
        self.current_dim = self.core_dim

        # Mapping: (Internal Name) -> (Should Fit?)
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

        # All 9 parameters are always present in the state vector at fixed indices
        self.map_param_to_idx = {name: idx for idx, (name, _) in enumerate(orbital_flags)}
        self.active_flags = {name: should_fit for name, should_fit in orbital_flags}

        self.bias_groups: dict[str, BiasGroup] = {}

        # Static mapping from our friendly names to Functional SGP4 arg names
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

    def get_initial_state(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x0 = torch.zeros(
            self.current_dim, dtype=torch.float32, requires_grad=True, device=device
        )

        with torch.no_grad():
            for name, idx in self.map_param_to_idx.items():
                _, tle_attr = self._func_arg_map[name]
                x0[idx] = getattr(self.init_tle, tle_attr, 0.0)
                
        return x0

    def get_functional_args(self, x_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts all 9 SGP4 parameters directly from x_state.
        Because x_state always contains the full core state, we bypass the TLE fallback.
        """
        args = {}
        for friendly_name, (func_arg_name, _) in self._func_arg_map.items():
            idx = self.map_param_to_idx[friendly_name]
            # Slicing keeps it as a scalar tensor to preserve gradients
            args[func_arg_name] = x_state[idx] 
            
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
        raise ValueError(f"Bias group '{name}' not found.")

    def get_active_map(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Generates a boolean selection vector for Normal Equation solving.
        True -> Parameter is Active (columns to select from the Jacobian).
        """
        active_map = torch.zeros(self.current_dim, dtype=torch.bool, device=device)
        
        # 1. Flag core orbital parameters based on initialization arguments
        for name, idx in self.map_param_to_idx.items():
            active_map[idx] = self.active_flags[name]
            
        # 2. Bias parameters are dynamically added and assumed active
        for bg in self.bias_groups.values():
            start = bg.global_offset
            end = start + bg.num_params
            active_map[start:end] = True
            
        return active_map

    def get_estimate_map(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Returns the logical inverse of the active map for Consider Covariance Analysis.
        True -> Parameter is Considered (uncertainty accounted for, but state fixed).
        False -> Parameter is Estimated.
        """
        return ~self.get_active_map(device=device)
    
    def export(self, x: torch.Tensor) -> dsgp4.TLE:
        """
        Exports a TLE object with the given state vector.
        
        :param x: State vector
        :return: dsgp4 TLE object
        :rtype: dsgp4.TLE
        """
        sat_obj = update(tle=self.init_tle,
                         x=x,
                         map_param_to_idx=self.map_param_to_idx)
        
        assert isinstance(sat_obj, dsgp4.TLE)

        return sat_obj
