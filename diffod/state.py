import torch
from abc import ABC, abstractmethod
from dsgp4.tle import TLE
from diffod.utils import BiasGroup, transform_tle_to_mee, transform_mee_to_tle
from diffod.functional.tle import update
import dsgp4

class BaseSSV(ABC):
    """
    Abstract Base Class for Smart-State-Vectors. 
    Handles common logic for dynamic state sizing, active maps, and bias groups.
    """
    def __init__(self, init_tle: TLE, num_measurements: int, orbital_flags: list[tuple[str, bool]]):
        self.init_tle = init_tle
        self.num_measurements = num_measurements
        
        # The core dimension is determined by the length of the provided flags
        self.core_dim = len(orbital_flags)
        self.current_dim = self.core_dim

        self.map_param_to_idx = {name: idx for idx, (name, _) in enumerate(orbital_flags)}
        self.active_flags = {name: should_fit for name, should_fit in orbital_flags}

        self.bias_groups: dict[str, BiasGroup] = {}
        self.aux_params: dict[str, float] = {}
        self.aux_param_indices: dict[str, int] = {}

    @abstractmethod
    def get_initial_state(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Constructs the initial state tensor based on the specific coordinate representation."""
        pass

    @abstractmethod
    def get_functional_args(self, x_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extracts and transforms the state into the 9 standard SGP4 arguments."""
        pass

    @abstractmethod
    def export(self, x: torch.Tensor) -> dsgp4.TLE:
        """Exports the current state back into a dsgp4.TLE object."""
        pass

    # --- Standardized Methods ---

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
        active_map = torch.zeros(self.current_dim, dtype=torch.bool, device=device)
        
        for name, idx in self.map_param_to_idx.items():
            active_map[idx] = self.active_flags[name]
            
        for bg in self.bias_groups.values():
            start = bg.global_offset
            end = start + bg.num_params
            active_map[start:end] = True
            
        for idx in self.aux_param_indices.values():
            active_map[idx] = True
            
        return active_map

    def get_estimate_map(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return ~self.get_active_map(device=device)

    def add_parameter(self, key: str, value: float) -> None:
        if key in self.aux_param_indices:
            raise ValueError(f"Auxiliary parameter '{key}' already exists.")
            
        self.aux_params[key] = value
        self.aux_param_indices[key] = self.current_dim
        self.current_dim += 1

    def get_aux_parameter(self, x_state: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.aux_param_indices:
            raise KeyError(f"Auxiliary parameter '{key}' not found in state definition.")
        return x_state[self.aux_param_indices[key]]

class TLE_SSV(BaseSSV):
    def __init__(self, init_tle: TLE, num_measurements: int, **fit_kwargs):
        orbital_flags = [
            ("mean_motion", fit_kwargs.get("fit_mean_motion", False)),
            ("eccentricity", fit_kwargs.get("fit_eccentricity", False)),
            ("inclination", fit_kwargs.get("fit_inclination", False)),
            ("raan", fit_kwargs.get("fit_raan", False)),
            ("argument_of_perigee", fit_kwargs.get("fit_argp", False)),
            ("mean_anomaly", fit_kwargs.get("fit_ma", False)),
            ("b_star", fit_kwargs.get("fit_bstar", False)),
            ("ndot", fit_kwargs.get("fit_ndot", False)),
            ("nddot", fit_kwargs.get("fit_nddot", False)),
        ]
        super().__init__(init_tle, num_measurements, orbital_flags)

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
        x0 = torch.zeros(self.current_dim, dtype=torch.float32, requires_grad=True, device=device)
        with torch.no_grad():
            for name, idx in self.map_param_to_idx.items():
                _, tle_attr = self._func_arg_map[name]
                x0[idx] = getattr(self.init_tle, tle_attr, 0.0)
                
            for key, idx in self.aux_param_indices.items():
                x0[idx] = self.aux_params[key]
        return x0

    def get_functional_args(self, x_state: torch.Tensor) -> dict[str, torch.Tensor]:
        args = {}
        for friendly_name, (func_arg_name, _) in self._func_arg_map.items():
            idx = self.map_param_to_idx[friendly_name]
            args[func_arg_name] = x_state[idx] 
        return args

    def export(self, x: torch.Tensor) -> dsgp4.TLE:
        sat_obj = update(tle=self.init_tle, x=x, map_param_to_idx=self.map_param_to_idx)
        return sat_obj

class MEE_SSV(BaseSSV):
    def __init__(self, init_tle: TLE, num_measurements: int, **fit_kwargs):
        orbital_flags = [
            ("n", fit_kwargs.get("fit_mean_motion", False)),
            ("f", fit_kwargs.get("fit_f", False)),
            ("g", fit_kwargs.get("fit_g", False)),
            ("h", fit_kwargs.get("fit_h", False)),
            ("k", fit_kwargs.get("fit_k", False)),
            ("L", fit_kwargs.get("fit_L", False)),
            ("bstar", fit_kwargs.get("fit_bstar", False)),
            ("ndot", fit_kwargs.get("fit_ndot", False)),
            ("nddot", fit_kwargs.get("fit_nddot", False)),
        ]
        super().__init__(init_tle, num_measurements, orbital_flags)

    def get_initial_state(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x0 = torch.zeros(self.current_dim, dtype=torch.float32, requires_grad=True, device=device)
        
        with torch.no_grad():
            # 1. Cast TLE floats to tensors for the transformation block
            n = torch.tensor(self.init_tle.no_kozai, device=device)
            e = torch.tensor(self.init_tle.ecco, device=device)
            i = torch.tensor(self.init_tle.inclo, device=device)
            omega = torch.tensor(self.init_tle.argpo, device=device)
            raan = torch.tensor(self.init_tle.nodeo, device=device)
            m = torch.tensor(self.init_tle.mo, device=device)

            # 2. Transform
            mee_dict = transform_tle_to_mee(n, e, i, omega, raan, m)

            # 3. Populate state vector
            for key, val in mee_dict.items():
                x0[self.map_param_to_idx[key]] = val
            
            x0[self.map_param_to_idx["bstar"]] = getattr(self.init_tle, "bstar", 0.0)
            x0[self.map_param_to_idx["ndot"]] = getattr(self.init_tle, "ndot", 0.0)
            x0[self.map_param_to_idx["nddot"]] = getattr(self.init_tle, "nddot", 0.0)
            
            for key, idx in self.aux_param_indices.items():
                x0[idx] = self.aux_params[key]
                
        return x0

    def get_functional_args(self, x_state: torch.Tensor) -> dict[str, torch.Tensor]:
        n = x_state[self.map_param_to_idx["n"]]
        f = x_state[self.map_param_to_idx["f"]]
        g = x_state[self.map_param_to_idx["g"]]
        h = x_state[self.map_param_to_idx["h"]]
        k = x_state[self.map_param_to_idx["k"]]
        L = x_state[self.map_param_to_idx["L"]]
        
        tle_dict = transform_mee_to_tle(n, f, g, h, k, L)
        
        return {
            "no_kozai": tle_dict["no_kozai"],
            "ecco": tle_dict["ecco"],
            "inclo": tle_dict["inclo"],
            "nodeo": tle_dict["nodeo"],
            "argpo": tle_dict["argpo"],
            "mo": tle_dict["mo"],
            "bstar": x_state[self.map_param_to_idx["bstar"]],
            "ndot": x_state[self.map_param_to_idx["ndot"]],
            "nddot": x_state[self.map_param_to_idx["nddot"]]
        }

    def export(self, x: torch.Tensor) -> dsgp4.TLE:
        """
        Exports the optimized MEE state vector back into a dsgp4.TLE object.
        """
        with torch.no_grad():
            # 1. Run the forward transformation
            tle_dict = self.get_functional_args(x)
            
            # 2. Build a proxy TLE state vector and mapping specifically for diffod's `update`
            # The update function likely expects string keys mapping to standard SGP4 terminology
            tle_keys = ["mean_motion", "eccentricity", "inclination", "raan", 
                        "argument_of_perigee", "mean_anomaly", "b_star", "ndot", "nddot"]
                        
            tle_func_keys = ["no_kozai", "ecco", "inclo", "nodeo", "argpo", "mo", "bstar", "ndot", "nddot"]
            
            x_tle_proxy = torch.zeros(len(tle_keys), dtype=x.dtype, device=x.device)
            proxy_map = {}
            
            for idx, (friendly_name, func_name) in enumerate(zip(tle_keys, tle_func_keys)):
                x_tle_proxy[idx] = tle_dict[func_name]
                proxy_map[friendly_name] = idx
            
            # 3. Utilize your existing diffod updater
            sat_obj = update(tle=self.init_tle, x=x_tle_proxy, map_param_to_idx=proxy_map)
            
        return sat_obj

# import torch
# from dsgp4.tle import TLE
# # Assuming BiasGroup is available in diffod.utils
# from diffod.functional.tle import update
# from diffod.utils import BiasGroup 
# import dsgp4

# """
# Smart state vector
# """

# class SSV:
#     """
#     Docstring for Smart-State-Vector
#     """
#     def __init__(
#         self,
#         init_tle: TLE,
#         num_measurements: int,
#         fit_mean_motion: bool = False,
#         fit_eccentricity: bool = False,
#         fit_inclination: bool = False,
#         fit_raan: bool = False,
#         fit_argp: bool = False,
#         fit_ma: bool = False,
#         fit_bstar: bool = False,
#         fit_ndot: bool = False,
#         fit_nddot: bool = False,
#     ) -> None:
#         self.init_tle = init_tle
#         self.num_measurements = num_measurements
        
#         # The core state vector is permanently fixed to 9 elements.
#         self.core_dim = 9
#         self.current_dim = self.core_dim

#         # Mapping: (Internal Name) -> (Should Fit?)
#         orbital_flags = [
#             ("mean_motion", fit_mean_motion),
#             ("eccentricity", fit_eccentricity),
#             ("inclination", fit_inclination),
#             ("raan", fit_raan),
#             ("argument_of_perigee", fit_argp),
#             ("mean_anomaly", fit_ma),
#             ("b_star", fit_bstar),
#             ("ndot", fit_ndot),
#             ("nddot", fit_nddot),
#         ]

#         # All 9 parameters are always present in the state vector at fixed indices
#         self.map_param_to_idx = {name: idx for idx, (name, _) in enumerate(orbital_flags)}
#         self.active_flags = {name: should_fit for name, should_fit in orbital_flags}

#         self.bias_groups: dict[str, BiasGroup] = {}

#         # Static mapping from our friendly names to Functional SGP4 arg names
#         self._func_arg_map = {
#             "mean_motion": ("no_kozai", "_no_kozai"),
#             "eccentricity": ("ecco", "_ecco"),
#             "inclination": ("inclo", "_inclo"),
#             "raan": ("nodeo", "_nodeo"),
#             "argument_of_perigee": ("argpo", "_argpo"),
#             "mean_anomaly": ("mo", "_mo"),
#             "b_star": ("bstar", "_bstar"),
#             "ndot": ("ndot", "_ndot"),
#             "nddot": ("nddot", "_nddot"),
#         }

#         self.aux_params: dict[str, float] = {}
#         self.aux_param_indices: dict[str, int] = {}

#     def get_initial_state(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
#             x0 = torch.zeros(
#                 self.current_dim, dtype=torch.float32, requires_grad=True, device=device
#             )

#             with torch.no_grad():
#                 for name, idx in self.map_param_to_idx.items():
#                     _, tle_attr = self._func_arg_map[name]
#                     x0[idx] = getattr(self.init_tle, tle_attr, 0.0)
                    
#                 # Populate auxiliary parameters
#                 for key, idx in self.aux_param_indices.items():
#                     x0[idx] = self.aux_params[key]
                    
#             return x0

#     def get_functional_args(self, x_state: torch.Tensor) -> dict[str, torch.Tensor]:
#         """
#         Extracts all 9 SGP4 parameters directly from x_state.
#         Because x_state always contains the full core state, we bypass the TLE fallback.
#         """
#         args = {}
#         for friendly_name, (func_arg_name, _) in self._func_arg_map.items():
#             idx = self.map_param_to_idx[friendly_name]
#             # Slicing keeps it as a scalar tensor to preserve gradients
#             args[func_arg_name] = x_state[idx] 
            
#         return args

#     def add_linear_bias(self, name: str, group_indices: torch.Tensor):
#         valid_mask = group_indices >= 0
#         if not valid_mask.any():
#             return

#         max_idx = group_indices[valid_mask].max().item()
#         num_new_params = int(max_idx + 1)

#         group = BiasGroup(
#             name=name,
#             indices=group_indices,
#             global_offset=self.current_dim,
#             num_params=num_new_params,
#         )
#         self.bias_groups[name] = group
#         self.current_dim += num_new_params

#     def get_bias_group(self, name: str) -> BiasGroup:
#         if name in self.bias_groups:
#             return self.bias_groups[name]
#         raise ValueError(f"Bias group '{name}' not found.")

#     def get_active_map(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
#         active_map = torch.zeros(self.current_dim, dtype=torch.bool, device=device)
        
#         # 1. Flag core orbital parameters
#         for name, idx in self.map_param_to_idx.items():
#             active_map[idx] = self.active_flags[name]
            
#         # 2. Bias parameters
#         for bg in self.bias_groups.values():
#             start = bg.global_offset
#             end = start + bg.num_params
#             active_map[start:end] = True
            
#         # 3. Auxiliary parameters are dynamic and assumed active
#         for idx in self.aux_param_indices.values():
#             active_map[idx] = True
            
#         return active_map

#     def get_estimate_map(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
#         """
#         Returns the logical inverse of the active map for Consider Covariance Analysis.
#         True -> Parameter is Considered (uncertainty accounted for, but state fixed).
#         False -> Parameter is Estimated.
#         """
#         return ~self.get_active_map(device=device)
    
#     def export(self, x: torch.Tensor) -> dsgp4.TLE:
#         """
#         Exports a TLE object with the given state vector.
        
#         :param x: State vector
#         :return: dsgp4 TLE object
#         :rtype: dsgp4.TLE
#         """
#         sat_obj = update(tle=self.init_tle,
#                          x=x,
#                          map_param_to_idx=self.map_param_to_idx)
        
#         assert isinstance(sat_obj, dsgp4.TLE)

#         return sat_obj

#     def add_parameter(self, key: str, value: float) -> None:
#         """
#         Adds an auxiliary parameter (e.g., time offset) to the state vector.
        
#         :param key: Unique string identifier for the parameter
#         :param value: Initial float value for the parameter
#         """
#         if key in self.aux_param_indices:
#             raise ValueError(f"Auxiliary parameter '{key}' already exists.")
            
#         self.aux_params[key] = value
#         self.aux_param_indices[key] = self.current_dim
#         self.current_dim += 1

#     def get_aux_parameter(self, x_state: torch.Tensor, key: str) -> torch.Tensor:
#         """
#         Extracts the scalar tensor for an auxiliary parameter, preserving gradients.
#         """
#         if key not in self.aux_param_indices:
#             raise KeyError(f"Auxiliary parameter '{key}' not found in state definition.")
        
#         # Slicing keeps it as a scalar tensor for Autograd
#         return x_state[self.aux_param_indices[key]]