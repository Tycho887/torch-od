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
    def __init__(self, num_measurements: int, orbital_flags: list[tuple[str, bool]], dtype=torch.float64) -> None:
        # self.init_tle = init_tle
        self.num_measurements = num_measurements
        
        # The core dimension is determined by the length of the provided flags
        self.core_dim = len(orbital_flags)
        self.current_dim = self.core_dim

        self.map_param_to_idx = {name: idx for idx, (name, _) in enumerate(orbital_flags)}
        self.active_flags = {name: should_fit for name, should_fit in orbital_flags}

        self.bias_groups: dict[str, BiasGroup] = {}
        self.aux_params: dict[str, float] = {}
        self.aux_param_indices: dict[str, int] = {}
        self.dtype = dtype

    @abstractmethod
    def get_initial_state(self, device: torch.device = torch.device(device="cpu")) -> torch.Tensor:
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

    def add_linear_bias(self, name: str, group_indices: torch.Tensor) -> None:
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
    def __init__(self, init_tle: TLE, num_measurements: int, **fit_kwargs) -> None:
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
        super().__init__(num_measurements=num_measurements, orbital_flags=orbital_flags)
        self.init_tle = init_tle
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
        x0 = torch.zeros(self.current_dim, dtype=self.dtype, requires_grad=True, device=device)
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
    def __init__(self, init_tle: TLE, num_measurements: int, **fit_kwargs) -> None:
        orbital_flags = [
            ("n", fit_kwargs.get("fit_mean_motion", True)),
            ("f", fit_kwargs.get("fit_f", True)),
            ("g", fit_kwargs.get("fit_g", True)),
            ("h", fit_kwargs.get("fit_h", True)),
            ("k", fit_kwargs.get("fit_k", True)),
            ("L", fit_kwargs.get("fit_L", True)),
            ("bstar", fit_kwargs.get("fit_bstar", True)),
            ("ndot", fit_kwargs.get("fit_ndot", False)),
            ("nddot", fit_kwargs.get("fit_nddot", False)),
        ]
        super().__init__(num_measurements=num_measurements, orbital_flags=orbital_flags)
        self.init_tle = init_tle


    def get_initial_state(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x0 = torch.zeros(self.current_dim, dtype=self.dtype, requires_grad=True, device=device)
        
        with torch.no_grad():
            # 1. Cast TLE floats to tensors for the transformation block
            n = self.init_tle._no_kozai.detach().clone()#torch.tensor(self.init_tle._no_kozai.detach().numpy()), device=device)
            e = self.init_tle._ecco.detach().clone()#torch.tensor(self.init_tle._ecco, device=device)
            i = self.init_tle._inclo.detach().clone()#torch.tensor(self.init_tle._inclo, device=device)
            omega = self.init_tle._argpo.detach().clone()#torch.tensor(self.init_tle._argpo, device=device)
            raan = self.init_tle._nodeo.detach().clone()#torch.tensor(self.init_tle._nodeo, device=device)
            m = self.init_tle._mo.detach().clone()#torch.tensor(self.init_tle._mo, device=device)

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


class CalibrationSSV(BaseSSV):
    """
    SSV dedicated purely to measurement system calibration with per-pass bias support.
    Includes parameter normalization to ensure well-conditioned Jacobian matrices.
    """
    def __init__(
        self, 
        num_measurements: int, 
        fit_time_offset: bool = True,
        fit_frequency_offset: bool = True,
    ) -> None:
        orbital_flags = [
            ("time_offset", fit_time_offset),
            ("freq_offset", fit_frequency_offset),
        ]
        
        super().__init__(num_measurements=num_measurements, orbital_flags=orbital_flags)
        
        # Define physical scales. 
        # A value of 1.0 in the optimizer's state vector maps to these physical magnitudes.
        self.scales = {
            "time_offset": 1.0,    # 1.0 state unit = 10.0 seconds
            "freq_offset": 1.0,  # 1.0 state unit = 1000.0 Hz
        }

    def get_initial_state(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        x0 = torch.zeros(self.current_dim, dtype=self.dtype, requires_grad=True, device=device)
        return x0

    def get_functional_args(self, x_state: torch.Tensor) -> dict[str, torch.Tensor]:
        # Inflate the normalized state back to physical units for the forward pass
        return {
            "time_offset": x_state[self.map_param_to_idx["time_offset"]] * self.scales["time_offset"],
            "freq_offset": x_state[self.map_param_to_idx["freq_offset"]] * self.scales["freq_offset"],
        }

    def export(self, x: torch.Tensor) -> dict[str, float]:
        # get_functional_args already applies the scaling, so we just extract the physical values
        args = self.get_functional_args(x)
        output = {k: v.item() for k, v in args.items()}
        
        # Scale the solved pass biases 
        if "pass_bias" in self.bias_groups:
            bg = self.bias_groups["pass_bias"]
            start = bg.global_offset
            end = start + bg.num_params
            
            # Since apply_linear_bias in physics.py defaults to scaling=1e3 (1000 Hz),
            # we multiply the state by 1000.0 to export the true Hz value.
            output["pass_biases"] = (x[start:end]).detach().cpu().numpy().tolist()
            
        return output