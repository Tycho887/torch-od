import torch
from dataclasses import dataclass
from dsgp4.tle import TLE

@dataclass
class BiasGroup:
    """
    Stores metadata for a specific group of bias parameters.
    """
    name: str
    indices: torch.Tensor  # (N_measurements,) Mapping: meas_idx -> local_param_idx
    global_offset: int     # Where this group starts in the state vector 'x'
    num_params: int        # How many parameters are in this group

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
    ):
        self.init_tle = init_tle
        self.num_measurements = num_measurements

        # 1. Map Booleans to State Indices
        self.map_param_to_idx = {}
        self.current_dim = 0

        # Define the order of orbital elements in the state vector
        # (Using a list of tuples ensures deterministic insertion order)
        orbital_flags = [
            ("mean_motion", fit_mean_motion),
            ("eccentricity", fit_eccentricity),
            ("inclination", fit_inclination),
            ("raan", fit_raan),
            ("argument_of_perigee", fit_argp),
            ("mean_anomaly", fit_ma),
            ("b_star", fit_bstar),
        ]

        for name, should_fit in orbital_flags:
            if should_fit:
                self.map_param_to_idx[name] = self.current_dim
                self.current_dim += 1

        # 2. Storage for Bias Groups
        # Dictionary mapping name -> BiasGroup dataclass
        self.bias_groups: dict[str, BiasGroup] = {}

    def get_initial_state(self) -> torch.Tensor:
        """
        Returns the initial x0 tensor populated with TLE values.
        """
        x0 = torch.zeros(self.current_dim, dtype=torch.float64, requires_grad=True)

        # Mapping of friendly names to internal TLE attributes
        # (This handles the difference between 'raan' and '_nodeo', etc.)
        tle_attr_map = {
            "mean_motion": "_no_kozai", # Note: Check if your TLE uses _no_kozai (rad/min) or mean_motion (rev/day)
            "eccentricity": "_ecco",
            "inclination": "_inclo",
            "raan": "_nodeo",
            "argument_of_perigee": "_argpo",
            "mean_anomaly": "_mo",
            "b_star": "_bstar"
        }

        with torch.no_grad():
            for name, idx in self.map_param_to_idx.items():
                attr_name = tle_attr_map.get(name, name)
                val = getattr(self.init_tle, attr_name, 0.0)
                x0[idx] = val

        return x0

    def build_tle_args(self, x_state: torch.Tensor) -> dict:
        """
        Constructs a dictionary of TLE parameters using values from x_state.
        This dict can be passed to dSGP4's sgp4init.
        
        IMPORTANT: This returns TENSORS for optimized params to maintain gradients.
        """
        
        def get(key, default_attr):
            """Get from State Vector (Tensor) or TLE Object (Float)"""
            if key in self.map_param_to_idx:
                idx = self.map_param_to_idx[key]
                return x_state[idx] # Returns Tensor, graph connected
            return getattr(self.init_tle, default_attr)

        # Note: We return a dict, because creating a TLE object might force casting to float
        # depending on dSGP4 implementation. 
        arguments = {
            "mean_motion": get("mean_motion", "_no_kozai"),
            "eccentricity": get("eccentricity", "_ecco"),
            "inclination": get("inclination", "_inclo"),
            "raan": get("raan", "_nodeo"),
            "argument_of_perigee": get("argument_of_perigee", "_argpo"),
            "mean_anomaly": get("mean_anomaly", "_mo"),
            "b_star": get("b_star", "_bstar"),
            # Pass through statics
            "satellite_catalog_number": self.init_tle.satellite_catalog_number,
            "epoch_year": self.init_tle.epoch_year,
            "epoch_days": self.init_tle.epoch_days,
        }
        return arguments

    def add_linear_bias(self, name: str, group_indices: torch.Tensor):
        """
        Registers a bias parameter group using the BiasGroup dataclass.
        
        Args:
            name: ID for this bias (e.g., 'doppler')
            group_indices: (N,) tensor where value 'j' means use bias parameter 'j'. 
                           -1 implies no bias.
        """
        # Filter out -1 to find the max index
        valid_mask = group_indices >= 0
        if not valid_mask.any():
            return # No biases to add

        max_idx = group_indices[valid_mask].max().item()
        num_new_params = int(max_idx + 1)

        # Create the metadata object
        group = BiasGroup(
            name=name,
            indices=group_indices,
            global_offset=self.current_dim,
            num_params=num_new_params
        )

        # Store it
        self.bias_groups[name] = group
        
        # Advance the global state dimension
        self.current_dim += num_new_params

    def get_bias_group(self, name: str) -> BiasGroup | None:
        """Retrieve the dataclass for a specific bias."""
        return self.bias_groups.get(name)

# import torch
# from dsgp4.newton_method import newton_method
# from dsgp4.tle import TLE


# class StateDefinition:
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
#     ):
#         self.init_tle = init_tle
#         self.num_measurements = num_measurements

#         # 1. Map Booleans to State Indices
#         # We store tuples of (TLE_Key_Name, State_Index)
#         self.map_param_to_idx = {}
#         self.current_dim = 0

#         # Order matters for the state vector construction
#         if fit_mean_motion:
#             self.map_param_to_idx["mean_motion"] = self.current_dim
#             self.current_dim += 1

#         if fit_eccentricity:
#             self.map_param_to_idx["eccentricity"] = self.current_dim
#             self.current_dim += 1

#         if fit_inclination:
#             self.map_param_to_idx["inclination"] = self.current_dim
#             self.current_dim += 1

#         if fit_raan:
#             self.map_param_to_idx["raan"] = self.current_dim
#             self.current_dim += 1

#         if fit_argp:
#             self.map_param_to_idx["argument_of_perigee"] = self.current_dim
#             self.current_dim += 1

#         if fit_ma:
#             self.map_param_to_idx["mean_anomaly"] = self.current_dim
#             self.current_dim += 1

#         if fit_bstar:
#             self.map_param_to_idx["b_star"] = self.current_dim
#             self.current_dim += 1

#         # Storage for bias matrices
#         self.bias_maps = {}
#         self.bias_slices = {}

#     def get_initial_state(self):
#         """
#         Returns the initial x0 tensor populated with TLE values.
#         """
#         x0 = torch.zeros(self.current_dim, dtype=torch.float64, requires_grad=True)

#         # We need to access the TLE data.
#         # Note: dSGP4 TLE attributes often match the constructor keys.
#         # We wrap in `with torch.no_grad()` to ensure x0 is a leaf node initialization
#         with torch.no_grad():
#             if "mean_motion" in self.map_param_to_idx:
#                 x0[self.map_param_to_idx["mean_motion"]] = self.init_tle._no_kozai
#             if "eccentricity" in self.map_param_to_idx:
#                 x0[self.map_param_to_idx["eccentricity"]] = self.init_tle._ecco
#             if "inclination" in self.map_param_to_idx:
#                 x0[self.map_param_to_idx["inclination"]] = self.init_tle._inclo
#             if "raan" in self.map_param_to_idx:
#                 x0[self.map_param_to_idx["raan"]] = self.init_tle._nodeo
#             if "argument_of_perigee" in self.map_param_to_idx:
#                 x0[self.map_param_to_idx["argument_of_perigee"]] = self.init_tle._argpo
#             if "mean_anomaly" in self.map_param_to_idx:
#                 x0[self.map_param_to_idx["mean_anomaly"]] = self.init_tle._mo
#             if "b_star" in self.map_param_to_idx:
#                 x0[self.map_param_to_idx["b_star"]] = self.init_tle._bstar

#         return x0

#     def build_tle(self, x_state: torch.Tensor) -> TLE:
#         """
#         Constructs a NEW TLE object using values from x_state where applicable,
#         and falling back to init_tle for static parameters.

#         This maintains the computational graph because we pass the 'x' tensors
#         directly into the dictionary used to initialize the TLE.
#         """

#         # Helper to decide: Get from Tensor (Optimization) OR Get from Object (Static)
#         def get_val(key, default_val):
#             if key in self.map_param_to_idx:
#                 idx = self.map_param_to_idx[key]
#                 return x_state[idx]  # This is a Tensor, keeps Grad!
#             return default_val

#         arguments = {
#             # --- Dynamic Parameters (Potentially Tensors) ---
#             "mean_motion": float(
#                 get_val("mean_motion", self.init_tle._no_kozai).detach() / 60
#             ),
#             "eccentricity": float(
#                 get_val("eccentricity", self.init_tle._ecco).detach()
#             ),
#             "inclination": float(get_val("inclination", self.init_tle._inclo).detach()),
#             "raan": float(get_val("raan", self.init_tle._nodeo).detach()),
#             "argument_of_perigee": float(
#                 get_val("argument_of_perigee", self.init_tle._argpo).detach()
#             ),
#             "mean_anomaly": float(get_val("mean_anomaly", self.init_tle._mo).detach()),
#             "b_star": float(get_val("b_star", self.init_tle._bstar).detach()),
#             # --- Static Parameters (Pass-through) ---
#             "satellite_catalog_number": self.init_tle.satellite_catalog_number,
#             "epoch_year": self.init_tle.epoch_year,
#             "epoch_days": self.init_tle.epoch_days,
#             "mean_motion_first_derivative": self.init_tle.mean_motion_first_derivative,
#             "mean_motion_second_derivative": self.init_tle.mean_motion_second_derivative,
#             "classification": self.init_tle.classification,
#             "ephemeris_type": self.init_tle.ephemeris_type,
#             "international_designator": self.init_tle.international_designator,
#             "revolution_number_at_epoch": self.init_tle.revolution_number_at_epoch,
#             "element_number": self.init_tle.element_number,
#         }

#         # 2. Return a fresh TLE object
#         # dSGP4 will store the tensors inside. When you call propagate() on this object,
#         # the operations will trace back to x_state.
#         return TLE(arguments)

#     def add_linear_bias(self, name: str, group_indices: torch.Tensor):
#         """
#         Registers a bias parameter group.
#         Stores the indices directly for efficient gathering later.
#         """
#         # 1. Determine size
#         # Assuming -1 is used for "no bias", we filter those out for counting
#         valid_indices = group_indices[group_indices >= 0]
#         if valid_indices.numel() == 0:
#             return

#         num_new_params = valid_indices.max().item() + 1

#         # 2. Assign state vector slots
#         start_idx = self.current_dim
#         self.current_dim += num_new_params

#         # 3. Store the mapping info
#         # We need the indices and the start_idx to locate params in 'x' later
#         self.bias_maps[name] = {
#             "indices": group_indices,  # Shape (N,)
#             "offset": start_idx,  # Integer scalar
#         }

#     def get_bias_map(self, name):
#         """Returns the mapping data needed for physics.apply_linear_bias"""
#         return self.bias_maps.get(name, None)
