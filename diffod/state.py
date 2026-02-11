import torch
from dsgp4.newton_method import newton_method
from dsgp4.tle import TLE


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
        # We store tuples of (TLE_Key_Name, State_Index)
        self.map_param_to_idx = {}
        self.current_dim = 0

        # Order matters for the state vector construction
        if fit_mean_motion:
            self.map_param_to_idx["mean_motion"] = self.current_dim
            self.current_dim += 1

        if fit_eccentricity:
            self.map_param_to_idx["eccentricity"] = self.current_dim
            self.current_dim += 1

        if fit_inclination:
            self.map_param_to_idx["inclination"] = self.current_dim
            self.current_dim += 1

        if fit_raan:
            self.map_param_to_idx["raan"] = self.current_dim
            self.current_dim += 1

        if fit_argp:
            self.map_param_to_idx["argument_of_perigee"] = self.current_dim
            self.current_dim += 1

        if fit_ma:
            self.map_param_to_idx["mean_anomaly"] = self.current_dim
            self.current_dim += 1

        if fit_bstar:
            self.map_param_to_idx["b_star"] = self.current_dim
            self.current_dim += 1

        # Storage for bias matrices
        self.bias_maps = {}
        # self.bias_slices = {}

    def get_initial_state(self):
        """
        Returns the initial x0 tensor populated with TLE values.
        """
        x0 = torch.zeros(self.current_dim, dtype=torch.float64, requires_grad=True)

        # We need to access the TLE data.
        # Note: dSGP4 TLE attributes often match the constructor keys.
        # We wrap in `with torch.no_grad()` to ensure x0 is a leaf node initialization
        with torch.no_grad():
            if "mean_motion" in self.map_param_to_idx:
                x0[self.map_param_to_idx["mean_motion"]] = self.init_tle._no_kozai
            if "eccentricity" in self.map_param_to_idx:
                x0[self.map_param_to_idx["eccentricity"]] = self.init_tle._ecco
            if "inclination" in self.map_param_to_idx:
                x0[self.map_param_to_idx["inclination"]] = self.init_tle._inclo
            if "raan" in self.map_param_to_idx:
                x0[self.map_param_to_idx["raan"]] = self.init_tle._nodeo
            if "argument_of_perigee" in self.map_param_to_idx:
                x0[self.map_param_to_idx["argument_of_perigee"]] = self.init_tle._argpo
            if "mean_anomaly" in self.map_param_to_idx:
                x0[self.map_param_to_idx["mean_anomaly"]] = self.init_tle._mo
            if "b_star" in self.map_param_to_idx:
                x0[self.map_param_to_idx["b_star"]] = self.init_tle._bstar

        return x0

    def build_tle(self, x_state: torch.Tensor) -> TLE:
        """
        Constructs a NEW TLE object using values from x_state where applicable,
        and falling back to init_tle for static parameters.

        This maintains the computational graph because we pass the 'x' tensors
        directly into the dictionary used to initialize the TLE.
        """

        # Helper to decide: Get from Tensor (Optimization) OR Get from Object (Static)
        def get_val(key, default_val):
            if key in self.map_param_to_idx:
                idx = self.map_param_to_idx[key]
                return x_state[idx]  # This is a Tensor, keeps Grad!
            return default_val

        arguments = {
            # --- Dynamic Parameters (Potentially Tensors) ---
            "mean_motion": float(
                get_val("mean_motion", self.init_tle._no_kozai).detach() / 60
            ),
            "eccentricity": float(
                get_val("eccentricity", self.init_tle._ecco).detach()
            ),
            "inclination": float(get_val("inclination", self.init_tle._inclo).detach()),
            "raan": float(get_val("raan", self.init_tle._nodeo).detach()),
            "argument_of_perigee": float(
                get_val("argument_of_perigee", self.init_tle._argpo).detach()
            ),
            "mean_anomaly": float(get_val("mean_anomaly", self.init_tle._mo).detach()),
            "b_star": float(get_val("b_star", self.init_tle._bstar).detach()),
            # --- Static Parameters (Pass-through) ---
            "satellite_catalog_number": self.init_tle.satellite_catalog_number,
            "epoch_year": self.init_tle.epoch_year,
            "epoch_days": self.init_tle.epoch_days,
            "mean_motion_first_derivative": self.init_tle.mean_motion_first_derivative,
            "mean_motion_second_derivative": self.init_tle.mean_motion_second_derivative,
            "classification": self.init_tle.classification,
            "ephemeris_type": self.init_tle.ephemeris_type,
            "international_designator": self.init_tle.international_designator,
            "revolution_number_at_epoch": self.init_tle.revolution_number_at_epoch,
            "element_number": self.init_tle.element_number,
        }

        # 2. Return a fresh TLE object
        # dSGP4 will store the tensors inside. When you call propagate() on this object,
        # the operations will trace back to x_state.
        return TLE(arguments)

    def add_linear_bias(self, name: str, group_indices: torch.Tensor):
        """
        Registers a bias parameter group.
        Stores the indices directly for efficient gathering later.
        """
        # 1. Determine size
        # Assuming -1 is used for "no bias", we filter those out for counting
        valid_indices = group_indices[group_indices >= 0]
        if valid_indices.numel() == 0:
            return

        num_new_params = valid_indices.max().item() + 1

        # 2. Assign state vector slots
        start_idx = self.current_dim
        self.current_dim += num_new_params

        # 3. Store the mapping info
        # We need the indices and the start_idx to locate params in 'x' later
        self.bias_maps[name] = {
            "indices": group_indices,  # Shape (N,)
            "offset": start_idx,  # Integer scalar
        }

    def get_bias_map(self, name):
        """Returns the mapping data needed for physics.apply_linear_bias"""
        return self.bias_maps.get(name, None)
