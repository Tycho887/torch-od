from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSensor(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass

    @abstractmethod
    def forward(self, geometry_data, sensor_params, contact_indices) -> torch.Tensor:
        pass

    def get_initial_params(self) -> torch.Tensor:
        """
        Returns the initial guess for this sensor's parameters.
        Default: Vector of zeros.
        Override this if you need non-zero initialization (e.g. nominal freq).
        """
        return torch.zeros(self.num_params, dtype=torch.float64)


class DopplerSensor(BaseSensor):
    def __init__(
        self,
        num_passes,
        center_freq,
        fit_center_freq=False,
        station_id=None,
    ) -> None:
        super().__init__()
        # if station_teme_pos is not None:
        #     self.register_buffer(name="station_pos", tensor=station_teme_pos)
        #     self.register_buffer(name="station_vel", tensor=station_teme_vel)

        self.station_id = station_id
        self.nominal_freq = center_freq
        self.fit_fc = fit_center_freq
        self.n_passes = num_passes

        # [Bias_Pass_1, Bias_Pass_2, ..., Delta_Freq]
        self._n_params = num_passes + (1 if fit_center_freq else 0)

    @property
    def num_params(self) -> int:
        return self._n_params

    def get_initial_params(self) -> torch.Tensor:
        # Biases start at 0.0
        # Freq offset starts at 0.0 (since we model it as delta from nominal)
        return torch.zeros(self.num_params, dtype=torch.float64)

    def forward(self, geometry_data, sensor_params, contact_indices) -> torch.Tensor:
        # 1. Physics: Range Rate

        assert torch.max(contact_indices) == self.n_passes

        # 2. Determine Effective Frequency
        # If we are fitting Fc, the last parameter is the correction (delta_hz)
        freq_to_use = self.nominal_freq
        pass_biases = sensor_params

        if self.fit_fc:
            # Split params: [Pass_Biases (N) | Freq_Bias (1)]
            pass_biases = sensor_params[:-1]
            delta_freq = sensor_params[-1]
            freq_to_use = freq_to_use + delta_freq

        range_rate = geometry_data["range_rate"]

        print(f"range rate shape: {range_rate.shape}")
        print(f"Pass biases shape: {pass_biases.shape}")

        c = 299792.458  # km/s
        doppler_ideal = -(range_rate / c) * freq_to_use

        # if contact_indices is not None and self.n_passes > 0:
        # assert torch.max(contact_indices) == self.n_passes

        # 3. Ideal Doppler
        # bias_correction = sensor_params[contact_indices]
        return doppler_ideal + pass_biases
        # else:
        #     return doppler_ideal

        # # 4. Apply Per-Pass Bias
        # # contact_indices maps measurement -> pass ID
        # if self.n_passes > 0:
        #     bias_correction = pass_biases[contact_indices]
        #     return doppler_ideal + bias_correction

        # return doppler_ideal


class RadarSensor(BaseSensor):
    def __init__(self, num_passes, station_teme_pos=None, station_id=None) -> None:
        super().__init__()
        if station_teme_pos is not None:
            self.register_buffer(name="station_pos", tensor=station_teme_pos)
        self.station_id = station_id
        self.n_passes = num_passes

    @property
    def num_params(self) -> int:
        return self.n_passes

    def forward(self, geometry_data, sensor_params, contact_indices) -> torch.Tensor:

        range_km = geometry_data["range"]

        if contact_indices is not None and self.n_passes > 0:
            assert torch.max(contact_indices) == self.n_passes
            bias_correction = sensor_params[contact_indices]
            return range_km + bias_correction
        else:
            return range_km
