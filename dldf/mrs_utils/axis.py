import warnings
from typing import Optional

import numpy as np
from numpy.fft import fftfreq, fftshift
from typing_extensions import Self

from dldf.mrs_utils.constants import CS_0_D2, CS_0_H1, GAMMA_D2, GAMMA_H1


class Axis:
    def __init__(self) -> None:
        """A class implementing functions to translate between axes in the time and frequency domains."""
        self.cs0 = None  # chemical shift in ppm of proton with gamma = GAMMA_H1 with respect to TMS
        self._frequency = None  # frequency axis
        self._time = None  # time axis
        self._ppm = None  # frequency/chemical shift axis in ppm
        self.b0 = None  # magnetic field strength in Tesla
        self._f0 = None  # frequency GAMMA_H! * B0 in Hz

    @property
    def dwelltime(self) -> float:
        """Get the dwell time of the time axis in seconds."""
        return self._time[1] - self._time[0]

    @property
    def n(self) -> int:
        """Get the number of points in the axes."""
        return len(self._time)

    @property
    def bandwidth(self) -> float:
        """Get the frequency bandwidth in Hz."""
        return 1 / self.dwelltime

    @property
    def frequency_spacing(self) -> float:
        """Get the frequency spacing of the frequency axis in Hz."""
        return self._frequency[1] - self._frequency[0]

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self._time = np.arange(len(value)) / (self.frequency_spacing * len(value) * 1e6)
        self._ppm = (self._frequency - self._f0) / self._f0 * 1e6 + self.cs0

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value
        self._frequency = (
            self._f0 + fftshift(fftfreq(self.n, self._time[1] - self._time[0])) * 1e-6
        )
        self._ppm = (self._frequency - self._f0) / self._f0 * 1e6 + self.cs0

    @property
    def ppm(self):
        return self._ppm

    @ppm.setter
    def ppm(self, value):
        self._ppm = value
        self._frequency = self._f0 + (value - self.cs0) / 1e6 * self._f0
        self._time = np.arange(len(self._frequency)) / (
            self.frequency_spacing * len(self._frequency) * 1e6
        )

    def to_index(self, value: float, domain: str = "time") -> int:
        """Convert a value to the corresponding index in the specified domain.

        Args:
            value (float): value to convert to index.
            domain (str): domain to convert the value to. Can be "time", "frequency" or "ppm".
        """

        if domain not in ["time", "frequency", "ppm"]:
            raise ValueError("Domain must be either 'time', 'frequency' or 'ppm'.")

        if domain == "time":
            if value < np.min(self._time) or value > np.max(self._time):
                warnings.warn(
                    "The value is outside the covered time interval. The closest value will be used."
                )
            return np.argmin(np.abs(self._time - value))
        if domain == "frequency":
            if value < np.min(self._frequency) or value > np.max(self._frequency):
                warnings.warn(
                    "The value is outside the covered frequency interval. The closest value will be used."
                )
            return np.argmin(np.abs(self._frequency - value))
        if domain == "ppm":
            if value < np.min(self._ppm) or value > np.max(self._ppm):
                warnings.warn(
                    "The value is outside the covered ppm interval. The closest value will be used."
                )
            return np.argmin(np.abs(self._ppm - value))

    def frequency_to_ppm(self, frequency: float, frequency_shift: bool = True) -> float:
        """Convert a frequency value to ppm.

        Args:
            frequency (float): frequency / frequency shift value in MHz / Hz.
            frequency_shift (bool): Optional. If True, the frequency value is a frequency shift. Default is True.
        """
        if frequency_shift:
            return frequency / self._f0
        else:
            return (frequency - self._f0) / self._f0 * 1e6 + self.cs0

    def ppm_to_frequency(self, ppm: float, frequency_shift: bool = True) -> float:
        """Convert a ppm value to frequency.

        Args:
            ppm (float): ppm value.
            frequency_shift (bool): Optional. If True, the frequency value is a frequency shift. Default is False.
                The returned value is in Hz for frequency_shift = True and in MHz for frequency_shift = False.
        """
        if frequency_shift:
            return ppm * self._f0
        else:
            return (ppm - self.cs0) * self._f0 * 1e-6 + self._f0

    @classmethod
    def from_time_axis(
        cls, time: np.ndarray, b0: Optional[float] = 3.0, nucleus: str = "H1"
    ) -> Self:
        """Generate an axis object from a time axis, which is a linspace of time values.

        Args:
            time (np.ndarray): time axis in seconds. Must be a 1D linspace of time values.
            b0 (float): Optional. Magnetic field strength in Tesla.
        """
        if nucleus not in ["H1", "D2"]:
            raise ValueError("Nucleus must be either 'H1' or 'D2'.")

        obj = cls()
        if nucleus == "D2":
            obj._f0 = GAMMA_D2 * b0
            obj.cs0 = CS_0_D2
        else:
            obj._f0 = GAMMA_H1 * b0
            obj.cs0 = CS_0_H1

        obj.b0 = b0
        obj._time = time
        obj._frequency = (
            obj._f0 + fftshift(fftfreq(obj.n, obj._time[1] - obj._time[0])) * 1e-6
        )
        obj._ppm = (obj._frequency - obj._f0) / obj._f0 * 1e6 + obj.cs0
        return obj

    @classmethod
    def from_frequency_axis(
        cls, frequency: np.ndarray, b0: Optional[float] = 3.0, nucleus: str = "H1"
    ) -> Self:
        """Generate an axis object from a frequency axis, which is a linspace of frequency values.

        Args:
            frequency (np.ndarray): frequency axis in MHz. Must be a 1D linspace of frequency values.
            b0 (float): Optional. Magnetic field strength in Tesla.
            nucleus (str): Optional. Nucleus the axis should be used for. Can be "H1" or "D2". Default is "H1".
        """
        if nucleus not in ["H1", "D2"]:
            raise ValueError("Nucleus must be either 'H1' or 'D2'.")

        obj = cls()
        obj.b0 = b0
        if nucleus == "D2":
            obj._f0 = GAMMA_D2 * b0
            obj.cs0 = CS_0_D2
        else:
            obj._f0 = GAMMA_H1 * b0
            obj.cs0 = CS_0_H1
        obj._frequency = frequency
        obj._time = np.arange(len(frequency)) / (
            obj.frequency_spacing * len(frequency) * 1e6
        )
        obj._ppm = (obj._frequency - obj._f0) / obj._f0 * 1e6 + obj.cs0
        return obj

    @classmethod
    def from_ppm_axis(
        cls, ppm: np.ndarray, b0: Optional[float] = 3.0, nucleus: str = "H1"
    ) -> Self:
        """Generate an axis object from a ppm axis, which is a linspace of ppm values.

        Args:
            ppm (np.ndarray): ppm axis. Must be a 1D linspace of ppm values.
            b0 (float): Optional. Magnetic field strength in Tesla.
            nucleus (str): Optional. Nucleus the axis should be used for. Can be either 'H1' or 'D2'.
        """
        if nucleus not in ["H1", "D2"]:
            raise ValueError("Nucleus must be either 'H1' or 'D2'.")

        obj = cls()
        obj.b0 = b0
        if nucleus == "D2":
            obj._f0 = GAMMA_D2 * b0
            obj.cs0 = CS_0_D2
        else:
            obj._f0 = GAMMA_H1 * b0
            obj.cs0 = CS_0_H1
        obj._ppm = ppm
        obj._frequency = obj._f0 + (ppm - obj.cs0) / 1e6 * obj._f0
        obj._time = np.arange(len(obj._frequency)) / (
            obj.frequency_spacing * len(obj._frequency) * 1e6
        )
        return obj
