from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from torch import Tensor

from dldf.mrs_utils.utils import fft, fftshift, ifft, ifftshift


class Transform(ABC):
    """Base class for all transforms. Must implement a __call__ method."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        return NotImplemented


class FourierTransform(Transform):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        """FFT of an input signal x"""
        return fftshift(fft(x))


class InverseFourierTransform(Transform):
    """Inverse FFT of an input signal x"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        return ifft(ifftshift(x))


class OneHotEncodingTransform(Transform):
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes

    def __call__(self, x: Tensor) -> Tensor:
        for x_ in x.flatten():
            if x_ not in range(self.n_classes):
                raise ValueError(
                    f"Class {x_} is not in the range [0, {self.n_classes})"
                )
        return torch.zeros((x.shape[0], self.n_classes), dtype=torch.float).scatter_(
            -1, index=torch.tensor(x), value=1
        )


class AugmentationTransform(Transform):

    def __init__(
        self,
        frequency_shift: float,
        phase_shift: float,
        lorentzian_damping: float,
        gaussian_damping: float,
        noise_level: float,
        time: Tensor,
        domain: str = "time",
    ) -> None:
        """Class that applied a random frequency shift, phase shift, damping and noise addition to an input signal.

        Args:
            frequency_shift (float): Maximum frequency shift in Hz
            phase_shift (float): Maximum phase shift in radians
            lorentzian_damping (float): Maximum lorentzian damping factor
            gaussian_damping (float): Maximum gaussian damping factor
            noise_level (float): Maximum noise level
            time (Tensor): Time tensor
            domain (str): Domain of the input signals to be transformed. Can be 'time' or 'frequency'.
        """
        super().__init__()

        self.frequency_shift = frequency_shift
        self.phase_shift = phase_shift
        self.lorentzian_damping = lorentzian_damping
        self.gaussian_damping = gaussian_damping
        self.noise_level = noise_level
        self.time = time
        if domain not in ["time", "frequency"]:
            raise ValueError("Domain must be 'time' or 'frequency'")
        self.domain = domain

    def __call__(self, x: Tensor) -> Tensor:
        frequency_shift = self.frequency_shift * (
            torch.rand(x.shape[:-1]) - 0.5
        ).unsqueeze(-1)
        phase_shift = (
            self.phase_shift * np.pi * (torch.rand(x.shape[:-1]) - 0.5).unsqueeze(-1)
        )
        lorentzian_damping = self.lorentzian_damping * torch.rand(
            x.shape[:-1]
        ).unsqueeze(-1)
        gaussian_damping = self.gaussian_damping * torch.rand(x.shape[:-1]).unsqueeze(
            -1
        )

        if self.domain == "frequency":
            x = ifft(ifftshift(x, dim=-1), dim=-1)

        x *= torch.exp(1j * (phase_shift + 2 * np.pi * frequency_shift * self.time))
        x *= torch.exp(-lorentzian_damping * self.time)
        x *= torch.exp(-gaussian_damping * self.time**2)
        x += self.noise_level * (torch.randn(x.shape) + 1j * torch.randn(x.shape))

        if self.domain == "frequency":
            x = fftshift(fft(x, dim=-1), dim=-1)

        return x


class IdentityTransform(Transform):
    """Transformation that applies the identity map to input tensors"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x

    def invert(
        self, x: Tensor, scaling_only: bool = False, squeeze: bool = False
    ) -> Tensor:
        return x


class LorentzianLineshapeTransform(Transform):
    """Applying a Lorentian lineshape according to the exp(-t/T) damping"""

    def __init__(self, T: float, time: Tensor, domain: str = "time") -> None:
        """
        Args:
            T (float): Damping factor
            time (Tensor or int): time axis according to the input to be transformed.
            domain (str): Domain of the input signals to be transformed. Can be 'time' or 'frequency'.

        """
        super().__init__()
        self.time = time
        self.T = T
        if domain not in ["time", "frequency"]:
            raise ValueError("Domain must be 'time' or 'frequency'")
        self.domain = domain

    def __call__(self, x: Tensor) -> Tensor:

        if self.domain == "frequency":
            x = ifft(ifftshift(x, dim=-1), dim=-1)

        x *= torch.exp(-self.time / self.T)

        if self.domain == "frequency":
            x = fftshift(fft(x, dim=-1), dim=-1)

        return x


class GaussianLineshapeTransform(Transform):
    """Applying a Lorentian lineshape according to the exp(-t**2/T**2) damping"""

    def __init__(self, T: float, time: Tensor, domain: str = "time") -> None:
        """
        Args:
            T (float): Damping factor
            time (Tensor or int): time axis according to the input to be transformed.
            domain (str): Domain of the input signals to be transformed. Can be 'time' or 'frequency'.

        """
        super().__init__()
        self.time = time
        self.T = T
        if domain not in ["time", "frequency"]:
            raise ValueError("Domain must be 'time' or 'frequency'")
        self.domain = domain

    def __call__(self, x: Tensor) -> Tensor:

        if self.domain == "frequency":
            x = ifft(ifftshift(x, dim=-1), dim=-1)

        x *= torch.exp(-self.time**2 / self.T**2)

        if self.domain == "frequency":
            x = fftshift(fft(x, dim=-1), dim=-1)

        return x


class VoigtLineshapeTransform(Transform):
    """Applying a Voigt lineshape according to the exp(-t/T_lorentz - t**2/T_gauss**2) damping"""

    def __init__(
        self, T_lorentz: float, T_gauss: float, time: Tensor, domain: str = "time"
    ) -> None:
        """
        Args:
            T_lorentz (float): Damping factor for the Lorentzian part
            T_gauss (float): Damping factor for the Gaussian part
            time (Tensor or int): time axis according to the input to be transformed.
            domain (str): Domain of the input signals to be transformed. Can be 'time' or 'frequency'.

        """
        super().__init__()
        self.time = time
        self.T_lorentz = T_lorentz
        self.T_gauss = T_gauss
        if domain not in ["time", "frequency"]:
            raise ValueError("Domain must be 'time' or 'frequency'")
        self.domain = domain

    def __call__(self, x: Tensor) -> Tensor:

        if self.domain == "frequency":
            x = ifft(ifftshift(x, dim=-1), dim=-1)

        x *= torch.exp(-self.time * (1 / self.T_lorentz + self.time / self.T_gauss**2))

        if self.domain == "frequency":
            x = fftshift(fft(x, dim=-1), dim=-1)

        return x
