from typing import Union

import numpy as np
import torch
from torch import Tensor

from dldf.mrs_utils.transform import IdentityTransform, Transform  # noqa: F401
from dldf.mrs_utils.utils import fft, fftshift, ifft, ifftshift

try:
    import hlsvdpropy
except:
    pass


class RotationToRealAxis(Transform):

    def __init__(self, mode: str = "index", index: int = None) -> None:
        """
        Args:
            mode (str): Mode of rotation. Can be 'index' or 'absolute'. If 'index', the index of the signal to be
                rotated to the real axis must be provided. If 'absolute', the signal will be rotated to the real axis
                at the position of maximum absolute value.
            index (int): Index of the signal to be rotated to the real axis. Only used if mode is 'index'.
        """
        self.index = index
        if mode not in ["index", "absolute"]:
            raise ValueError("Mode must be 'index' or 'absolute'.")
        self.mode = mode

        self.angles = 0.0

    def __call__(
        self, x: Union[np.ndarray, Tensor], update_trafo: bool = True
    ) -> Union[np.ndarray, Tensor]:
        """Rotates complex signals saved as an np.array or Tensor along the last axis. The value at index 'self.index'
        is rotated to the real axis. The same rotation is applied to all values of each signal.

        Args:
            x (Union[np.ndarray, Tensor]): Complex signals to be rotated.
            update_trafo (bool): If True, the rotation angles are stored in the object.

        Returns:
            Union[np.ndarray, Tensor]: Rotated complex signals.
        """

        if not isinstance(x, (np.ndarray, Tensor)):
            raise ValueError("x must be a numpy array or a torch tensor.")

        if self.mode == "index":
            if isinstance(x, np.ndarray):
                angles = np.angle(x[..., self.index])
                if update_trafo:
                    self.angles = angles
                return x * np.exp(-angles * 1j)[..., np.newaxis]
            if isinstance(x, Tensor):
                angles = torch.angle(x[..., self.index])
                if update_trafo:
                    self.angles = angles
                return x * torch.exp(-angles * 1j).unsqueeze(-1)
        else:
            if isinstance(x, np.ndarray):
                angles = np.angle(
                    np.take_along_axis(
                        x, np.argmax(np.abs(x), axis=-1)[..., np.newaxis], axis=-1
                    )
                )
                if update_trafo:
                    self.angles = angles
                return x * np.exp(-angles * 1j)
            if isinstance(x, Tensor):
                angles = torch.angle(
                    torch.take_along_axis(
                        x, torch.argmax(torch.abs(x), dim=-1)[..., None], dim=-1
                    )
                )
                if update_trafo:
                    self.angles = angles
                return x * torch.exp(-angles * 1j)

    def invert(
        self,
        x: Union[np.ndarray, Tensor],
        squeeze: Union[str, int] = True,
    ) -> Union[np.ndarray, Tensor]:
        """Inverts the last operation __call__() operation that was executed by this object.

        Args:
            x (Union[np.ndarray, Tensor]): Data the inverse trafo is applied to.
            squeeze (bool): if True, scaling and mean are squeezed before applying the inversion

        Returns:
            Union[np.ndarray, Tensor]: Inverted spectra.
        """
        assert squeeze in [True, False, "adaptive"]

        angles = self.angles
        ndim = len(x.shape) - len(angles.squeeze().shape)
        if isinstance(angles, Tensor):
            if squeeze == "adaptive":
                angles = torch.squeeze(angles)
                for i in range(ndim):
                    angles = angles.unsqueeze(-1)
            elif squeeze:
                angles = torch.squeeze(angles)
            factors = torch.exp(angles * 1j)
        else:
            if squeeze == "adaptive":
                angles = np.squeeze(angles)
                for i in range(ndim):
                    angles = angles[..., np.newaxis]
            elif squeeze:
                angles = np.squeeze(angles)
            factors = np.exp(angles * 1j)

        return x * factors


class Normalization1D(Transform):
    def __init__(
        self,
        domain: Union[float, str] = "frequency",
        shift_mean_to_zero: bool = True,
        scaling_mode: str = "max",
    ) -> None:
        """Normalizes input spectra to the maximum absolute value along the last axis.

        Args:
            domain (Union[float, str]): Domain of the input signals. Can be 'time' or 'frequency'.
            shift_mean_to_zero (bool): If True, the mean of the input signals is shifted to zero before normalization.
            scaling_mode (str): Mode of normalization. Can be 'max' or 'mean' or 'z-score'.
        """
        if domain not in ["time", "frequency"] and not isinstance(domain, float):
            raise ValueError("Domain must be 'time' or 'frequency' or a float.")
        self.domain = domain
        self.shift_mean_to_zero = shift_mean_to_zero

        # cache for inversion of the trafo
        self.mean_shift = 0.0
        self.scaling = 1.0

        # function that is normalized to one
        if scaling_mode not in ["max", "mean", "z-score"]:
            raise ValueError("Mode must be 'max' or 'mean' or 'z-score'.")
        self.scaling_mode = scaling_mode

    def __call__(
        self, x: Union[np.ndarray, Tensor], update_trafo: bool = True
    ) -> Union[np.ndarray, Tensor]:
        """Normalizes input spectra to the maximum absolute value along the last axis.
        If the domain is 'time', the input is transformed to the time domain before normalization.
        and then transformed back to the spectral domain. If the domain is a float, the input is
        scaled by this value.

        Args:
            x (Union[np.ndarray, Tensor]): Input spectra to be normalized.
            update_trafo (bool): If True, the scaling and mean are stored in the object.

        Returns:
            Union[np.ndarray, Tensor]: Normalized spectra.
        """
        # shift mean if activated
        if self.shift_mean_to_zero:
            if isinstance(x, np.ndarray):
                mean = np.mean(x, axis=-1, keepdims=True)
                x = x - mean
                if update_trafo:
                    self.mean = mean
            if isinstance(x, Tensor):
                mean = torch.mean(x, dim=-1, keepdim=True)
                x = x - mean
                if update_trafo:
                    self.mean = mean
        else:
            self.mean = 0.0

        if isinstance(self.domain, float):
            self.scaling = 1.0 / self.domain
            return x * self.domain

        if self.domain == "time":
            x = ifft(ifftshift(x, dim=-1), dim=-1)

        if isinstance(x, np.ndarray):
            type = x.dtype
            if self.scaling_mode == "max":
                scaling = np.max(np.abs(x), axis=-1)[..., np.newaxis]
            elif self.scaling_mode == "z-score":
                scaling = np.std(x, axis=-1)[..., np.newaxis]
            else:
                scaling = np.mean(np.abs(x), axis=-1)[..., np.newaxis]
            x = x / scaling
            x = x.astype(type)
            if update_trafo:
                self.scaling = scaling

        elif isinstance(x, Tensor):
            if self.scaling_mode == "max":
                scaling = torch.max(torch.abs(x), dim=-1)[0].unsqueeze(-1)
            elif self.scaling_mode == "z-score":
                scaling = torch.std(x, dim=-1).unsqueeze(-1)
            else:
                scaling = torch.mean(torch.abs(x), dim=-1).unsqueeze(-1)
            x = x / scaling
            if update_trafo:
                self.scaling = scaling

        if self.domain == "time":
            x = fft(fftshift(x, dim=-1), dim=-1)

        return x

    def invert(
        self,
        x: Union[np.ndarray, Tensor],
        scaling_only: bool = False,
        squeeze: Union[str, int] = True,
    ) -> Union[np.ndarray, Tensor]:
        """Inverts the last operation __call__() operation that was executed by this object.

        Args:
            x (Union[np.ndarray, Tensor]): Data the inverse trafo is applied to.
            scaling_only (bool): Only the scaling is inverted, if set to true
            squeeze (bool): if True, scaling and mean are squeezed before applying the inversion

        Returns:
            Union[np.ndarray, Tensor]: Inverted spectra.
        """
        assert squeeze in [True, False, "adaptive"]

        scaling = self.scaling
        mean = self.mean
        ndim = len(x.shape)
        if isinstance(scaling, Tensor):
            if squeeze == "adaptive":
                scaling = torch.squeeze(scaling)
                for i in range(ndim - 1):
                    scaling = scaling.unsqueeze(-1)
            elif squeeze:
                scaling = torch.squeeze(scaling)
        else:
            if squeeze == "adaptive":
                scaling = np.squeeze(scaling)
                for i in range(ndim - 1):
                    scaling = scaling[..., np.newaxis]
            elif squeeze:
                scaling = np.squeeze(scaling)
        if isinstance(mean, Tensor):
            if squeeze == "adaptive":
                mean = torch.squeeze(mean)
                for i in range(ndim - 1):
                    mean = mean.unsqueeze(-1)
            elif squeeze:
                mean = torch.squeeze(mean)
        else:
            if squeeze == "adaptive":
                mean = np.squeeze(mean)
                for i in range(ndim - 1):
                    mean = mean[..., np.newaxis]
            elif squeeze:
                mean = np.squeeze(mean)

        if scaling_only:
            return x * scaling
        else:
            return x * scaling + mean


class Normalization2D(Transform):
    def __init__(
        self,
        domain: Union[float, str] = "frequency",
        shift_mean_to_zero: bool = True,
        scaling_mode: str = "max",
    ) -> None:
        """Normalizes input spectra to the maximum absolute value along the last two axes.

        Args:
            domain (Union[float, str]): Domain of the input signals. Can be 'time' or 'frequency'.
            shift_mean_to_zero (bool): If True, the mean of the input signals is shifted to zero before normalization.
            scaling_mode (str): Mode of normalization. Can be 'max' or 'mean' or 'z-score'.
        """
        if domain not in ["time", "frequency"] and not isinstance(domain, float):
            raise ValueError("Domain must be 'time' or 'frequency' or a float.")
        self.domain = domain
        self.shift_mean_to_zero = shift_mean_to_zero

        # function that is normalized to one
        if scaling_mode not in ["max", "mean", "z-score"]:
            raise ValueError("Mode must be 'max' or 'mean' or 'z-score'.")
        self.scaling_mode = scaling_mode

        # cache for inversion of the trafo
        self.mean_shift = 0.0
        self.scaling = 1.0

    def __call__(
        self, x: Union[np.ndarray, Tensor], update_trafo: bool = True
    ) -> Union[np.ndarray, Tensor]:
        """Normalizes input spectra to the maximum absolute value along the last two axes.
        If the domain is 'time', the input is transformed to the time domain before normalization.
        and then transformed back to the spectral domain. If the domain is a float, the input is
        scaled by this value.

        Args:
            x (Union[np.ndarray, Tensor]): Input spectra to be normalized.
            update_trafo (bool): If True, the scaling and mean are stored in the object.

        Returns:
            Union[np.ndarray, Tensor]: Normalized spectra.
        """
        # shift mean if activated
        if self.shift_mean_to_zero:
            if isinstance(x, np.ndarray):
                mean = np.mean(x, axis=(-2, -1), keepdims=True)
                x = x - mean
                if update_trafo:
                    self.mean = mean
            if isinstance(x, Tensor):
                mean = torch.mean(
                    torch.mean(x, dim=-1, keepdim=True), dim=-2, keepdim=True
                )
                x = x - mean
                if update_trafo:
                    self.mean = mean
        else:
            self.mean = 0.0

        if isinstance(self.domain, float):
            self.scaling = 1.0 / self.domain
            return x * self.domain

        if self.domain == "time":
            x = ifft(ifftshift(x, dim=-1), dim=-1)

        if isinstance(x, np.ndarray):
            type = x.dtype
            if self.scaling_mode == "max":
                scaling = np.max(np.abs(x), axis=(-2, -1))[..., np.newaxis, np.newaxis]
            elif self.scaling_mode == "z-score":
                scaling = np.std(x, axis=(-2, -1))[..., np.newaxis, np.newaxis]
            else:
                scaling = np.mean(np.abs(x), axis=(-2, -1))[..., np.newaxis, np.newaxis]
            x = x / scaling
            x = x.astype(type)
            if update_trafo:
                self.scaling = scaling

        elif isinstance(x, Tensor):
            if self.scaling_mode == "max":
                scaling = (
                    torch.max(torch.max(torch.abs(x), dim=-1)[0], dim=-1)[0]
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
            elif self.scaling_mode == "z-score":
                scaling = torch.std(x, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
            else:
                scaling = (
                    torch.mean(torch.mean(torch.abs(x), dim=-1), dim=-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
            x = x / scaling
            if update_trafo:
                self.scaling = scaling

        if self.domain == "time":
            x = fft(fftshift(x, dim=-1), dim=-1)

        return x

    def invert(
        self,
        x: Union[np.ndarray, Tensor],
        scaling_only: bool = False,
        squeeze: bool = True,
    ) -> Union[np.ndarray, Tensor]:
        """Inverts the last operation __call__() operation that was executed by this object.

        Args:
            x (Union[np.ndarray, Tensor]): Data the inverse trafo is applied to.
            scaling_only (bool): Only the scaling is inverted, if set to true
            squeeze (bool): if True, scaling and mean are squeezed before applying the inversion


        Returns:
            Union[np.ndarray, Tensor]: Inverted spectra.
        """
        scaling = self.scaling
        mean = self.mean
        ndim = len(x.shape)

        if isinstance(scaling, float):
            return x * scaling

        if isinstance(scaling, Tensor):
            if squeeze == "adaptive":
                scaling = torch.squeeze(scaling)
                for i in range(ndim - 1):
                    scaling = scaling.unsqueeze(-1)
            elif squeeze:
                scaling = torch.squeeze(scaling)
        else:
            if squeeze == "adaptive":
                scaling = np.squeeze(scaling)
                for i in range(ndim - 1):
                    scaling = scaling[..., np.newaxis]
            elif squeeze:
                scaling = np.squeeze(scaling)
        if isinstance(mean, Tensor):
            if squeeze == "adaptive":
                mean = torch.squeeze(mean)
                for i in range(ndim - 1):
                    mean = mean.unsqueeze(-1)
            elif squeeze:
                mean = torch.squeeze(mean)
        else:
            if squeeze == "adaptive":
                mean = np.squeeze(mean)
                for i in range(ndim - 1):
                    mean = mean[..., np.newaxis]
            elif squeeze:
                mean = np.squeeze(mean)

        if scaling_only:
            return x * scaling
        else:
            return x * scaling + mean


class B0Correction(Transform):
    """B0 correction for FIDs"""

    def __init__(self, time: Union[np.ndarray, Tensor], **kwargs) -> None:
        """Initializes the B0Correction object.

        Args:
            time (Union[np.ndarray, Tensor]): Time axis of the FIDs.
            **kwargs: Additional keyword arguments.
        """
        if not isinstance(time, (np.ndarray, Tensor)):
            raise ValueError("time must be a numpy array or a torch tensor.")
        elif isinstance(time, np.ndarray):
            self.time_np = time
            self.time_tensor = torch.from_numpy(time)
        elif isinstance(time, Tensor):
            self.time_np = time.cpu().numpy()
            self.time_tensor = time

    def __call__(
        self, x: Union[np.ndarray, Tensor], b0_shift: Union[np.ndarray, Tensor]
    ) -> Union[np.ndarray, Tensor]:
        """Applies a frequency shift according to b0_shift.

        Args:
            x (Union[np.ndarray, Tensor]): FID signals to be corrected of shape (n_fids, signal_length).
            b0_shift (Union[np.ndarray, Tensor]): B0 shift in Hz of shape (n_fids, 1).

        Returns:
            Union[np.ndarray, Tensor]: B0 corrected signals.
        """

        if not isinstance(x, (np.ndarray, Tensor)):
            raise ValueError("x must be a numpy array or a torch tensor.")

        if isinstance(x, np.ndarray):
            return x * np.exp(-1j * 2 * np.pi * self.time_np * b0_shift)
        if isinstance(x, Tensor):
            return x * torch.exp(-1j * 2 * np.pi * self.time_tensor * b0_shift)


class WaterRemoval(Transform):
    """A class that removes the water signal from MRS data"""

    def __init__(
        self, dwelltime: float, nsv_sought: int = 20, method: str = "hlsvd"
    ) -> None:
        """
        Args:
            dwelltime (float): Dwelltime of the input data.
            nsv_sought (int): Number of singular values sought in the water removal. Default is 20.
            method (str): Method for water removal. Currently, only hlsvd is supported. Default is 'hlsvd'.
        """
        if method not in ["hlsvd"]:
            raise ValueError("Method must be 'hlsvd'.")
        self.method = method
        self.dwelltime = dwelltime
        self.nsv_sought = nsv_sought

    def __call__(
        self, x: Union[np.ndarray, Tensor], frequency_threshold: float = 50.0
    ) -> np.ndarray:
        """Applies water removal to the input data.

        Args:
            x (np.ndarray): Input data of shape (n_signals, signal_length). Must be a numpy array, Tensors are
                not supported.
            frequency_threshold (float): Frequency threshold for water removal (in Hz?) Default is 50.

        Returns:
            np.ndarray: Water removed data.
        """
        if not isinstance(x, (np.ndarray, Tensor)):
            raise ValueError("x must be a numpy array or a torch tensor.")
        signal_length = x.shape[-1]
        n_signals = x.shape[0]

        for i in range(n_signals):
            result = hlsvdpropy.hlsvd(
                x[i, :], nsv_sought=self.nsv_sought, dwell_time=self.dwelltime
            )
            idx = np.where(np.abs(result[2]) < frequency_threshold)
            result = (
                len(idx),
                result[1],
                result[2][idx],
                result[3][idx],
                result[4][idx],
                result[5][idx],
            )
            fid = hlsvdpropy.create_hlsvd_fids(
                result=result,
                npts=signal_length,
                dwell=self.dwelltime,
                sum_results=True,
                convert=False,
            )
            x[i, :] = x[i, :] - fid

        return x
