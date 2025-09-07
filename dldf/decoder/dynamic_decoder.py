from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.fft import fft, fftshift

from dldf.decoder.dynamics import get_dynamics_class
from dldf.mrs_utils.b_spline import CubicBSpline
from dldf.utils import get_default_type


class DynamicDecoder(ABC):
    """
    A base class for a generalized version of the LCModelDecoder that accepts inputs along one additional dimension
    (repetition/time axis).
    """

    basis: Tensor
    time: Tensor
    interval_bounds: List[int]
    ppm_bounds: List[float]
    complex_output: bool
    device: torch.device
    scaling_factors: dict
    n_splines: int
    index_dict: Dict[str, Union[List[int], Dict[str, List[int]]]]

    # Constant tensors for signal calculations
    _first_order_phase_correction_axis: (
        Tensor  # freq. axis for 1-order phase correction
    )
    ppm_axis_points: Tensor  # spline positions of the knots along the ppm axis
    ppm_interval: Tensor  # axis for the spline evaluation

    # Cubic B-spline for the baseline
    b_spline: CubicBSpline

    shared_baseline_spline: bool

    @abstractmethod
    def _unpack(self, x: Tensor) -> Dict[str, Tensor]:
        pass

    def get_index_dict(
        self, collapse_amplitudes: bool = False, *args, **kwargs
    ) -> Dict[str, Union[List[int], Dict[str, List[int]]]]:
        """Returns the index dict.

        Args:
            collapse_amplitudes (bool): Whether to collapse the amplitudes into a single list.
        """
        index_dict = self.index_dict.copy()
        if collapse_amplitudes and isinstance(index_dict["amplitudes"], dict):
            new_amplitudes = []
            for name in index_dict["amplitudes"].keys():
                new_amplitudes += index_dict["amplitudes"][name]
            index_dict["amplitudes"] = new_amplitudes
        return index_dict

    @abstractmethod
    def _calculate_dynamics(
        self, inputs: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Calculates the time evolution of the amplitude, the frequency shift and the damping.
        The function is responsible to implement the physical nonnegativity constraints.

        Args:
            inputs (Dict[Tensor]): The input parameters, such as returned by the _unpack() method.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Dict]: The time evolution of the amplitude, the frequency shift and the
            lorentzian damping. In addition a dict with additional tensors can be returned.
        """
        pass

    def _calculate_basis_signals(
        self,
        amplitudes: Tensor,
        delta_f: Tensor,
        lorentzian_damping: Tensor,
        delta_phi_0: Tensor,
        delta_phi_1: Tensor,
        lineshape_kernel: Tensor,
    ) -> Tensor:
        """
        Calculates the time-domain signals for the basis signals (metabolites) given the amplitudes, global frequency
        shifts, global phase shifts, and global damping (Lorentzian, Gaussian) constants.

        Args:
            amplitudes (Tensor): amplitudes for the basis signals
            delta_f (Tensor): frequency shifts.
            lorentzian_damping (Tensor): Lorentzian damping constants.
            delta_phi_0 (Tensor): 0th order phase shifts.
            delta_phi_1 (Tensor): 1st order phase shifts.
            lineshape_kernel (Tensor): The lineshape kernels for the convolution of the basis signals.

        Returns:
            Tensor: time-domain signals for the basis signals
        """

        signals = 0.0
        for i in range(amplitudes.shape[2]):
            # calculate the signal for the i-th basis signal
            signal = (
                self.scaling_factors["basis_signals"]
                * amplitudes[:, :, i].unsqueeze(dim=-1)
                * self.basis[:, i]
            )
            # apply the frequency shift
            signal = signal * torch.exp(
                (
                    -2
                    * np.pi
                    * self.scaling_factors["frequency"]
                    * delta_f[:, :, i].unsqueeze(dim=-1)
                    * 1j
                )
                * self.time
            )
            # apply the Lorentzian damping
            signal = signal * torch.exp(
                (
                    -lorentzian_damping[:, :, i].unsqueeze(dim=-1)
                    * self.scaling_factors["lorentzian_damping"]
                )
                * self.time
            )
            signals += signal

        signals = fftshift(fft(signals, dim=-1), dim=-1)

        # apply the phase correction (0th and 1st order)
        signals = signals * torch.exp(
            (
                delta_phi_0 * self.scaling_factors["delta_phi"]
                + delta_phi_1
                * self.scaling_factors["delta_phi_1"]
                * self._first_order_phase_correction_axis
            )
            * 1j
        )

        # apply the lineshape kernel
        signal_shape = signals.shape
        signals_broadened = signals.clone()
        signals = signals.reshape(1, -1, signals.shape[-1])

        lineshape_kernel = lineshape_kernel.reshape(
            -1, lineshape_kernel.shape[-2], lineshape_kernel.shape[-1]
        )

        signals_broadened.real = (
            torch.nn.functional.conv1d(
                input=signals.real,
                weight=lineshape_kernel,
                groups=signals.shape[1],
                padding="same",
            )
            .squeeze()
            .reshape(signal_shape)
        )
        signals_broadened.imag = (
            torch.nn.functional.conv1d(
                input=signals.imag,
                weight=lineshape_kernel,
                groups=signals.shape[1],
                padding="same",
            )
            .squeeze()
            .reshape(signal_shape)
        )

        return signals_broadened

    def _calculate_baseline_signals(self, baseline_spline_values: Tensor) -> Tensor:
        """
        Args:
            baseline_spline_values (Tensor)
        """

        return self.b_spline(
            baseline_spline_values.swapaxes(0, 2), evaluate_extended_interval=True
        ).swapaxes(0, 2)

    def __call__(
        self,
        x: Tensor,
        return_kinetic_axis: bool = False,
        active_metabolites: List[str] = None,
        crop_signal: bool = True,
        *args,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Calculates the baseline and basis signals from the input parameters. The input tensor is expected to have the
        following shape: (batch_size, n_inputs).

        Args:
            x (Tensor): The input tensor.
            return_kinetic_axis (bool): Whether to additionally return the kinetic axis values.
            active_metabolites (List[str]): If not None, only the specified metabolites are considered. This is only
                relevant for non-standard usage and is not made of usually.
            crop_signal (bool): Whether to crop the signal to the user-defined ppm range. Defaults to True.
        """

        inputs = self._unpack(x)

        # calculate time evolution from the spline values of the dynamic spline
        amplitudes, delta_f, optionals = self._calculate_dynamics(inputs)

        if active_metabolites is not None:
            idxs = [self.metabolite_names.index(name) for name in active_metabolites]
            for idx in range(amplitudes.shape[-1]):
                if idx not in idxs:
                    amplitudes[:, :, idx] = 0.0

        # calculate the basis signals
        basis_signals = self._calculate_basis_signals(
            amplitudes=amplitudes,
            delta_f=delta_f + inputs["basis_delta_f"],
            lorentzian_damping=inputs["basis_lorentzian_damping"],
            delta_phi_0=inputs["delta_phi"],
            delta_phi_1=inputs["delta_phi_1"],
            lineshape_kernel=inputs["lineshape_kernel"],
        )

        # calculate the baseline signal
        baseline_signals = self._calculate_baseline_signals(
            inputs["baseline_spline_values_real"]
        )

        if self.complex_output:
            baseline_signals = baseline_signals + 1j * self._calculate_baseline_signals(
                inputs["baseline_spline_values_imag"]
            )

        # crop to the user-defined ppm range
        if crop_signal:
            basis_signals = basis_signals[
                :, :, self.interval_bounds[0] : self.interval_bounds[1]
            ]
            baseline_signals = baseline_signals[
                :, :, self.interval_bounds[0] : self.interval_bounds[1]
            ]

        if self.shared_baseline_spline:
            baseline_signals[...] = torch.mean(baseline_signals, dim=-2, keepdim=True)

        return (
            basis_signals,
            baseline_signals,
            optionals,
        )


class UniversalDynamicDecoder(DynamicDecoder):
    """A dynamic decoder that allows for arbitrary parametric functions to be used for the time evolution of the
    parameters."""

    def __init__(
        self,
        basis: Tensor,
        time: Tensor,
        interval_bounds: List[int],
        ppm_bounds: List[float],
        n_splines: int,
        complex_output: bool,
        device: torch.device,
        metabolite_names: List[str],
        lineshape_kernel_size: float,
        dynamic_decoder_options: Dict[str, Any] = None,
        scaling_factors: dict = None,
        repetition_axis: Tensor = None,
        shared_baseline_spline: bool = False,
        **kwargs,
    ):
        """Initializes the decoder object.

        Args:
            basis (Basisset): The basis set for the metabolites.
            time (Tensor): The time axis of the signals.
            interval_bounds (List[int]): The bounds of the interval to be used for the reconstruction.
            ppm_bounds (List[float]): The bounds of the ppm interval to be used for the reconstruction.
            n_splines (int): The number of spline values to be used for the baseline.
            complex_output (bool): Whether the output is complex or real.
            device (torch.device): The device to use for the calculations.
            metabolite_names (List[str]): The names of the metabolites. Defaults to None.
            lineshape_kernel_size (float): The size of the kernel used for the convolution of the basis signals.
            dynamic_decoder_options (Dict[str, int]): The dictionary containing the quantity as key and the number of
                splines used to model the time evolution of this quantity as item. Should have the keys "delta_f",
                "lorentzian_damping", "amplitudes".
            scaling_factors (dict, optional): Scaling factors for the input parameters. Defaults to None.
            repetition_axis (List[float]): Tensor containing the acquisition times of each repetition in minutes.
                The length of the tensor is used to determine the number of repetitions. Defaults to None.
            shared_baseline_spline (bool): Whether to use a shared baseline for all metabolites. Defaults to False.
        """

        self.basis = basis
        self.basis.requires_grad = False
        self.n_basis = basis.shape[1]

        self.n_splines = n_splines

        self.time = time
        self.interval_bounds = interval_bounds
        self.ppm_bounds = ppm_bounds
        self.complex_output = complex_output

        # Constant axes ...
        # ... for the 1st order phase correction
        self._first_order_phase_correction_axis = (
            torch.linspace(0.0, 1.0, len(time), requires_grad=False)
            .to(device)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )
        # ... for the time evolution along the kinetic axis
        self.n_repetitions = len(repetition_axis)
        self._repetition_axis = torch.tensor(
            repetition_axis,
            dtype=get_default_type(domain="real", framework="torch"),
            requires_grad=False,
        ).to(device) / (7 * self.n_repetitions)
        self._dynamic_spline_axis = torch.linspace(
            torch.min(self._repetition_axis),
            torch.max(self._repetition_axis),
            100,
            requires_grad=False,
        ).to(device)
        # ... for baseline spline
        interval_length = self.interval_bounds[1] - self.interval_bounds[0]
        if n_splines == 2:
            dx = 0.0
        elif n_splines == 3:
            dx = (self.ppm_bounds[1] - self.ppm_bounds[0]) / 2
        else:
            dx = (self.ppm_bounds[1] - self.ppm_bounds[0]) / (n_splines - 3)
        self.ppm_axis_points = torch.linspace(
            -dx + self.ppm_bounds[0], self.ppm_bounds[1] + dx, n_splines
        ).to(device)
        self.ppm_interval = torch.linspace(
            self.ppm_bounds[0], self.ppm_bounds[1], interval_length
        ).to(device)

        # set lineshape kernel size
        self.lineshape_kernel_length = int(
            interval_length
            / (self.ppm_bounds[1] - self.ppm_bounds[0])
            * lineshape_kernel_size
            * 2
            + 1
        )

        self.metabolite_names = metabolite_names

        self.scaling_factors = {
            "amplitudes": 1e-2,
            "delta_f": 1,
            "lorentzian_damping": 1,
            "baseline_spline_values": 1e-1,
            "delta_phi": 1,
            "delta_phi_1": 1,
        }
        if scaling_factors is not None:
            self.scaling_factors = scaling_factors

        # create dict containing the dynamic models for the metabolites, delta_f and damping (Lorentzian)
        self.dynamics_dict = {
            name: get_dynamics_class(
                dynamic_decoder_options["amplitudes"][name]["type"]
            )(
                repetition_axis=self._repetition_axis,
                kinetic_axis=self._dynamic_spline_axis,
                knot_positions=torch.linspace(
                    torch.min(self._repetition_axis),
                    torch.max(self._repetition_axis),
                    dynamic_decoder_options["amplitudes"][name]["parameters"],
                ),  # only relevant for spline dynamics
                apply_softplus=(
                    False
                    if dynamic_decoder_options["amplitudes"][name]["type"] in ["Linear"]
                    else True
                ),  # only relevant for spline dynamics and constants
            )
            for name in self.metabolite_names
        }
        self.dynamics_dict["delta_f"] = get_dynamics_class(
            dynamic_decoder_options["delta_f"]["type"],
        )(
            repetition_axis=self._repetition_axis,
            kinetic_axis=self._dynamic_spline_axis,
            knot_positions=torch.linspace(
                torch.min(self._repetition_axis),
                torch.max(self._repetition_axis),
                dynamic_decoder_options["delta_f"]["parameters"],
            ),  # only relevant for spline dynamics
            apply_softplus=False,  # only relevant for spline dynamics and constants
        )

        self.index_dict = {}
        self.index_dict["delta_phi"] = list(np.arange(0, self.n_repetitions))
        self.index_dict["delta_phi_1"] = list(
            np.arange(self.n_repetitions, 2 * self.n_repetitions)
        )
        idx0 = 2 * self.n_repetitions
        self.index_dict["delta_f"] = list(
            np.arange(idx0, idx0 + self.dynamics_dict["delta_f"].n_parameters)
        )
        idx0 = self.index_dict["delta_f"][-1] + 1
        self.index_dict["amplitudes"] = {}
        for name in self.metabolite_names:
            self.index_dict["amplitudes"][name] = list(
                np.arange(
                    idx0,
                    idx0 + self.dynamics_dict[name].n_parameters,
                )
            )
            idx0 = self.index_dict["amplitudes"][name][-1] + 1

        idx0 = self.index_dict["amplitudes"][self.metabolite_names[-1]][-1] + 1
        self.index_dict["basis_delta_f"] = list(
            np.arange(idx0, idx0 + self.n_basis * self.n_repetitions)
        )
        idx0 = self.index_dict["basis_delta_f"][-1] + 1
        self.index_dict["basis_lorentzian_damping"] = list(
            np.arange(idx0, idx0 + self.n_basis * self.n_repetitions)
        )
        idx0 = self.index_dict["basis_lorentzian_damping"][-1] + 1
        self.index_dict["lineshape_kernel"] = list(
            np.arange(idx0, idx0 + self.lineshape_kernel_length * self.n_repetitions)
        )
        idx0 = self.index_dict["lineshape_kernel"][-1] + 1
        self.index_dict["baseline_spline_values_real"] = list(
            np.arange(idx0, idx0 + self.n_splines * self.n_repetitions)
        )
        if len(self.index_dict["baseline_spline_values_real"]) > 0:
            idx0 = self.index_dict["baseline_spline_values_real"][-1] + 1
        self.index_dict["baseline_spline_values_imag"] = (
            list(np.arange(idx0, idx0 + self.n_splines * self.n_repetitions))
            if self.complex_output
            else []
        )

        self.n_inputs = idx0
        if self.complex_output:
            self.n_inputs += (
                self.n_splines * self.n_repetitions
            )  # for the imaginary part of the splines

        self.dynamic_decoder_options = dynamic_decoder_options

        n_points = interval_bounds[1] - interval_bounds[0]
        self.b_spline = CubicBSpline(
            device=device,
            n_splines=n_splines,
            n_points=n_points,
            add_boundary_knots=True,
            extend_interval_by=[
                interval_bounds[0],
                basis.shape[0] - interval_bounds[1],
            ],
        )

        self.shared_baseline_spline = shared_baseline_spline

    def _unpack(self, x: Tensor) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """Unpacks the input tensor into the individual parameters.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Dict[str, Tensor]: The unpacked parameters.
        """
        delta_phi = torch.unsqueeze(x[:, self.index_dict["delta_phi"]], dim=-1)
        delta_phi_1 = torch.unsqueeze(x[:, self.index_dict["delta_phi_1"]], dim=-1)
        delta_f = torch.unsqueeze(x[:, self.index_dict["delta_f"]], dim=-1)

        amplitudes = {}
        for name in self.metabolite_names:
            amplitudes[name] = torch.unsqueeze(
                x[:, self.index_dict["amplitudes"][name]], dim=-1
            )

        basis_delta_f = x[:, self.index_dict["basis_delta_f"]]
        basis_delta_f = basis_delta_f.reshape(
            shape=(basis_delta_f.shape[0], self.n_basis, self.n_repetitions)
        ).swapaxes(-1, -2)

        basis_lorentzian_damping = x[:, self.index_dict["basis_lorentzian_damping"]]
        basis_lorentzian_damping = basis_lorentzian_damping.reshape(
            shape=(basis_lorentzian_damping.shape[0], self.n_basis, self.n_repetitions)
        ).swapaxes(-1, -2)

        lineshape_kernel = x[:, self.index_dict["lineshape_kernel"]]
        lineshape_kernel = (
            lineshape_kernel.reshape(
                shape=(
                    lineshape_kernel.shape[0],
                    self.lineshape_kernel_length,
                    self.n_repetitions,
                )
            )
            .swapaxes(-1, -2)
            .unsqueeze(-2)
        )

        baseline_spline_values_real = x[
            :, self.index_dict["baseline_spline_values_real"]
        ]
        baseline_spline_values_real = baseline_spline_values_real.reshape(
            shape=(
                baseline_spline_values_real.shape[0],
                self.n_splines,
                self.n_repetitions,
            )
        ).swapaxes(-1, -2)

        if self.complex_output:
            baseline_spline_values_imag = x[
                :, self.index_dict["baseline_spline_values_imag"]
            ]
            baseline_spline_values_imag = baseline_spline_values_imag.reshape(
                shape=(
                    baseline_spline_values_imag.shape[0],
                    self.n_splines,
                    self.n_repetitions,
                )
            ).swapaxes(-1, -2)
        else:
            baseline_spline_values_imag = None

        return {
            "delta_phi": delta_phi,
            "delta_phi_1": delta_phi_1,
            "delta_f": delta_f,
            "amplitudes": amplitudes,
            "basis_delta_f": basis_delta_f,
            "basis_lorentzian_damping": basis_lorentzian_damping,
            "baseline_spline_values_real": baseline_spline_values_real,
            "baseline_spline_values_imag": baseline_spline_values_imag,
            "lineshape_kernel": lineshape_kernel,
        }

    def _calculate_dynamics(
        self, inputs: Dict[str, Union[Tensor, Dict[str, Tensor]]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict]:

        amplitudes = []
        kinetic_axis_amplitudes = []
        for name in self.metabolite_names:
            result = self.dynamics_dict[name](params=inputs["amplitudes"][name][..., 0])
            amplitudes.append(result[0])
            kinetic_axis_amplitudes.append(result[1])
        amplitudes = torch.stack(amplitudes, dim=-1)
        kinetic_axis_amplitudes = torch.stack(kinetic_axis_amplitudes, dim=-1)
        kinetic_axis_amplitudes = torch.swapaxes(kinetic_axis_amplitudes, -1, -2)

        out_time_points = (
            []
        )  # kinetic axis evaluated at the time points for which data is provided
        out_kinetic_axis = []  # ḱinetic axis parameters
        for name in [
            "delta_f",
        ]:
            result = self.dynamics_dict[name](params=inputs[name][..., 0])
            out_time_points.append(result[0].unsqueeze(dim=-1))
            out_kinetic_axis.append(torch.swapaxes(result[1].unsqueeze(dim=-1), -1, -2))
        delta_f = out_time_points[0]

        return (
            amplitudes,
            delta_f,
            {
                "kinetic_axis_amplitudes": kinetic_axis_amplitudes,
                "kinetic_axis_delta_f": out_kinetic_axis[0],
            },
        )
