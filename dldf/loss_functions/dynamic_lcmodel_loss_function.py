from typing import List, Tuple

import torch
from torch import Tensor

from dldf.config import LossFunctionConfig
from dldf.decoder.dynamics import (
    AgnosticDynamics,
    BSpline,
    NaturalSpline,
    get_dynamics_class,
)
from dldf.loss_functions.lcmodel_loss_function import (
    ComplexLCModelLossFunction,
    LCModelLossFunction,
)


class DynamicLCModelLossFunction(LCModelLossFunction):
    """A class mimicking the loss function of LCModel with an additional penalty along the kinetic axis."""

    def __init__(
        self,
        loss_function_config: LossFunctionConfig,
        dynamic_decoder_options: List[str],
        basis_signal_names: List[str],
        *args,
        **kwargs
    ) -> None:
        super().__init__(loss_function_config=loss_function_config)
        self.dynamic_spline_regularization_curvature = (
            loss_function_config.dynamic_spline_regularization_curvature
        )

        self.dynamic_penalty_delta_f = (
            "delta_f"
            in loss_function_config.dynamic_spline_regularization_target_quantities
        )
        self.dynamic_penalty_amplitudes = (
            "amplitudes"
            in loss_function_config.dynamic_spline_regularization_target_quantities
        )
        self.penalty_indices_amplitudes = []
        if self.dynamic_penalty_amplitudes:
            for i, name in enumerate(basis_signal_names):
                if get_dynamics_class(dynamic_decoder_options[name]["type"]) in [
                    AgnosticDynamics,
                    BSpline,
                    NaturalSpline,
                ]:
                    self.penalty_indices_amplitudes.append(i)

    def __call__(
        self,
        x: Tensor,
        reconstruction: Tensor,
        baseline: Tensor,
        basis_lorentzian_dampings: Tensor,
        basis_frequency_shifts: Tensor,
        kinetic_axis_amplitudes: Tensor,
        kinetic_axis_delta_f: Tensor,
        lineshape_kernel: Tensor,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Computes the loss function for the autoencoder model.

        Args:
            x (Tensor): Target data .
            reconstruction (Tensor): Reconstructed data.
            baseline (Tensor): Baseline for the data.
            basis_lorentzian_dampings (Tensor): Lorentzian dampings of the individual basis signals.
            basis_frequency_shifts (Tensor): Frequency shifts of the individual basis signals.
            kinetic_axis_amplitudes (Tensor): Values of the dynamics for amplitudes (kinetic axis).
            kinetic_axis_delta_f (Tensor): Values of the dynamics for delta_f (kinetic axis).
            lineshape_kernel (Tensor): Lineshape kernel.
            *args: Additional arguments (which will be ignored).
            **kwargs: Additional keyword arguments (which will be ignored).

        Returns:
            A tuple containing the individual loss components and the total loss.
        """

        (
            loss,
            reconstruction_loss,
            gamma_l_loss,
            epsilon_l_loss,
            baseline_spline_regularization_loss_curvature,
            lineshape_kernel_regularization_loss_curvature,
        ) = super().__call__(
            x=x,
            reconstruction=reconstruction,
            baseline=baseline,
            basis_lorentzian_dampings=basis_lorentzian_dampings,
            basis_frequency_shifts=basis_frequency_shifts,
            lineshape_kernel=lineshape_kernel,
        )

        # dynamic spline regularization
        dynamic_spline_regularization_loss_curvature = torch.tensor(
            0.0, device=x.device
        )
        if self.dynamic_spline_regularization_curvature is not None:
            if self.dynamic_spline_regularization_curvature != 0:
                if self.dynamic_penalty_amplitudes:
                    dynamic_spline_regularization_loss_curvature += (
                        self.dynamic_spline_regularization_curvature
                        * self.mse_loss(
                            kinetic_axis_amplitudes[
                                ..., self.dynamic_penalty_amplitudes, 2:
                            ]
                            - 2 * kinetic_axis_amplitudes[..., 1:-1]
                            + kinetic_axis_amplitudes[
                                ..., self.dynamic_penalty_amplitudes, :-2
                            ],
                            torch.zeros_like(
                                kinetic_axis_amplitudes[
                                    ..., self.dynamic_penalty_amplitudes, 2:
                                ],
                                device=x.device,
                            ),
                        )
                    )
                if self.dynamic_penalty_delta_f:
                    dynamic_spline_regularization_loss_curvature += (
                        self.dynamic_spline_regularization_curvature
                        * self.mse_loss(
                            kinetic_axis_delta_f[..., 2:]
                            - 2
                            * kinetic_axis_delta_f[
                                ..., self.dynamic_penalty_amplitudes, 1:-1
                            ]
                            + kinetic_axis_delta_f[..., :-2],
                            torch.zeros_like(
                                kinetic_axis_delta_f[..., 2:], device=x.device
                            ),
                        )
                    )
        loss += dynamic_spline_regularization_loss_curvature

        return (
            loss,
            reconstruction_loss,
            gamma_l_loss,
            epsilon_l_loss,
            baseline_spline_regularization_loss_curvature,
            lineshape_kernel_regularization_loss_curvature,
        )


class DynamicComplexLCModelLossFunction(ComplexLCModelLossFunction):
    def __init__(
        self,
        loss_function_config: LossFunctionConfig,
        dynamic_decoder_options: List[str],
        basis_signal_names: List[str],
        *args,
        **kwargs
    ) -> None:
        super().__init__(loss_function_config=loss_function_config)
        self.dynamic_spline_regularization_curvature = (
            loss_function_config.dynamic_spline_regularization_curvature
        )

        self.dynamic_penalty_delta_f = (
            "delta_f"
            in loss_function_config.dynamic_spline_regularization_target_quantities
        )
        self.dynamic_penalty_amplitudes = (
            "amplitudes"
            in loss_function_config.dynamic_spline_regularization_target_quantities
        )
        self.penalty_indices_amplitudes = []
        if self.dynamic_penalty_amplitudes:
            for i, name in enumerate(basis_signal_names):
                if get_dynamics_class(
                    dynamic_decoder_options["amplitudes"][name]["type"]
                ) in [AgnosticDynamics, BSpline, NaturalSpline]:
                    self.penalty_indices_amplitudes.append(i)

    def __call__(
        self,
        x: Tensor,
        reconstruction: Tensor,
        baseline: Tensor,
        basis_lorentzian_dampings: Tensor,
        basis_frequency_shifts: Tensor,
        kinetic_axis_amplitudes: Tensor,
        kinetic_axis_delta_f: Tensor,
        lineshape_kernel: Tensor,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Computes the loss function for the autoencoder model.

        Args:
            x (Tensor): Target data.
            reconstruction (Tensor): Reconstructed data.
            baseline (Tensor): Complex.
            basis_lorentzian_dampings (Tensor): Lorentzian dampings of the individual basis signals.
            basis_frequency_shifts (Tensor): Frequency shifts of the individual basis signals.
            kinetic_axis_amplitudes (Tensor): Values of the dynamics for amplitudes (kinetic axis).
            kinetic_axis_delta_f (Tensor): Values of the dynamics for delta_f (kinetic axis).
            lineshape_kernel (Tensor): Lineshape kernel.
            *args: Additional arguments (which will be ignored).
            **kwargs: Additional keyword arguments (which will be ignored).

        Returns:
            A tuple containing the individual loss components and the total loss.
        """

        (
            loss,
            reconstruction_loss,
            gamma_l_loss,
            epsilon_l_loss,
            baseline_spline_regularization_loss_curvature,
            lineshape_kernel_regularization_loss_curvature,
        ) = super().__call__(
            x=x,
            reconstruction=reconstruction,
            baseline=baseline,
            basis_lorentzian_dampings=basis_lorentzian_dampings,
            basis_frequency_shifts=basis_frequency_shifts,
            lineshape_kernel=lineshape_kernel,
        )

        # dynamic spline regularization
        dynamic_spline_regularization_loss_curvature = torch.tensor(
            0.0, device=x.device
        )

        if self.dynamic_spline_regularization_curvature is not None:
            if self.dynamic_spline_regularization_curvature != 0:
                if self.dynamic_penalty_amplitudes:
                    dynamic_spline_regularization_loss_curvature += (
                        self.dynamic_spline_regularization_curvature
                        * self.mse_loss(
                            kinetic_axis_amplitudes[
                                ..., self.dynamic_penalty_amplitudes, 2:
                            ]
                            - 2
                            * kinetic_axis_amplitudes[
                                ..., self.dynamic_penalty_amplitudes, 1:-1
                            ]
                            + kinetic_axis_amplitudes[
                                ..., self.dynamic_penalty_amplitudes, :-2
                            ],
                            torch.zeros_like(
                                kinetic_axis_amplitudes[
                                    ..., self.dynamic_penalty_amplitudes, 2:
                                ],
                                device=x.device,
                            ),
                        )
                    )
                if self.dynamic_penalty_delta_f:
                    dynamic_spline_regularization_loss_curvature += (
                        self.dynamic_spline_regularization_curvature
                        * self.mse_loss(
                            kinetic_axis_delta_f[..., 2:]
                            - 2 * kinetic_axis_delta_f[..., 1:-1]
                            + kinetic_axis_delta_f[..., :-2],
                            torch.zeros_like(
                                kinetic_axis_delta_f[..., 2:], device=x.device
                            ),
                        )
                    )
        loss += dynamic_spline_regularization_loss_curvature

        return (
            loss,
            reconstruction_loss,
            gamma_l_loss,
            epsilon_l_loss,
            baseline_spline_regularization_loss_curvature,
            lineshape_kernel_regularization_loss_curvature,
        )
