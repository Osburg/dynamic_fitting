from typing import Tuple

import torch
from torch import Tensor
from torch.nn import MSELoss

from dldf.config import LossFunctionConfig


class LCModelLossFunction:
    """A class mimicking the loss function of LCModel."""

    def __init__(
        self, loss_function_config: LossFunctionConfig, *args, **kwargs
    ) -> None:
        self.mse_loss = MSELoss(reduction="sum")
        self.baseline_spline_regularization_curvature = (
            loss_function_config.baseline_spline_regularization_curvature
        )
        self.lineshape_kernel_regularization_curvature = (
            loss_function_config.lineshape_kernel_regularization_curvature
        )
        self.sigma_gamma_l = loss_function_config.sigma_gamma_l
        self.sigma_epsilon_l = loss_function_config.sigma_epsilon_l
        self.reconstruction_scaling = loss_function_config.reconstruction_scaling
        self.weights_time_axis = loss_function_config.weights_time_axis
        self.baseline_spline_regularization_slope = (
            loss_function_config.baseline_spline_regularization_slope
        )

    def __call__(
        self,
        x: Tensor,
        reconstruction: Tensor,
        baseline: Tensor,
        basis_lorentzian_dampings: Tensor,
        basis_frequency_shifts: Tensor,
        lineshape_kernel: Tensor,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Computes the loss function for the autoencoder model.

        Args:
            x (Tensor): Target data.
            reconstruction (Tensor): Reconstructed data.
            baseline (Tensor): Baseline for the data.
            basis_lorentzian_dampings (Tensor): Lorentzian dampings of the individual basis signals.
            basis_frequency_shifts (Tensor): Frequency shifts of the individual basis signals.
            lineshape_kernel (Tensor): Lineshape kernel.
            *args: Additional arguments (which will be ignored).
            **kwargs: Additional keyword arguments (which will be ignored).

        Returns:
            A tuple containing the individual loss components and the total loss.
        """

        # reconstruction loss
        weights = (
            torch.tensor(self.weights_time_axis, device=x.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        if self.reconstruction_scaling == "auto":
            with torch.no_grad():
                std_reconstruction = (
                    torch.std(input=torch.abs(reconstruction - x), dim=-1, keepdim=True)
                    + 1e-8
                )
        else:
            std_reconstruction = self.reconstruction_scaling
        reconstruction_loss = self.mse_loss(
            reconstruction.real * weights**0.5 / std_reconstruction,
            x.real * weights**0.5 / std_reconstruction,
        )

        # curvature baseline spline regularization
        baseline_spline_regularization_loss_curvature = torch.tensor(
            0.0, device=x.device
        )
        if self.baseline_spline_regularization_curvature is not None:
            if self.baseline_spline_regularization_curvature != 0:
                baseline_spline_regularization_loss_curvature = (
                    self.baseline_spline_regularization_curvature
                    * self.mse_loss(
                        baseline[..., 2:]
                        - 2 * baseline[..., 1:-1]
                        + baseline[..., :-2],
                        torch.zeros_like(baseline[..., 2:], device=x.device),
                    )
                )

        if self.lineshape_kernel_regularization_curvature is not None:
            if self.lineshape_kernel_regularization_curvature != 0:
                lineshape_kernel_regularization_loss_curvature = (
                    self.lineshape_kernel_regularization_curvature
                    * self.mse_loss(
                        lineshape_kernel[..., 2:]
                        - 2 * lineshape_kernel[..., 1:-1]
                        + lineshape_kernel[..., :-2],
                        torch.zeros_like(lineshape_kernel[..., 2:], device=x.device),
                    )
                    + self.mse_loss(
                        -2 * lineshape_kernel[..., 0] + lineshape_kernel[..., 1],
                        torch.zeros_like(lineshape_kernel[..., 0], device=x.device),
                    )
                    + self.mse_loss(
                        -2 * lineshape_kernel[..., -1] + lineshape_kernel[..., -2],
                        torch.zeros_like(lineshape_kernel[..., 0], device=x.device),
                    )
                    + self.mse_loss(
                        lineshape_kernel[..., 0],
                        torch.zeros_like(lineshape_kernel[..., 0], device=x.device),
                    )
                    + self.mse_loss(
                        lineshape_kernel[..., -1],
                        torch.zeros_like(lineshape_kernel[..., -1], device=x.device),
                    )
                )

        # gamma_l (damping loss)
        gamma_l_loss = (
            self.mse_loss(
                basis_lorentzian_dampings,
                torch.zeros_like(basis_lorentzian_dampings, device=x.device),
            )
            / self.sigma_gamma_l**2
        )

        # epsilon_l (frequency shift loss)
        epsilon_l_loss = (
            self.mse_loss(
                basis_frequency_shifts,
                torch.zeros_like(basis_frequency_shifts, device=x.device),
            )
            / self.sigma_epsilon_l**2
        )

        loss = (
            reconstruction_loss
            + baseline_spline_regularization_loss_curvature
            + gamma_l_loss
            + epsilon_l_loss
            + lineshape_kernel_regularization_loss_curvature
        )

        return (
            loss,
            reconstruction_loss,
            gamma_l_loss,
            epsilon_l_loss,
            baseline_spline_regularization_loss_curvature,
            lineshape_kernel_regularization_loss_curvature,
        )


class ComplexLCModelLossFunction(LCModelLossFunction):
    def __init__(
        self, loss_function_config: LossFunctionConfig, *args, **kwargs
    ) -> None:
        super().__init__(loss_function_config=loss_function_config)

    def __call__(
        self,
        x: Tensor,
        reconstruction: Tensor,
        baseline: Tensor,
        basis_lorentzian_dampings: Tensor,
        basis_frequency_shifts: Tensor,
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
            lineshape_kernel (Tensor): Lineshape kernel.
            *args: Additional arguments (which will be ignored).
            **kwargs: Additional keyword arguments (which will be ignored).

        Returns:
            A tuple containing the individual loss components and the total loss.
        """

        # reconstruction loss
        weights = (
            torch.tensor(self.weights_time_axis, device=x.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        if self.reconstruction_scaling == "auto":
            with torch.no_grad():
                std_reconstruction = (
                    torch.std(input=torch.abs(reconstruction - x), dim=-1, keepdim=True)
                    + 1e-8
                )
                print("scaling", torch.mean(std_reconstruction).detach().cpu().numpy())
        else:
            std_reconstruction = self.reconstruction_scaling
        reconstruction_loss_real = self.mse_loss(
            reconstruction.real * weights**0.5 / std_reconstruction,
            x.real * weights**0.5 / std_reconstruction,
        )
        reconstruction_loss_imag = self.mse_loss(
            reconstruction.imag * weights**0.5 / std_reconstruction,
            x.imag * weights**0.5 / std_reconstruction,
        )
        reconstruction_loss = reconstruction_loss_real + reconstruction_loss_imag

        baseline_spline_regularization_loss_curvature = torch.tensor(
            0.0, device=x.device
        )
        if self.baseline_spline_regularization_curvature is not None:
            if self.baseline_spline_regularization_curvature != 0:
                baseline_spline_regularization_loss_curvature = (
                    self.baseline_spline_regularization_curvature
                    * self.mse_loss(
                        baseline.real[..., 2:]
                        - 2 * baseline.real[..., 1:-1]
                        + baseline.real[..., :-2],
                        torch.zeros_like(baseline.real[..., 2:], device=x.device),
                    )
                    + self.baseline_spline_regularization_curvature
                    * self.mse_loss(
                        baseline.imag[..., 2:]
                        - 2 * baseline.imag[..., 1:-1]
                        + baseline.imag[..., :-2],
                        torch.zeros_like(baseline.imag[..., 2:], device=x.device),
                    )
                )

        baseline_spline_regularization_loss_slope = self.mse_loss(
            baseline.real[..., 1:] - baseline.real[..., :-1],
            torch.zeros_like(baseline.real[..., 1:], device=x.device),
        ) + self.mse_loss(
            baseline.imag[..., 1:] - baseline.imag[..., :-1],
            torch.zeros_like(baseline.imag[..., 1:], device=x.device),
        )
        baseline_spline_regularization_loss_slope *= (
            self.baseline_spline_regularization_slope
        )

        if self.lineshape_kernel_regularization_curvature is not None:
            if self.lineshape_kernel_regularization_curvature != 0:
                lineshape_kernel_regularization_loss_curvature = (
                    self.lineshape_kernel_regularization_curvature
                    * self.mse_loss(
                        lineshape_kernel[..., 2:]
                        - 2 * lineshape_kernel[..., 1:-1]
                        + lineshape_kernel[..., :-2],
                        torch.zeros_like(lineshape_kernel[..., 2:], device=x.device),
                    )
                    + self.mse_loss(
                        -2 * lineshape_kernel[..., 0] + lineshape_kernel[..., 1],
                        torch.zeros_like(lineshape_kernel[..., 0], device=x.device),
                    )
                    + self.mse_loss(
                        -2 * lineshape_kernel[..., -1] + lineshape_kernel[..., -2],
                        torch.zeros_like(lineshape_kernel[..., -1], device=x.device),
                    )
                    + self.mse_loss(
                        lineshape_kernel[..., 0],
                        torch.zeros_like(lineshape_kernel[..., 0], device=x.device),
                    )
                    + self.mse_loss(
                        lineshape_kernel[..., -1],
                        torch.zeros_like(lineshape_kernel[..., -1], device=x.device),
                    )
                )

        # gamma_l (damping loss)
        gamma_l_loss = (
            self.mse_loss(
                basis_lorentzian_dampings,
                torch.zeros_like(basis_lorentzian_dampings, device=x.device),
            )
            / self.sigma_gamma_l**2
        )

        # epsilon_l (frequency shift loss)
        epsilon_l_loss = (
            self.mse_loss(
                basis_frequency_shifts,
                torch.zeros_like(basis_frequency_shifts, device=x.device),
            )
            / self.sigma_epsilon_l**2
        )

        loss = (
            reconstruction_loss
            + baseline_spline_regularization_loss_curvature
            + baseline_spline_regularization_loss_slope
            + gamma_l_loss
            + epsilon_l_loss
            + lineshape_kernel_regularization_loss_curvature
        )

        return (
            loss,
            reconstruction_loss,
            gamma_l_loss,
            epsilon_l_loss,
            baseline_spline_regularization_loss_curvature,
            lineshape_kernel_regularization_loss_curvature,
        )
