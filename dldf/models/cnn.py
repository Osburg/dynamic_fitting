from typing import Dict, List

import torch
from torch import Tensor, nn
from torch.nn import LazyLinear


class CNN(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.0) -> None:
        """CNN part of the encoder with 9 convolutional layers and ReLU activation function.

        Args:
            in_channels (int): number of input channels
            dropout (bool): dropout for the convolutional layers. Currently removed from the implementation of the CNN.
        """
        super().__init__()

        self._in_channels = in_channels
        self.dropout = dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, (3, 5), stride=(1, 1), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 5), stride=(1, 1), padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (3, 5), stride=(1, 1), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 5), stride=(1, 1), padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3, 5), stride=(1, 1), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 5), stride=(1, 1), padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 5), stride=(1, 2), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 5), stride=(1, 2), padding=(1, 2)),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Expects a tensor of shape (batch_size, n_channels, n_samples)"""
        x_real = x.real
        x_imag = x.imag
        output_shape = (x_real.shape[0], x_real.shape[1], -1)
        output = self.cnn(torch.cat((x_real.unsqueeze(1), x_imag.unsqueeze(1)), dim=1))
        # output = self.cnn(torch.cat((x_real, x_imag), dim=1))
        output = output.reshape(output_shape)
        return output

    @property
    def input_channels(self):
        return self._in_channels


class LatentSpaceTrafo(nn.Module):
    """Transforms the latent space of the autoencoder to the parameters of the LCModelCostFunction. Separate linear
    layers are used for the independent and dependent parameters. The independent parameters are repeated delta_f,
    damping (Lorentzian) and amplitudes (whose time evolution is described by a model, e.g. a spline) and lineshape
    kernel
    """

    def __init__(
        self, index_dict: Dict[str, List[int]], shared_lineshape_kernel: bool
    ) -> None:
        super().__init__()

        self.index_dict = index_dict
        self.shared_lineshape_kernel = shared_lineshape_kernel

        self.n_repetition = len(index_dict["delta_phi"])

        self.n_independent = (
            len(index_dict["delta_phi"])
            + len(index_dict["delta_phi_1"])
            + len(index_dict["basis_delta_f"])
            + len(index_dict["basis_lorentzian_damping"])
            + len(index_dict["baseline_spline_values_real"])
            + len(index_dict["baseline_spline_values_imag"])
        )
        self.n_dependent = len(index_dict["delta_f"]) + len(index_dict["amplitudes"])
        if self.shared_lineshape_kernel:
            self.n_dependent += len(index_dict["lineshape_kernel"]) // self.n_repetition
        else:
            self.n_independent += len(index_dict["lineshape_kernel"])
        assert (
            self.n_independent % self.n_repetition == 0
        ), "The number of independent parameters must be a multiple of the number of repetitions."

        self.independent_linear = nn.Sequential(
            LazyLinear(self.n_independent // self.n_repetition),
        )

        self.dependent_linear = nn.Sequential(
            nn.Flatten(),
            LazyLinear(self.n_dependent),
        )

        self.independent_quantities = [
            "delta_phi",
            "delta_phi_1",
            "basis_delta_f",
            "basis_lorentzian_damping",
            "baseline_spline_values_real",
            "baseline_spline_values_imag",
        ]
        self.dependent_quantities = ["delta_f", "amplitudes"]

    def forward(self, x: Tensor) -> Tensor:
        """Expects a tensor of shape (n_batch, n_repetition, n_features)

        Returns:
            Tensor: The independent (shape: (n_batch, n_independent)) and dependent
            (shape: (n_batch, n_dependent)) parameters. As one tensor ordered by the keys in index_dict.
        """

        n_batch, n_repetition, n_features = x.shape
        independent = self.independent_linear(x)
        independent = torch.swapaxes(independent, -1, -2).reshape(
            independent.shape[0], -1
        )

        dependent = torch.swapaxes(x, -1, -2)
        dependent = self.dependent_linear(dependent)

        output = torch.empty(n_batch, sum(len(v) for v in self.index_dict.values()))
        idx0 = 0
        for key in self.independent_quantities:
            output[:, self.index_dict[key]] = independent[
                :, idx0 : (idx0 + len(self.index_dict[key]))
            ]
            idx0 += len(self.index_dict[key])
        if not self.shared_lineshape_kernel:
            lineshape_kernel = independent[
                :, idx0 : (idx0 + len(self.index_dict["lineshape_kernel"]))
            ]
            lineshape_kernel = torch.nn.functional.softplus(lineshape_kernel)
            lineshape_kernel = lineshape_kernel.reshape(
                lineshape_kernel.shape[0], -1, self.n_repetition
            )
            lineshape_kernel /= torch.sum(lineshape_kernel, dim=-2, keepdim=True)
            lineshape_kernel = lineshape_kernel.reshape(lineshape_kernel.shape[0], -1)
            output[:, self.index_dict["lineshape_kernel"]] = lineshape_kernel

        idx0 = 0
        for key in self.dependent_quantities:
            output[:, self.index_dict[key]] = dependent[
                :, idx0 : (idx0 + len(self.index_dict[key]))
            ]
            idx0 += len(self.index_dict[key])
        if self.shared_lineshape_kernel:
            lineshape_kernel = dependent[
                :, idx0 : (idx0 + len(self.index_dict["lineshape_kernel"]))
            ]
            lineshape_kernel = torch.nn.functional.softplus(lineshape_kernel)
            lineshape_kernel /= torch.sum(lineshape_kernel, dim=-1, keepdim=True)
            lineshape_kernel = (
                lineshape_kernel.unsqueeze(1)
                .repeat(1, n_repetition, 1)
                .swapaxes(-2, -1)
            )
            lineshape_kernel = lineshape_kernel.reshape(lineshape_kernel.shape[0], -1)
            output[:, self.index_dict["lineshape_kernel"]] = (
                lineshape_kernel  # .repeat(1, n_repetition)
            )

        return output
