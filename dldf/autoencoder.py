from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from dldf.config import LossFunctionConfig, PreprocessorConfig, TrainingConfig
from dldf.decoder.dynamic_decoder import UniversalDynamicDecoder
from dldf.loss_functions.utils import get_loss_function_class
from dldf.models.cnn import CNN
from dldf.mrs_utils.transform import AugmentationTransform, IdentityTransform, Transform


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder: Union[CNN, nn.Module],
        latent_space_trafo: nn.Module,
        decoder: UniversalDynamicDecoder,
        training_config: TrainingConfig,
        loss_function_config: LossFunctionConfig,
        preprocessor_config: PreprocessorConfig,
        augmentation: Union[AugmentationTransform, IdentityTransform] = None,
        normalization: Transform = None,
        interval_bounds: List[int] = None,
    ) -> None:
        """Pytorch Lightning wrapper for the autoencoder model.

        Args:
            encoder (nn.Module): Encoder model.
            decoder (StandardModelDecoder): Decoder model.
            training_config (TrainingConfig): Configuration for the optimizer.
            loss_function_config (LossFunctionConfig): Configuration for the loss function.
            preprocessor_config (PreprocessorConfig): Configuration for the preprocessor.
            augmentation (AugmentationTransform, optional): Augmentation to apply to the input data. Defaults to None.
            normalization (Transform, optional): Normalization of the data that is called on data right before being
                sent to the encoder.
            interval_bounds (List[int], optional): Indices of the interval bounds of the spectral interval to be fed
            to the network. Defaults to [None, None].
        """
        super().__init__()
        self.encoder = encoder
        self.latent_space_trafo = latent_space_trafo
        self.decoder = decoder

        self.training_config = training_config
        self.loss_function_config = loss_function_config
        self.preprocessor_config = preprocessor_config
        self.augmentation = augmentation or IdentityTransform()
        self.normalization = normalization or IdentityTransform()

        # interval bounds of the spectral interval to be fed to the network
        self.interval_bounds = (
            interval_bounds if interval_bounds is not None else [None, None]
        )

        # create loss function fitting to the decoder output
        loss_function_class = get_loss_function_class(
            loss_function_type=loss_function_config.loss_function_type,
            complex_decoder_output=decoder.complex_output,
        )
        self.loss_function = loss_function_class(
            loss_function_config=loss_function_config,
            dynamic_decoder_options=decoder.dynamic_decoder_options,
            basis_signal_names=preprocessor_config.basis_signal_names,
        )

        self.save_hyperparameters()  # saves hyperparameters for checkpointing

    def forward(
        self, x: Tensor, crop_signal: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        """
        Forward pass through the autoencoder.

        Args:
            x (Tensor): Input data.
            crop_signal (bool, optional): Whether to crop the signal to the interval bounds. Defaults to True.
        """

        # forward pass through the encoder
        z = self.encoder(x)
        # forward pass through the linear layer
        z = self.latent_space_trafo(z)

        basis_lorentzian_dampings = (
            z[:, self.decoder.get_index_dict()["basis_lorentzian_damping"]]
            * self.decoder.scaling_factors["lorentzian_damping"]
        )
        basis_frequency_shifts = (
            z[:, self.decoder.get_index_dict()["basis_delta_f"]]
            * self.decoder.scaling_factors["frequency"]
        )
        lineshape_kernel = self.decoder._unpack(z)["lineshape_kernel"]

        # forward pass through the decoder
        basis_signals, baseline_signal, optionals = self.decoder(
            z, crop_signal=crop_signal
        )

        optionals["decoder_input"] = z

        return (
            basis_signals,
            baseline_signal,
            basis_lorentzian_dampings,
            basis_frequency_shifts,
            lineshape_kernel,
            optionals,
        )

    def training_step(self, batch, batch_idx) -> Dict:
        # augment and crop the data and only use the real part
        x = batch[0]
        x = self.augmentation(x)
        x = x[..., self.interval_bounds[0] : self.interval_bounds[1]]
        x = self.normalization(x)

        (
            basis_signals,
            baseline_signals,
            basis_lorentzian_dampings,
            basis_frequency_shifts,
            lineshape_kernel,
            optionals,
        ) = self.forward(x)

        kwargs = optionals

        reconstruction = basis_signals + baseline_signals
        (
            loss,
            reconstruction_loss,
            gamma_l_loss,
            epsilon_l_loss,
            baseline_spline_regularization_loss_curvature,
            lineshape_kernel_regularization_loss_curvature,
        ) = self.loss_function(
            x=x,
            reconstruction=reconstruction,
            baseline=baseline_signals,
            basis_lorentzian_dampings=basis_lorentzian_dampings,
            basis_frequency_shifts=basis_frequency_shifts,
            lineshape_kernel=lineshape_kernel,
            **kwargs,
        )

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "gamma_l_loss": gamma_l_loss,
            "epsilon_l_loss": epsilon_l_loss,
            "baseline_spline_regularization_loss_curvature": baseline_spline_regularization_loss_curvature,
            "lineshape_kernel_regularization_loss_curvature": lineshape_kernel_regularization_loss_curvature,
            "basis_signals": basis_signals,
            "baseline_signals": baseline_signals,
            "input": x,
        }

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                (
                    int(self.training_config.milestones[i])
                    if self.training_config.milestones[i] >= 1
                    else int(
                        self.training_config.milestones[i] * self.training_config.epochs
                    )
                )
                for i in range(len(self.training_config.milestones))
            ],
            gamma=self.training_config.reduce_learning_rate,
        )
        return [optimizer], [learning_rate_scheduler]

    def validation_step(self, batch, batch_idx) -> Dict:
        """Evaluating spectra from the validation set elementwise in no_grad mode and without augmentation."""

        with torch.no_grad():
            x = batch[0]
            x = x[..., self.interval_bounds[0] : self.interval_bounds[1]]
            x = self.normalization(x)

            (
                basis_signals,
                baseline_signals,
                basis_lorentzian_dampings,
                basis_frequency_shifts,
                lineshape_kernel,
                optionals,
            ) = self.forward(x)

            kwargs = optionals

            reconstruction = basis_signals + baseline_signals
            (
                loss,
                reconstruction_loss,
                gamma_l_loss,
                epsilon_l_loss,
                baseline_spline_regularization_loss_curvature,
                lineshape_kernel_regularization_loss_curvature,
            ) = self.loss_function(
                x=x,
                reconstruction=reconstruction,
                baseline=baseline_signals,
                basis_lorentzian_dampings=basis_lorentzian_dampings,
                basis_frequency_shifts=basis_frequency_shifts,
                lineshape_kernel=lineshape_kernel,
                **kwargs,
            )

            return {
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "gamma_l_loss": gamma_l_loss,
                "epsilon_l_loss": epsilon_l_loss,
                "baseline_spline_regularization_loss_curvature": baseline_spline_regularization_loss_curvature,
                "lineshape_kernel_regularization_loss_curvature": lineshape_kernel_regularization_loss_curvature,
                "basis_signals": basis_signals,
                "baseline_signals": baseline_signals,
                "input": x,
            }

    def test_step(self, batch, batch_idx) -> Dict:
        return self.validation_step(batch, batch_idx)
