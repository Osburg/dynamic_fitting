import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

from dldf.autoencoder import Autoencoder
from dldf.config import Configuration
from dldf.decoder.dynamic_decoder import DynamicDecoder, UniversalDynamicDecoder
from dldf.decoder.utils import get_decoder_class
from dldf.logger import MRSProgressLogger, SubjectQuantifier
from dldf.models.cnn import CNN, LatentSpaceTrafo
from dldf.mrs_utils.axis import Axis
from dldf.mrs_utils.basisset import Basisset
from dldf.mrs_utils.constants import GAMMA_D2, GAMMA_H1
from dldf.mrs_utils.container import MRSContainer
from dldf.mrs_utils.preprocessing import (
    B0Correction,
    Normalization2D,
    RotationToRealAxis,
)
from dldf.mrs_utils.transform import AugmentationTransform, IdentityTransform
from dldf.utils.misc import get_default_type


class Engine:
    config: (
        Configuration  # configuration of the engine as defined in a .json config file
    )
    time: Optional[torch.Tensor]  # time axis of the signal
    signal_length: Optional[int]  # length of the signals (in time/spectral domain)
    interval_bounds: Optional[List[int]]  # interval bounds along the ppm axis

    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.time = None  # Will be set when loading data via _load_and_preprocess_data
        self.signal_length = (
            None  # Will be set when loading data via _load_and_preprocess_data
        )
        self.n_repetitions = (
            None  # Will be set when loading data via _load_and_preprocess_data
        )
        self.interval_bounds = (
            None  # Will be set when loading data via _load_and_preprocess_data
        )

        # create the directory to save the models and checkpoints
        if self.config.io_config.saving_path is not None:
            Path(self.config.io_config.saving_path).mkdir(parents=True, exist_ok=True)
            Path(self.config.io_config.saving_path + "/checkpoints").mkdir(
                parents=True, exist_ok=True
            )

    def train(self) -> None:
        """Starts the training pipeline based on the configuration of the engine."""

        # load and preprocess the data.
        training_dataloader, validation_dataloader = self._load_and_preprocess_data(
            which="train"
        )

        # load the model from a checkpoint if specified, create a new one otherwise
        decoder = self._create_decoder_model()
        encoder, latent_space_trafo = self._create_encoder_model(
            index_dict=decoder.get_index_dict(collapse_amplitudes=True)
        )

        augmentation = AugmentationTransform(
            frequency_shift=self.config.preprocessor_config.augmentation_params[0],
            phase_shift=self.config.preprocessor_config.augmentation_params[1],
            lorentzian_damping=self.config.preprocessor_config.augmentation_params[2],
            noise_level=self.config.preprocessor_config.augmentation_params[3],
            gaussian_damping=self.config.preprocessor_config.augmentation_params[4],
            time=self.time,
            domain="frequency",
        )
        normalization = IdentityTransform()
        if self.config.preprocessor_config.normalize_signals is not None:
            normalization = Normalization2D(
                domain=self.config.preprocessor_config.normalize_signals,
                shift_mean_to_zero=self.config.preprocessor_config.shift_mean_to_zero,
                scaling_mode=self.config.preprocessor_config.normalization_scaling_mode,
            )
        autoencoder = Autoencoder(
            encoder=encoder,
            latent_space_trafo=latent_space_trafo,
            decoder=decoder,
            training_config=self.config.training_config,
            loss_function_config=self.config.loss_function_config,
            preprocessor_config=self.config.preprocessor_config,
            augmentation=augmentation,
            normalization=normalization,
            interval_bounds=self.interval_bounds,
        )

        if self.config.training_config.load_from_checkpoint:
            try:
                print("Trying to load the model from the latest checkpoint...")
                autoencoder = Autoencoder.load_from_checkpoint(
                    self.config.io_config.saving_path
                    + "/checkpoints/"
                    + self.config.training_config.load_from_checkpoint
                )
                print("Model loaded from latest checkpoint.")
            except:
                print(
                    "Model could not be loaded from latest checkpoint. Continuing with a new model."
                )

        # create trainer and start the training
        trainer = self._create_trainer()
        trainer.fit(
            model=autoencoder,
            train_dataloaders=training_dataloader,
            val_dataloaders=validation_dataloader,
        )

    def test(self) -> None:
        """Starts the test pipeline based on the configuration of the engine. The device, the batch size and the number
        of workers used are the same as in the training configuration."""

        # load and preprocess the data.
        test_dataloader, _ = self._load_and_preprocess_data(which="test")

        decoder = self._create_decoder_model()
        encoder, latent_space_trafo = self._create_encoder_model(
            index_dict=decoder.get_index_dict(collapse_amplitudes=True)
        )

        normalization = IdentityTransform()
        if self.config.preprocessor_config.normalize_signals is not None:
            normalization = Normalization2D(
                domain=self.config.preprocessor_config.normalize_signals,
                shift_mean_to_zero=self.config.preprocessor_config.shift_mean_to_zero,
                scaling_mode=self.config.preprocessor_config.normalization_scaling_mode,
            )

        autoencoder = Autoencoder(
            encoder=encoder,
            latent_space_trafo=latent_space_trafo,
            decoder=decoder,
            training_config=self.config.training_config,
            loss_function_config=self.config.loss_function_config,
            preprocessor_config=self.config.preprocessor_config,
            augmentation=IdentityTransform(),
            normalization=normalization,
            interval_bounds=self.interval_bounds,
        )

        # try to load the model from a lightning or pytorch checkpoint
        if self.config.io_config.saving_path is not None:
            try:
                print(
                    "Trying to load Autoencoder from latest checkpoint...",
                )
                autoencoder = Autoencoder.load_from_checkpoint(
                    self.config.io_config.saving_path
                    + "/checkpoints/"
                    + self.config.training_config.load_from_checkpoint
                )
                print("Autoencoder loaded from latest checkpoint.")
            except:
                raise RuntimeError(
                    "Autoencoder could not be loaded from latest checkpoint. Exiting."
                )

        autoencoder.encoder.eval()
        autoencoder.latent_space_trafo.eval()

        # create trainer and start testing
        tester = self._create_tester()
        tester.test(model=autoencoder, dataloaders=test_dataloader)

    def _load_and_preprocess_data(
        self, which: str = "train"
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Loads the training data, splits them into validation and training data and conducts
        multiple preprocessing steps.

        Args:
            which (str): Which data to load. Default is "train". Can take values "train" or "test".
        """

        # set the validation split and filepath according to the mode
        if which not in ["train", "test"]:
            raise ValueError("which must be either 'train' or 'test'.")
        if which == "train":
            data_path = self.config.io_config.training_data_path
            validation_split = self.config.training_config.validation_split
        else:  # i.e. which == "test"
            data_path = self.config.io_config.test_data_path
            validation_split = 0.0

        # normalize data and rotate signals to real axis according to the preprocessor configuration
        data = np.load(data_path)
        if self.config.preprocessor_config.rotate_signals_to_real_axis:
            data = np.swapaxes(data, 1, 2)
            data = RotationToRealAxis(mode="absolute")(data, update_trafo=True)
            data = np.swapaxes(data, 1, 2)

        # split into validation and training data
        idx = np.random.permutation(data.shape[0])
        validation_idx = idx[: int(data.shape[0] * validation_split)]
        training_idx = idx[int(data.shape[0] * validation_split) :]
        validation_data = data[validation_idx]
        training_data = data[training_idx]

        # prepare dataset and dataloader
        training_dataset = MRSContainer(
            data=training_data,
            ground_truth=None,
            metabolite_maps=None,
            device=self.config.pytorch_config.device,
            dwelltime=self.config.io_config.dwelltime,
            reference_frequency=self.config.io_config.reference_frequency,
        )
        validation_dataset = MRSContainer(
            data=validation_data,
            ground_truth=None,
            metabolite_maps=None,
            device=self.config.pytorch_config.device,
            dwelltime=self.config.io_config.dwelltime,
            reference_frequency=self.config.io_config.reference_frequency,
        )

        training_dataloader = DataLoader(
            training_dataset,
            batch_size=self.config.training_config.batch_size,
            shuffle=True,
            pin_memory=False,
            generator=torch.Generator(device=self.config.pytorch_config.device),
            num_workers=self.config.training_config.n_workers,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=self.config.training_config.batch_size,
            pin_memory=False,
            generator=torch.Generator(device=self.config.pytorch_config.device),
            num_workers=self.config.training_config.n_workers,
        )

        # create time axis and interval bounds according to the loaded data
        self.signal_length = training_dataset.signal_length
        self.n_repetitions = training_dataset.n_repetitions
        self.time = torch.linspace(
            0,
            self.config.io_config.dwelltime * (self.signal_length - 1),
            self.signal_length,
            requires_grad=False,
        ).to(self.config.pytorch_config.device)

        # create the interval bounds of the considered interval along the ppm axis
        gamma = GAMMA_H1 if self.config.io_config.nucleus == "H1" else GAMMA_D2
        time_axis = Axis.from_time_axis(
            time=self.time.detach().cpu().numpy(),
            b0=self.config.io_config.reference_frequency / gamma,
            nucleus=self.config.io_config.nucleus,
        )
        self.interval_bounds = [
            time_axis.to_index(
                self.config.preprocessor_config.ppm_bounds[0], domain="ppm"
            ),
            time_axis.to_index(
                self.config.preprocessor_config.ppm_bounds[1], domain="ppm"
            ),
        ]

        return training_dataloader, validation_dataloader

    def _create_trainer(self) -> pl.Trainer:
        """Creates a PyTorch Lightning Trainer object based on the configuration of the engine."""

        logger = WandbLogger(project=self.config.wandb_config.project_name)
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                save_last=True,
                every_n_epochs=self.config.validation_config.every_n_epochs,
                dirpath=self.config.io_config.saving_path + "/checkpoints/",
            ),
        ]
        if self.config.training_config.early_stopping > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="epoch_loss_val",
                    patience=self.config.training_config.early_stopping,
                )
            )
        if self.config.io_config.logging_path is not None:
            gamma = GAMMA_H1 if self.config.io_config.nucleus == "H1" else GAMMA_D2
            ppm_axis = Axis.from_time_axis(
                time=self.time.detach().cpu().numpy(),
                b0=self.config.io_config.reference_frequency / gamma,
                nucleus=self.config.io_config.nucleus,
            )

            basis = Basisset.from_matlab(
                path=self.config.io_config.basis_path,
                metabolite_names=self.config.preprocessor_config.basis_signal_names,
                conjugate_basis=self.config.preprocessor_config.conjugate_basis,
                dwelltime=self.config.io_config.dwelltime,
                reference_frequency=self.config.io_config.reference_frequency,
                nucleus=self.config.io_config.nucleus,
            )

            if self.config.decoder_config.normalize_basis:
                basis.normalize(
                    reference_peak_interval=self.config.decoder_config.reference_peak_interval
                )

            callbacks.append(
                MRSProgressLogger(
                    ppm_bounds=self.config.preprocessor_config.ppm_bounds,
                    save_dir=self.config.io_config.logging_path,
                    basis=basis,
                    ppm_axis=ppm_axis,
                )
            )

            if self.config.validation_config.plot_at_index is not None:
                # assuming exactly three non-zero indices
                callbacks.append(
                    SubjectQuantifier.from_matlab(
                        subject_path=self.config.io_config.full_subject_path_validation,
                        basis=basis,
                        interval_bounds=self.interval_bounds,
                        signal_length=self.signal_length,
                        save_dir=self.config.io_config.logging_path,
                        numerator=None,
                        denominator=None,
                        index=self.config.validation_config.plot_at_index[0],
                        ppm_bounds=self.config.preprocessor_config.ppm_bounds,
                        conjugate_subject_signals=self.config.validation_config.conjugate_full_subject_signals,
                        magnitude_image_path=None,
                        rotate_subject_signals_to_real_axis=self.config.preprocessor_config.rotate_signals_to_real_axis,
                        subject_name=self.config.validation_config.full_subject_name,
                        maximum_plot_ratio=None,
                        device=self.config.pytorch_config.device.index,
                        time_point=self.config.validation_config.plot_at_time_point,
                        b0_correction=(
                            B0Correction(time=self.time.detach().cpu().numpy())
                            if self.config.validation_config.b0_correction_path
                            is not None
                            else IdentityTransform()
                        ),
                        b0_shift_path=self.config.validation_config.b0_correction_path,
                        divide_by_mean=self.config.validation_config.divide_by_mean,
                    )
                )
                for i in range(1, len(self.config.validation_config.plot_at_index)):
                    callbacks.append(
                        SubjectQuantifier(
                            subject_spectra=callbacks[-1].subject_spectra,
                            subject_mask=callbacks[-1].subject_mask,
                            interval_bounds=callbacks[-1].interval_bounds,
                            basis=callbacks[-1].basis,
                            save_dir=callbacks[-1].save_dir,
                            numerator=None,
                            denominator=None,
                            magnitude_image=None,
                            index=self.config.validation_config.plot_at_index[i],
                            ppm_bounds=callbacks[-1].ppm_bounds,
                            subject_name=callbacks[-1].subject_name,
                            maximum_plot_ratio=None,
                            time_point=self.config.validation_config.plot_at_time_point,
                            divide_by_mean=self.config.validation_config.divide_by_mean,
                        )
                    )
                idx = [None, None, self.config.validation_config.plot_ratio_slice]
                magnitude_image_path = None
                if self.config.validation_config.plot_at_index[0] is not None:
                    magnitude_image_path = (
                        os.path.split(
                            self.config.io_config.full_subject_path_validation
                        )[0]
                        + f"/{self.config.validation_config.full_subject_name}_/maps/magnitude.nii"
                    )

                callbacks.append(
                    SubjectQuantifier.from_matlab(
                        subject_path=self.config.io_config.full_subject_path_validation,
                        basis=basis,
                        interval_bounds=self.interval_bounds,
                        signal_length=self.signal_length,
                        save_dir=self.config.io_config.logging_path,
                        numerator=self.config.validation_config.plot_ratio_numerator,
                        denominator=self.config.validation_config.plot_ratio_denominator,
                        index=idx,
                        ppm_bounds=None,
                        conjugate_subject_signals=self.config.validation_config.conjugate_full_subject_signals,
                        magnitude_image_path=magnitude_image_path,
                        rotate_subject_signals_to_real_axis=self.config.preprocessor_config.rotate_signals_to_real_axis,
                        subject_name=self.config.validation_config.full_subject_name,
                        maximum_plot_ratio=self.config.validation_config.maximum_plot_ratio,
                        device=self.config.pytorch_config.device.index,
                        time_point=self.config.validation_config.plot_at_time_point,
                        b0_correction=(
                            B0Correction(time=self.time.detach().cpu().numpy())
                            if self.config.validation_config.b0_correction_path
                            is not None
                            else IdentityTransform()
                        ),
                        b0_shift_path=self.config.test_config.b0_correction_path,
                        divide_by_mean=self.config.validation_config.divide_by_mean,
                    )
                )

        trainer = pl.Trainer(
            max_epochs=self.config.training_config.epochs,
            logger=logger,
            callbacks=callbacks,
            accelerator=self.config.pytorch_config.device.type,
            devices=(
                [self.config.pytorch_config.device.index]
                if self.config.pytorch_config.device.index is not None
                else 1
            ),
            enable_checkpointing=True,
            default_root_dir=self.config.io_config.saving_path,
            check_val_every_n_epoch=self.config.validation_config.every_n_epochs,
        )
        logger.save()

        return trainer

    def _create_tester(self) -> pl.Trainer:
        """Creates a PyTorch Lightning Trainer object based on the configuration of the engine for testing.
        The device used is the same as in the training configuration.
        """

        if self.config.io_config.logging_path is not None:
            basis = Basisset.from_matlab(
                path=self.config.io_config.basis_path,
                metabolite_names=self.config.preprocessor_config.basis_signal_names,
                conjugate_basis=self.config.preprocessor_config.conjugate_basis,
                dwelltime=self.config.io_config.dwelltime,
                reference_frequency=self.config.io_config.reference_frequency,
                nucleus=self.config.io_config.nucleus,
            )

            if self.config.decoder_config.normalize_basis:
                basis.normalize(
                    reference_peak_interval=self.config.decoder_config.reference_peak_interval
                )

            # create for each metabolite a callback to generate quantification
            Path(
                self.config.io_config.logging_path
                + "test_results/"
                + self.config.test_config.full_subject_name
            ).mkdir(parents=True, exist_ok=True)
            callbacks = [
                SubjectQuantifier.from_matlab(
                    subject_path=self.config.io_config.full_subject_path_test,
                    basis=basis,
                    interval_bounds=self.interval_bounds,
                    signal_length=self.signal_length,
                    save_dir=self.config.io_config.logging_path
                    + "test_results/"
                    + self.config.test_config.full_subject_name,
                    numerator=None,
                    denominator=None,
                    index=[None, None, None],
                    ppm_bounds=self.config.preprocessor_config.ppm_bounds,
                    conjugate_subject_signals=self.config.test_config.conjugate_full_subject_signals,
                    magnitude_image_path=None,
                    rotate_subject_signals_to_real_axis=self.config.preprocessor_config.rotate_signals_to_real_axis,
                    subject_name=self.config.test_config.full_subject_name,
                    maximum_plot_ratio=None,
                    device=self.config.pytorch_config.device.index,
                    b0_correction=(
                        B0Correction(time=self.time.detach().cpu().numpy())
                        if self.config.test_config.b0_correction_path is not None
                        else IdentityTransform()
                    ),
                    b0_shift_path=self.config.test_config.b0_correction_path,
                    divide_by_mean=False,
                )
            ]
            for i in range(len(self.config.test_config.plot_at_index)):
                callbacks.append(
                    SubjectQuantifier(
                        subject_spectra=callbacks[0].subject_spectra,
                        subject_mask=callbacks[0].subject_mask,
                        interval_bounds=self.interval_bounds,
                        basis=basis,
                        save_dir=self.config.io_config.logging_path
                        + "test_results/"
                        + self.config.test_config.full_subject_name,
                        numerator=None,
                        denominator=None,
                        magnitude_image=None,
                        index=self.config.test_config.plot_at_index[i],
                        ppm_bounds=self.config.preprocessor_config.ppm_bounds,
                        subject_name=self.config.test_config.full_subject_name,
                        maximum_plot_ratio=None,
                        divide_by_mean=False,
                    )
                )

        tester = pl.Trainer(
            accelerator=self.config.pytorch_config.device.type,
            devices=(
                [self.config.pytorch_config.device.index]
                if self.config.pytorch_config.device.index is not None
                else 1
            ),
            callbacks=callbacks,
            enable_checkpointing=False,
        )

        return tester

    def _create_encoder_model(
        self,
        train: bool = True,
        initialize_encoder: bool = True,
        index_dict: Dict[str, List[int]] = None,
    ) -> Tuple[nn.Module, nn.Module]:
        """Creates a neural network encoder model based on the configuration of the engine.
        Args:
            train (bool): Whether the model is created for training or testing.
            initialize_encoder (bool): Whether a feedforward should be done to initialize lazy layers of the encoder.
                For this, the engine's attribute interval_bounds must be set.
            index_dict (Dict[str, List[int]]): index dict of the encoder.

        Returns:
            Tuple[nn.Module, nn.Module]: The encoder model and a linear latent space transformation.
        """
        # latent space consists of dimensions for the basis signals + 3 dimensions for the phase shift,
        # lorentzian_damping and frequency shift
        # + the number of degrees of freedom for the baseline signal spline
        # + if the decoder outputs complex signals, the same number of degrees of freedom for the imaginary part of the
        # baseline signal spline

        encoder = CNN(
            in_channels=2,
            dropout=self.config.training_config.dropout if train else 0,
        ).to(self.config.pytorch_config.device)

        latent_space_trafo = LatentSpaceTrafo(
            index_dict=index_dict,
            shared_lineshape_kernel=self.config.encoder_config.shared_lineshape_kernel,
        )

        # Initialize the lazy layers
        if initialize_encoder:
            with torch.no_grad():
                x = torch.zeros(
                    size=(
                        1,
                        self.n_repetitions,
                        self.interval_bounds[1] - self.interval_bounds[0],
                    ),
                    dtype=get_default_type(domain="complex"),
                ).to(self.config.pytorch_config.device)
                latent_space_trafo(encoder(x))

        return encoder, latent_space_trafo

    def _create_decoder_model(self) -> Union[UniversalDynamicDecoder, DynamicDecoder]:
        """Creates the physical model decoder based on the configuration of the engine."""

        basis = Basisset.from_matlab(
            path=self.config.io_config.basis_path,
            metabolite_names=self.config.preprocessor_config.basis_signal_names,
            conjugate_basis=self.config.preprocessor_config.conjugate_basis,
            dwelltime=self.config.io_config.dwelltime,
            reference_frequency=self.config.io_config.reference_frequency,
            nucleus=self.config.io_config.nucleus,
        )

        if self.config.decoder_config.normalize_basis:
            basis.normalize(
                reference_peak_interval=self.config.decoder_config.reference_peak_interval
            )

        decoder_class = get_decoder_class(self.config.decoder_config.decoder_class)
        decoder = decoder_class(
            n_splines=self.config.decoder_config.n_splines,
            basis=torch.from_numpy(basis.fids).to(self.config.pytorch_config.device),
            time=self.time,
            interval_bounds=self.interval_bounds,
            ppm_bounds=self.config.preprocessor_config.ppm_bounds,
            metabolite_names=basis.metabolite_names,
            complex_output=self.config.decoder_config.complex_output,
            scaling_factors=self.config.decoder_config.scaling_factors,
            device=self.config.pytorch_config.device,
            dynamic_decoder_options=self.config.decoder_config.dynamic_decoder_options,
            lineshape_kernel_size=self.config.decoder_config.lineshape_kernel_size,
            repetition_axis=self.config.decoder_config.repetition_axis,
            shared_baseline_spline=self.config.decoder_config.shared_baseline_spline,
        )

        return decoder
