import os
from pathlib import Path
from typing import Any, List, Tuple, Union

import mat73
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.image
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.fft import fft, fftshift
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from typing_extensions import Self

from dldf.autoencoder import Autoencoder
from dldf.mrs_utils.axis import Axis
from dldf.mrs_utils.basisset import Basisset
from dldf.mrs_utils.preprocessing import RotationToRealAxis
from dldf.utils.misc import get_default_type


class MRSProgressLogger(Callback):
    def __init__(
        self,
        ppm_bounds: List[float],
        basis: Basisset,
        save_dir: str = None,
        ppm_axis: Axis = None,
    ) -> None:
        """
        Args:
            save_dir (str): The directory to save the plots to.
            basis (Basisset): The metabolite basis set used for the model.
            ppm_bounds (List[float]): The bounds of the ppm axis.
            ppm_axis (Axis): ppm axis to be used for plots
        """

        self.epoch_loss = {
            "train": {
                "reconstruction_loss": [],
                "gamma_l_loss": [],
                "epsilon_l_loss": [],
                "baseline_spline_regularization_loss_curvature": [],
                "lineshape_kernel_regularization_loss_curvature": [],
            },
            "val": {
                "reconstruction_loss": [],
                "gamma_l_loss": [],
                "epsilon_l_loss": [],
                "baseline_spline_regularization_loss_curvature": [],
                "lineshape_kernel_regularization_loss_curvature": [],
            },
            "test": {
                "reconstruction_loss": [],
                "gamma_l_loss": [],
                "epsilon_l_loss": [],
                "baseline_spline_regularization_loss_curvature": [],
                "lineshape_kernel_regularization_loss_curvature": [],
            },
        }  # save (mean) metrics and quantities for each epoch
        self.step_loss = {
            "train": {
                "reconstruction_loss": [],
                "gamma_l_loss": [],
                "epsilon_l_loss": [],
                "baseline_spline_regularization_loss_curvature": [],
                "lineshape_kernel_regularization_loss_curvature": [],
            },
            "val": {
                "reconstruction_loss": [],
                "gamma_l_loss": [],
                "epsilon_l_loss": [],
                "baseline_spline_regularization_loss_curvature": [],
                "lineshape_kernel_regularization_loss_curvature": [],
            },
            "test": {
                "reconstruction_loss": [],
                "gamma_l_loss": [],
                "epsilon_l_loss": [],
                "baseline_spline_regularization_loss_curvature": [],
                "lineshape_kernel_regularization_loss_curvature": [],
            },
        }  # save metrics and quantities for each step of the current epoch
        self.validation_epoch = []

        self.save_dir = save_dir
        if self.save_dir is not None:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.ppm_bounds = ppm_bounds
        self.basis = basis
        self.ppm_axis = ppm_axis

    def on_train_start(self, trainer: Trainer, pl_module: Autoencoder) -> None:
        """Creates a plot of the basis signals used for the fitting."""

        fig = plt.figure(figsize=(10, 1.1 * len(self.basis)))
        ax = fig.add_subplot(111)
        for i in range(len(self.basis)):
            signal = self.basis.spectra[:, i]
            ax.plot(
                self.ppm_axis._ppm,
                signal.real - 1.5 * i,
                color="black",
                linestyle="solid",
            )
            ax.plot(
                self.ppm_axis._ppm,
                signal.imag - 1.5 * i,
                color="black",
                linestyle="dotted",
            )
            ax.text(
                self.ppm_bounds[1] + 0.25,
                -1.5 * i,
                self.basis.metabolite_names[i],
                fontsize=11,
                verticalalignment="center",
            )
        ax.set_xlabel("chemical shift [ppm]")
        ax.set_yticks([])
        ax.set_xlim(self.ppm_bounds[0], self.ppm_bounds[1])
        ax.set_title("Basis signals")
        ax.invert_xaxis()
        for spine in fig.gca().spines.values():
            spine.set_visible(False)
        plt.savefig(f"{self.save_dir}/basis_signals.png")
        plt.close(fig)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: Autoencoder,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pl_module.log(
            "epoch_loss_train", np.mean(outputs["loss"].detach().cpu().numpy())
        )
        self.step_loss["train"]["reconstruction_loss"].append(
            outputs["reconstruction_loss"].detach().cpu().numpy()
        )
        self.step_loss["train"]["gamma_l_loss"].append(
            outputs["gamma_l_loss"].detach().cpu().numpy()
        )
        self.step_loss["train"]["epsilon_l_loss"].append(
            outputs["epsilon_l_loss"].detach().cpu().numpy()
        )
        self.step_loss["train"]["baseline_spline_regularization_loss_curvature"].append(
            outputs["baseline_spline_regularization_loss_curvature"]
            .detach()
            .cpu()
            .numpy()
        )
        self.step_loss["train"][
            "lineshape_kernel_regularization_loss_curvature"
        ].append(
            outputs["lineshape_kernel_regularization_loss_curvature"]
            .detach()
            .cpu()
            .numpy()
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Autoencoder,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pl_module.log("epoch_loss_val", np.mean(outputs["loss"].detach().cpu().numpy()))
        self.step_loss["val"]["reconstruction_loss"].append(
            outputs["reconstruction_loss"].detach().cpu().numpy()
        )
        self.step_loss["val"]["gamma_l_loss"].append(
            outputs["gamma_l_loss"].detach().cpu().numpy()
        )
        self.step_loss["val"]["epsilon_l_loss"].append(
            outputs["epsilon_l_loss"].detach().cpu().numpy()
        )
        self.step_loss["val"]["baseline_spline_regularization_loss_curvature"].append(
            outputs["baseline_spline_regularization_loss_curvature"]
            .detach()
            .cpu()
            .numpy()
        )
        self.step_loss["val"]["lineshape_kernel_regularization_loss_curvature"].append(
            outputs["lineshape_kernel_regularization_loss_curvature"]
            .detach()
            .cpu()
            .numpy()
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Autoencoder) -> None:
        self.epoch_loss["train"]["reconstruction_loss"].append(
            np.mean(self.step_loss["train"]["reconstruction_loss"])
        )
        self.epoch_loss["train"]["gamma_l_loss"].append(
            np.mean(self.step_loss["train"]["gamma_l_loss"])
        )
        self.epoch_loss["train"]["epsilon_l_loss"].append(
            np.mean(self.step_loss["train"]["epsilon_l_loss"])
        )
        self.epoch_loss["train"][
            "baseline_spline_regularization_loss_curvature"
        ].append(
            np.mean(
                self.step_loss["train"]["baseline_spline_regularization_loss_curvature"]
            )
        )
        self.epoch_loss["train"][
            "lineshape_kernel_regularization_loss_curvature"
        ].append(
            np.mean(
                self.step_loss["train"][
                    "lineshape_kernel_regularization_loss_curvature"
                ]
            )
        )
        self.step_loss["train"] = {
            "reconstruction_loss": [],
            "gamma_l_loss": [],
            "epsilon_l_loss": [],
            "baseline_spline_regularization_loss_curvature": [],
            "lineshape_kernel_regularization_loss_curvature": [],
        }

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: Autoencoder) -> None:
        # reset the step dictionary and create a plot of the loss
        self.epoch_loss["val"]["reconstruction_loss"].append(
            np.mean(self.step_loss["val"]["reconstruction_loss"])
        )
        self.epoch_loss["val"]["gamma_l_loss"].append(
            np.mean(self.step_loss["val"]["gamma_l_loss"])
        )
        self.epoch_loss["val"]["epsilon_l_loss"].append(
            np.mean(self.step_loss["val"]["epsilon_l_loss"])
        )
        self.epoch_loss["val"]["baseline_spline_regularization_loss_curvature"].append(
            np.mean(
                self.step_loss["val"]["baseline_spline_regularization_loss_curvature"]
            )
        )
        self.epoch_loss["val"]["lineshape_kernel_regularization_loss_curvature"].append(
            np.mean(
                self.step_loss["val"]["lineshape_kernel_regularization_loss_curvature"]
            )
        )
        self.step_loss["val"] = {
            "reconstruction_loss": [],
            "gamma_l_loss": [],
            "epsilon_l_loss": [],
            "baseline_spline_regularization_loss_curvature": [],
            "lineshape_kernel_regularization_loss_curvature": [],
        }

        self.validation_epoch.append(trainer.current_epoch)

        fig, axs = plt.subplots(5, 1, figsize=(8, 10))
        axs[0].plot(
            self.epoch_loss["train"]["reconstruction_loss"],
            label="train",
            color="black",
            linestyle="solid",
        )
        axs[1].plot(
            self.epoch_loss["train"]["gamma_l_loss"],
            color="black",
            linestyle="dashed",
        )
        axs[2].plot(
            self.epoch_loss["train"]["epsilon_l_loss"],
            color="black",
            linestyle="dotted",
        )
        axs[3].plot(
            self.epoch_loss["train"]["baseline_spline_regularization_loss_curvature"],
            color="black",
            linestyle="dashdot",
        )
        axs[4].plot(
            self.epoch_loss["train"]["lineshape_kernel_regularization_loss_curvature"],
            color="black",
            linestyle="dashdot",
        )
        axs[0].plot(
            self.validation_epoch,
            self.epoch_loss["val"]["reconstruction_loss"],
            label="val",
            color="red",
            linestyle="solid",
        )

        axs[1].plot(
            self.validation_epoch,
            self.epoch_loss["val"]["gamma_l_loss"],
            color="red",
            linestyle="dashed",
        )

        axs[2].plot(
            self.validation_epoch,
            self.epoch_loss["val"]["epsilon_l_loss"],
            color="red",
            linestyle="dotted",
        )

        axs[3].plot(
            self.validation_epoch,
            self.epoch_loss["val"]["baseline_spline_regularization_loss_curvature"],
            color="red",
            linestyle="dashdot",
        )
        axs[4].plot(
            self.validation_epoch,
            self.epoch_loss["val"]["lineshape_kernel_regularization_loss_curvature"],
            color="red",
            linestyle="dashdot",
        )
        axs[4].set_xlabel("Epoch")
        ylabels = [
            "Reconstruction loss",
            r"$\gamma_l$ loss",
            r"$\epsilon_l$ loss",
            "Baseline spline reg.",
            "Lineshape kernel reg.",
        ]
        for i in range(5):
            axs[i].set_ylabel(ylabels[i])
            axs[i].set_yscale("log")
        # make all subplots share the same x axis
        for ax in axs:
            ax.label_outer()
        fig.add_gridspec()
        fig.legend()
        fig.tight_layout()
        plt.savefig(f"{self.save_dir}/loss_plot.png")
        plt.close(fig)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: Autoencoder,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.epoch_loss["test"]["reconstruction_loss"].append(
            np.mean(self.step_loss["test"]["reconstruction_loss"])
        )
        self.epoch_loss["test"]["gamma_l_loss"].append(
            np.mean(self.step_loss["test"]["gamma_l_loss"])
        )
        self.epoch_loss["test"]["epsilon_l_loss"].append(
            np.mean(self.step_loss["test"]["epsilon_l_loss"])
        )
        self.epoch_loss["test"]["baseline_spline_regularization_loss_curvature"].append(
            np.mean(
                self.step_loss["test"]["baseline_spline_regularization_loss_curvature"]
            )
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: Autoencoder) -> None:
        self.epoch_loss["test"]["reconstruction_loss"].append(
            np.mean(self.step_loss["test"]["reconstruction_loss"])
        )
        self.epoch_loss["test"]["gamma_l_loss"].append(
            np.mean(self.step_loss["test"]["gamma_l_loss"])
        )
        self.step_loss["test"] = {
            "reconstruction_loss": [],
            "gamma_l_loss": [],
        }
        print(
            "\nTEST EPOCH END REPORT: \n",
            f"reconstruction loss: {'{:.3e}'.format(self.epoch_loss['test']['reconstruction_loss'][-1])}\n",
            "gamma_l loss:",
            f"{'{:.3e}'.format(self.epoch_loss['test']['gamma_l_loss'][-1])}\n",
        )


class SubjectQuantifier(Callback):
    def __init__(
        self,
        subject_spectra: Tensor,
        subject_mask: Union[Tensor, np.ndarray],
        interval_bounds: List[int],
        basis: Basisset,
        save_dir: str = None,
        numerator: List[str] = None,
        denominator: str = None,
        magnitude_image: Union[Tensor, np.ndarray] = None,
        reference: List[np.ndarray] = None,
        index: List[Union[int, None]] = None,
        ppm_bounds: List[int] = None,
        subject_name: str = None,
        maximum_plot_ratio: List[float] = None,
        time_point: int = None,
        divide_by_mean: bool = False,
        rotation_transforms: List[RotationToRealAxis] = None,
    ) -> None:
        """
        Args:
            subject_spectra (Tensor):
                4D tensor with the first 3 dimensions corresponding to the spatial dimensions and the last
                dimension corresponding to the spectral dimension. One List entry (4D Tensor) per subject.
            subject_mask (Tensor): 3D tensor containing the masks for the subject brains.
            interval_bounds (List[int]): The indices of the bounds of the spectral interval fed into the network.
            basis (Basisset): The metabolite basis set used for the model. Used for creating the metabolite maps.
            save_dir (str): The directory to save the plots to.
            numerator (List[str]): The names of the metabolites to be used as the numerator in the quantification.
            denominator (str): The name of the metabolite to be used as the denominator in the quantification. Only the
                numerator is used if None.
            magnitude_image (Union[Tensor, np.ndarray]): Magnitude image of the subject to be quantified.
            reference (List[Union[Tensor, np.ndarray]]): Reference fits from LCModel.
            index (List[int]): The index along the axes of the subject to be quantified.
            ppm_bounds (List[int]): The indices of the bounds of the spectral interval fed into the network. Only needed
                if plots along the spectral dimension shall be created.
            subject_name (str): name of the subject to be quantified
            maximum_plot_ratio (List[float]): Maximum values of the ratios to be plotted.
            time_point (int): The time point of the subject data to be quantified. Default is None.
            divide_by_mean (bool): If True, the numerator map is divided by the mean of the denominator instead of
                dividing the numerator map by the denominator map.
            rotation_transforms (List[RotationToRealAxis]): List of rotation transforms to be applied to the subject
                signals and fitted spectra. This is only relevant if the initial preprocessing steps shall be reverted.
                In this case, the rotation transforms are applied to the fitted spectra before saving them.
        """
        self.save_dir = save_dir
        if self.save_dir is not None:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.basis = basis
        self.subject_mask = subject_mask
        self.subject_spectra = subject_spectra
        self.interval_bounds = interval_bounds
        self.magnitude_image = magnitude_image
        self.reference = reference

        self.numerator = numerator
        self.denominator = denominator
        self.index = [
            slice(idx, idx + 1) if idx is not None else slice(None) for idx in index
        ]
        self.ppm_bounds = ppm_bounds
        self.subject_name = subject_name if subject_name is not None else "subject"
        self.maximum_plot_ratio = maximum_plot_ratio
        self.time_point = time_point
        self.divide_by_mean = divide_by_mean

        self.rotation_transforms = rotation_transforms

    @classmethod
    def from_matlab(
        cls,
        subject_path: str,
        basis: Basisset,
        interval_bounds: List[int],
        signal_length: int,
        save_dir: str = None,
        numerator: List[str] = None,
        denominator: str = None,
        index: List[Union[int, None]] = None,
        ppm_bounds: List[float] = None,
        conjugate_subject_signals: bool = False,
        rotate_subject_signals_to_real_axis: bool = False,
        subject_name: str = None,
        maximum_plot_ratio: List[float] = None,
        magnitude_image_path: str = None,
        time_point: int = None,
        divide_by_mean: bool = False,
        **kwargs,
    ) -> Self:
        """
        Loads the subject spectra and masks from a matlab file.

        Args:
            subject_path (str): Path to the matlab file containing the subject data.
            basis (Basisset): The metabolite basis set used for the model. Used for creating the metabolite maps.
            interval_bounds (List[int]): The indices of the bounds of the spectral interval fed into the network.
            signal_length (int): The length of the spectral dimension. If the loaded signals are shorter, they are
            zero-filled to this length in the time domain. They are truncated, if they are longer.
            save_dir (str): The directory to save the plots to.
            numerator (List[str]): The names of the metabolites to be used as the numerators in the quantification.
            denominator (str): The name of the metabolite to be used as the denominator in the quantification. Only the
                numerator is used if None.
            index (List[int]): The index along the axes of the subject to be quantified.
            ppm_bounds (List[float]): The indices of the bounds of the spectral interval fed into the network. Only
                needed if plots along the spectral dimension shall be created.
            conjugate_subject_signals (bool): If True, the subject time-domain signals are conjugated.
            rotate_subject_signals_to_real_axis (bool): If True, the subject time-domain signals are rotated to the
                real axis.
            subject_name (str): name of the subject to be quantified
            maximum_plot_ratio (List[float]): Maximum values of the ratios to be plotted.
            magnitude_image_path (str): Path to the magnitude image of the subject.
            time_point (int): The time point of the subject data to be quantified. Default is None.
            divide_by_mean (bool): If True, the numerator map is divided by the mean of the denominator instead of
                dividing the numerator map by the denominator map.
            **kwargs: Additional keyword arguments.

        Returns:
            The SubjectQuantifier object.
        """
        device = (
            kwargs["device"]
            if "device" in kwargs.keys()
            else torch.get_default_device()
        )

        # load and preprocess the subject data
        data = mat73.loadmat(subject_path)
        subject_spectra = []
        for time_point in range(data["csi"]["Data"].shape[-1]):
            fids = data["csi"]["Data"][:, :, :, :, time_point]

            if conjugate_subject_signals:
                fids = np.conj(fids)
            if fids.shape[1] < signal_length:
                fids_ = np.zeros(
                    shape=(*fids.shape[:-1], signal_length),
                    dtype=get_default_type(domain="complex", framework="numpy"),
                )
                fids_[:, :, :, : fids.shape[-1]] = fids
                fids = fids_
            else:
                fids = fids[:, :signal_length]

            subject_spectra.append(
                np.array(
                    fftshift(fft(fids, axis=-1), axes=-1),
                    dtype=get_default_type(domain="complex", framework="numpy"),
                )
            )
        subject_mask = np.array(data["mask"], dtype=bool)

        rotation_transforms = None
        if rotate_subject_signals_to_real_axis:
            rotation_transforms = []
            for i in range(len(subject_spectra)):
                rotation_transforms.append(RotationToRealAxis(mode="absolute"))
                subject_spectra[i] = rotation_transforms[-1](
                    subject_spectra[i], update_trafo=True
                )

        subject_spectra = torch.from_numpy(np.array(subject_spectra)).to(device)

        # load lcmodel reference which is expected to be in the same folder as the subject data
        reference = []
        affine = np.eye(4)
        for name in basis.metabolite_names:
            try:
                for time_point in range(subject_spectra.shape[0]):
                    data = nib.load(
                        os.path.split(subject_path)[0]
                        + f"/maps/{time_point+1}/Orig/{name}_amp_map.nii"
                    )
                    affine = data.affine
                    shape = data.shape
                    data = data.get_fdata()
                    data = np.flip(data, axis=0)
                    data = np.flip(data, axis=1)
                    reference.append(data)
            except:
                print(f"Error loading reference for {name}. Skipping.")
                reference = None
                shape = subject_mask.shape

        # load magnitude image
        magnitude_image = None
        if magnitude_image_path is not None:
            try:
                data = nib.load(
                    os.path.split(subject_path)[0]
                    + f"/{subject_name}/maps/magnitude.nii"
                )
                data = nilearn.image.resample_img(
                    data, target_affine=affine, target_shape=shape
                ).get_fdata()
                magnitude_image = np.flip(np.flip(data.swapaxes(0, 1), 0), 2)
            except:
                print(
                    "Could not load magnitude image from",
                    magnitude_image_path,
                    ". Skipping.",
                )
                magnitude_image = np.zeros_like(subject_mask)

        return cls(
            subject_spectra=subject_spectra,
            subject_mask=subject_mask,
            interval_bounds=interval_bounds,
            basis=basis,
            save_dir=save_dir,
            numerator=numerator,
            denominator=denominator,
            index=index,
            ppm_bounds=ppm_bounds,
            subject_name=subject_name,
            maximum_plot_ratio=maximum_plot_ratio,
            magnitude_image=magnitude_image,
            reference=reference,
            time_point=time_point,
            divide_by_mean=divide_by_mean,
            rotation_transforms=rotation_transforms,
        )

    def _fit(self, pl_module: Autoencoder, x: Tensor) -> Tensor:
        """Fits the spectra in x using the pl_module and returns the fit parameters as Tensor."""

        with torch.no_grad():
            input_spectra = x.detach()

            input_spectra = input_spectra[
                ..., self.interval_bounds[0] : self.interval_bounds[1]
            ]
            input_spectra = pl_module.normalization(input_spectra)

            z = pl_module.encoder(input_spectra)
            z = pl_module.latent_space_trafo(z)

        return z

    def quantify_ratios_at_index(
        self,
        pl_module: Autoencoder,
        numerator: str,
        return_denominator: bool = False,
        time_point: int = None,
    ) -> np.ndarray:
        """
        Quantifies the spectrum at the given index by taking the ratio of two metabolites

        Args:
            pl_module (Autoencoder): The autoencoder used for the quantification.
            numerator (str): The name of the metabolite to be used as the numerator in the quantification.
            return_denominator (bool): If True, the denominator is also returned.
            time_point (int): The time point of the subject data to be quantified. Default is None.

        Returns:
            A numpy array containing the metabolite ratios as fitted by the pl_module.
        """

        input_spectra = self.subject_spectra[:, *self.index].detach()
        mask = self.subject_mask[*self.index]

        input_spectra = input_spectra[:, mask, :].permute(1, 0, 2)
        z = self._fit(pl_module=pl_module, x=input_spectra).detach()
        inputs = pl_module.decoder._unpack(z)
        amplitudes, delta_f, _ = pl_module.decoder._calculate_dynamics(inputs=inputs)
        amplitudes = pl_module.normalization.invert(
            amplitudes,
            scaling_only=True,
            squeeze="adaptive",
        )

        denominator = np.zeros_like(mask, dtype=np.float32)
        if self.basis.get_index_from_name(self.denominator) is not None:
            for idx in self.basis.get_index_from_name(self.denominator):
                denominator[mask] += (
                    amplitudes[:, time_point, idx].squeeze().detach().cpu().numpy()
                )
        else:
            denominator = np.ones_like(mask, dtype=np.float32)
        output = np.zeros_like(mask, dtype=np.float32)
        for idx in self.basis.get_index_from_name(numerator):
            output[mask] += (
                amplitudes[:, time_point, idx].squeeze().detach().cpu().numpy()
            )

        if self.divide_by_mean:
            scaling = (
                amplitudes[:, :, self.basis.get_index_from_name(self.denominator)]
                .mean()
                .detach()
                .cpu()
                .numpy()
            )
            output /= scaling
            denominator /= scaling
        else:
            output[mask] /= denominator[mask]

        if return_denominator:
            return output, denominator
        return output

    def fit_spectrum_at_index(self, pl_module: Autoencoder) -> Tuple[Tensor, Tensor]:
        """Fits the spectra using the pl_module at the indices self.index"""

        input_spectra = self.subject_spectra.detach()[:, *self.index, :].detach()
        input_spectra = input_spectra.squeeze().unsqueeze(0)

        z = self._fit(pl_module=pl_module, x=input_spectra).detach()

        basis_signals, baseline_signal, optionals = pl_module.decoder(z)

        return basis_signals, baseline_signal

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: Autoencoder) -> None:
        """Creates a plot of the metabolite ratio name/self.denominator at self.index for name in self.numerator.

        If self.index has three non-None entries, a spectral fit at the unique position defined by self.index is done
        and the result is plotted.

        If self.index has one non-None entry, a quantification along the two None axes is done and the result is
        plotted as a 2D image.

        If self.index has no non-None entries, a quantification along all three axes is done and the result is saved
        as a nifti image.
        """

        # check if there is an implementation
        if self.index is None:
            raise NotImplementedError(
                "An implementation is only available for index lists with exactly zero, one or three non-None entry."
            )
        if len([idx for idx in self.index if idx is not None]) not in [0, 1, 3]:
            raise NotImplementedError(
                "An implementation is only available for index lists with exactly zero, one or three non-None entry."
            )

        # create 2D image if there is 1 non-None entry
        elif len([idx.start for idx in self.index if idx.start is not None]) == 1:
            self.on_validation_epoch_end_2d(trainer=trainer, pl_module=pl_module)

        # create plot of single fit if there are no None entries
        elif len([idx.start for idx in self.index if idx.start is not None]) == 3:
            self.on_validation_epoch_end_1d(trainer=trainer, pl_module=pl_module)

        # create nifti image if there are 3 None entries
        elif len([idx.start for idx in self.index if idx.start is not None]) == 0:
            self.on_validation_epoch_end_3d(trainer=trainer, pl_module=pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: Autoencoder) -> None:
        self.on_validation_epoch_end(trainer, pl_module)

    def get_lcmodel_ratios_at_index(
        self, pl_module: Autoencoder, signal_name: str, time_point: int
    ) -> np.ndarray:
        """Returns the ratio of the LCModel fits for signal_name and self.denominator"""

        numerator = 0.0
        for i in self.basis.get_index_from_name(signal_name):
            numerator += self.reference[time_point + self.subject_spectra.shape[0] * i]

        denominator = 1.0
        if self.denominator is not None:
            denominator = 0.0
            for i in self.basis.get_index_from_name(self.denominator):
                denominator += self.reference[
                    time_point + self.subject_spectra.shape[0] * i
                ]

        output = np.zeros(numerator.shape, dtype=np.float32)
        if self.divide_by_mean:
            scaling = []
            mask = self.subject_mask[*self.index]
            for time_point in range(self.subject_spectra.shape[0]):
                for i in self.basis.get_index_from_name(self.denominator):
                    scaling.append(
                        (
                            self.reference[
                                time_point + self.subject_spectra.shape[0] * i
                            ][*self.index]
                        )[mask]
                    )
            scaling = np.mean(scaling)

            output = numerator / scaling
            return output[*self.index]

        output[self.subject_mask] = (numerator / denominator)[self.subject_mask]

        return output[*self.index]

    def on_validation_epoch_end_2d(
        self, trainer: Trainer, pl_module: Autoencoder
    ) -> None:
        """Helper function to make on_validation_epoch_end more readable."""
        for i, name in enumerate(self.numerator):
            # create plot
            fig, axs = plt.subplots(
                self.subject_spectra.shape[0],
                3,
                figsize=(9, 3 * self.subject_spectra.shape[0]),
            )

            for time_point in range(self.subject_spectra.shape[0]):
                # do quantification with the autoencoder
                ratios_model = self.quantify_ratios_at_index(
                    pl_module=pl_module, numerator=name, time_point=time_point
                )
                ratios_model = ratios_model[:, :, 0]
                # ratios_model[ratios_model > self.maximum_plot_ratio[i]] = np.nan

                # load reference quantification
                ratios_lcm = self.get_lcmodel_ratios_at_index(
                    pl_module=pl_module, signal_name=name, time_point=time_point
                )[:, :, 0]

                # plot the model quantification
                axs[time_point, 0].imshow(
                    self.magnitude_image[:, :, self.index[2].start].T,
                    cmap="grey",
                    alpha=0.4,
                )
                plot = axs[time_point, 0].imshow(ratios_model.T, cmap="hot", alpha=1)
                axs[time_point, 0].set_xticks([])
                axs[time_point, 0].set_yticks([])
                axs[time_point, 0].invert_xaxis()
                divider = make_axes_locatable(axs[time_point, 0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(plot, cax=cax)
                cbar.set_label(f"{name}/{self.denominator}")
                plot.set_clim(0.0, self.maximum_plot_ratio[i])
                axs[time_point, 0].set_title("Model")

                # plot the reference quantification
                if self.magnitude_image is not None:
                    axs[time_point, 1].imshow(
                        self.magnitude_image[:, :, self.index[2].start].T,
                        cmap="grey",
                        alpha=0.4,
                    )
                plot = axs[time_point, 1].imshow(ratios_lcm.T, cmap="hot", alpha=1)
                axs[time_point, 1].set_xticks([])
                axs[time_point, 1].set_yticks([])
                axs[time_point, 1].invert_xaxis()
                divider = make_axes_locatable(axs[time_point, 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(plot, cax=cax)
                cbar.set_label(f"{name}/{self.denominator}")
                plot.set_clim(0.0, self.maximum_plot_ratio[i])
                axs[time_point, 1].set_title("LCModel")

                # plot the difference
                if self.magnitude_image is not None:
                    axs[time_point, 2].imshow(
                        self.magnitude_image[:, :, self.index[2].start].T,
                        cmap="grey",
                        alpha=0.4,
                    )
                plot = axs[time_point, 2].imshow(
                    (ratios_model - ratios_lcm).T, cmap="RdBu", alpha=1.0
                )
                axs[time_point, 2].set_xticks([])
                axs[time_point, 2].set_yticks([])
                axs[time_point, 2].invert_xaxis()
                divider = make_axes_locatable(axs[time_point, 2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(plot, cax=cax)
                cbar.set_label(f"{name}/{self.denominator}")
                if self.maximum_plot_ratio[i] is not None:
                    plot.set_clim(
                        -self.maximum_plot_ratio[i] / 8.0,
                        self.maximum_plot_ratio[i] / 8.0,
                    )
                else:
                    plot.set_clim(
                        -2 * np.nanmean(np.abs(ratios_model - ratios_lcm)),
                        2 * np.nanmean(np.abs(ratios_model - ratios_lcm)),
                    )
                axs[time_point, 2].set_title("Absolute difference")

            axis = [
                (i, idx.start)
                for i, idx in enumerate(self.index)
                if idx.start is not None
            ][0]
            axis_names = ["x", "y", "z"]
            fig.suptitle(
                self.subject_name
                + " at slice "
                + str(axis[1])
                + " along axis "
                + axis_names[axis[0]]
            )
            fig.tight_layout()
            plt.savefig(
                f"{self.save_dir}/{self.subject_name}_{name}_{self.denominator}_subject_quantification.png"
            )
            plt.close(fig)

        # plot denominator
        fig, axs = plt.subplots(
            nrows=self.subject_spectra.shape[0],
            ncols=3,
            figsize=(9, 3 * self.subject_spectra.shape[0]),
        )
        for time_point in range(self.subject_spectra.shape[0]):
            _, denominator = self.quantify_ratios_at_index(
                pl_module=pl_module,
                numerator=name,
                return_denominator=True,
                time_point=time_point,
            )
            denominator = denominator[:, :, 0]

            axs[time_point, 0].imshow(
                self.magnitude_image[:, :, self.index[2].start].T,
                cmap="grey",
                alpha=0.4,
            )
            plot = axs[time_point, 0].imshow(denominator.T, cmap="hot", alpha=1)
            axs[time_point, 0].set_xticks([])
            axs[time_point, 0].set_yticks([])
            axs[time_point, 0].invert_xaxis()
            divider = make_axes_locatable(axs[time_point, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(plot, cax=cax)
            cbar.set_label(f"{self.denominator}")
            axs[time_point, 0].set_title("Model")

            # Reference fit
            if self.magnitude_image is not None:
                axs[time_point, 1].imshow(
                    self.magnitude_image[:, :, self.index[2].start].T,
                    cmap="grey",
                    alpha=0.4,
                )
            denominator_reference = 0.0
            if self.denominator is not None:
                for i in self.basis.get_index_from_name(self.denominator):
                    denominator_reference += self.reference[
                        time_point + self.subject_spectra.shape[0] * i
                    ][*self.index][:, :, 0]
                plot = axs[time_point, 1].imshow(
                    denominator_reference.T, cmap="hot", alpha=1
                )
                axs[time_point, 1].set_xticks([])
                axs[time_point, 1].set_yticks([])
                axs[time_point, 1].invert_xaxis()
                divider = make_axes_locatable(axs[time_point, 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(plot, cax=cax)
                cbar.set_label(f"{self.denominator}")
                axs[time_point, 1].set_title("LCModel")

            # difference
            if self.magnitude_image is not None:
                axs[time_point, 2].imshow(
                    self.magnitude_image[:, :, self.index[2].start].T,
                    cmap="grey",
                    alpha=0.4,
                )
            plot = axs[time_point, 2].imshow(
                (denominator - denominator_reference).T, cmap="RdBu", alpha=1.0
            )
            axs[time_point, 2].set_xticks([])
            axs[time_point, 2].set_yticks([])
            axs[time_point, 2].invert_xaxis()
            divider = make_axes_locatable(axs[time_point, 2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(plot, cax=cax)
            cbar.set_label(f"{self.denominator}")

        axis = [
            (i, idx.start) for i, idx in enumerate(self.index) if idx.start is not None
        ][0]
        axis_names = ["x", "y", "z"]
        fig.suptitle(
            self.subject_name
            + " at slice "
            + str(axis[1])
            + " along axis "
            + axis_names[axis[0]]
        )
        fig.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{self.subject_name}_{self.denominator}_subject_quantification.png"
        )
        plt.close(fig)

    def on_validation_epoch_end_1d(
        self, trainer: Trainer, pl_module: Autoencoder
    ) -> None:
        """Helper function to make on_validation_epoch_end more readable."""
        # do the spectral fit
        (
            basis_signals,
            baseline_signals,
        ) = self.fit_spectrum_at_index(pl_module=pl_module)
        basis_signals = basis_signals.detach().cpu().squeeze().numpy()
        baseline_signals = baseline_signals.detach().cpu().squeeze().numpy()

        # get input spectra as comparison and get latent space representation
        input_spectra = self.subject_spectra[:, *self.index].detach()
        input_spectra = input_spectra.squeeze().unsqueeze(0)
        fit = self._fit(pl_module=pl_module, x=input_spectra)
        z = pl_module.decoder._unpack(fit)
        amplitudes, delta_f, _ = pl_module.decoder._calculate_dynamics(inputs=z)

        input_spectra = input_spectra.detach()[
            ..., self.interval_bounds[0] : self.interval_bounds[1]
        ].cpu()
        input_spectra = pl_module.normalization(input_spectra)
        input_spectra = input_spectra.squeeze().numpy()

        ppm_axis = np.linspace(
            self.ppm_bounds[0], self.ppm_bounds[1], basis_signals.shape[-1]
        )

        fig, axs = plt.subplots(4, 1, figsize=(10, 16))
        axs[0].set_title("amplitudes")
        for i, name in enumerate(z["amplitudes"].keys()):
            axs[0].plot(
                amplitudes[0, :, i].squeeze().detach().cpu().numpy(), label=name
            )
        axs[0].legend()

        axs[1].set_title("delta_f")
        for i, name in enumerate(z["amplitudes"].keys()):
            axs[1].plot(
                (delta_f[0, :, 0] + z["basis_delta_f"][0, :, i])
                .squeeze()
                .detach()
                .cpu()
                .numpy(),
                label=name,
            )
        axs[1].plot(delta_f[0, :, 0].squeeze().detach().cpu().numpy(), label="global")
        axs[1].legend()

        axs[2].set_title("delta phi 0")
        axs[2].plot(z["delta_phi"].squeeze().detach().cpu().numpy())

        axs[3].set_title("delta phi 1")
        axs[3].plot(z["delta_phi_1"].squeeze().detach().cpu().numpy())

        fig.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{self.subject_name}_{self.index[0].start}_"
            + f"{self.index[1].start}_{self.index[2].start}_parameters.png"
        )
        plt.close(fig)

        for time_point in range(self.subject_spectra.shape[0]):
            fig, axs = plt.subplots(
                4,
                1,
                figsize=(
                    10,
                    10 + 0.65 * len(self.basis),
                ),
                height_ratios=[6, 1.5, 0.75 * len(self.basis), 1.5],
            )
            axs[0].plot(
                ppm_axis,
                basis_signals[time_point, :].real,
                color="green",
                label="basis",
            )
            axs[0].plot(
                ppm_axis,
                basis_signals[time_point, :].imag,
                color="green",
                linestyle="dotted",
            )
            axs[0].plot(
                ppm_axis,
                baseline_signals[time_point, :],
                color="grey",
                label="baseline",
            )
            axs[0].plot(
                ppm_axis,
                baseline_signals[time_point, :].imag,
                color="grey",
                linestyle="dotted",
            )
            axs[0].plot(
                ppm_axis,
                (basis_signals + baseline_signals)[time_point, :].real,
                color="blue",
                label="reconstruction",
            )
            axs[0].plot(
                ppm_axis,
                (basis_signals + baseline_signals)[time_point, :].imag,
                color="blue",
                linestyle="dotted",
            )
            axs[0].plot(
                ppm_axis,
                input_spectra[time_point, :].real,
                color="red",
                label="input",
            )
            axs[0].plot(
                ppm_axis,
                input_spectra[time_point, :].imag,
                color="red",
                linestyle="dotted",
            )
            axs[0].set_xlabel("chemical shift [ppm]")
            axs[0].set_ylabel("Signal [a.u.]")
            axs[0].legend(loc="upper left")

            # residuals
            axs[1].plot(
                ppm_axis,
                (basis_signals + baseline_signals - input_spectra)[time_point, :].real,
                color="black",
                label="residuals",
            )
            axs[1].plot(
                ppm_axis,
                (basis_signals + baseline_signals - input_spectra)[time_point, :].imag,
                color="black",
                linestyle="dotted",
            )
            axs[1].set_xlabel("chemical shift [ppm]")
            axs[1].set_ylabel("residuals [a.u.]")
            axs[1].grid()

            # basis components
            normalization = np.max(np.abs(basis_signals)) * 1 / 3
            z_ = fit.clone().detach()
            for i in range(len(self.basis)):
                signal = np.abs(
                    pl_module.decoder(
                        z_, active_metabolites=[self.basis.metabolite_names[i]]
                    )[0]
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
                axs[2].plot(
                    ppm_axis,
                    signal[time_point, :] / normalization - 3 * i,
                    color="black",  # accessed at time_point 0 since amplitudes were stored at first index
                )
                axs[2].text(
                    self.ppm_bounds[1] + 0.35,
                    -3 * i,
                    self.basis.metabolite_names[i],
                    fontsize=11,
                    verticalalignment="center",
                )
                axs[2].text(
                    self.ppm_bounds[0] - 0.1,
                    -3 * i + 0.18,
                    "A="
                    + "{:.2e}".format(amplitudes[0, time_point, i].item())
                    + "\n"
                    + r"$\Delta f=$"
                    + "{:.2e}".format(
                        delta_f[0, time_point, 0].item()
                        + z["basis_delta_f"][0, time_point, i]
                    )
                    + "\n"
                    + r"$\Delta \tau=$"
                    + "{:.2e}".format(z["basis_lorentzian_damping"][0, time_point, i]),
                    fontsize=11,
                    verticalalignment="center",
                )
            axs[2].set_xlabel("chemical shift [ppm]")
            axs[2].set_yticks([])
            axs[2].set_title(
                r"$\Delta f=$"
                + "{:.2e}".format(delta_f[0, time_point, 0].item())
                + r" Hz, $\Delta \phi_0=$"
                + "{:.2e}".format(z["delta_phi"][0, time_point, 0].item())
                + r" rad, $\Delta \phi_1=$"
                + "{:.2e}".format(z["delta_phi_1"][0, time_point, 0].item())
                + r" rad/Hz"
            )

            for side in ["right", "top", "left", "bottom"]:
                axs[2].spines[side].set_visible(False)

            axs[3].set_title("lineshape kernel")
            axs[3].plot(
                z["lineshape_kernel"].detach().cpu().numpy()[0, time_point, 0, :]
            )
            axs[3].set_xlabel("n [.]")
            axs[3].set_ylabel("signal [a.u.]")

            for spine in fig.gca().spines.values():
                spine.set_visible(False)
            axs[0].invert_xaxis()
            axs[1].invert_xaxis()
            axs[2].invert_xaxis()
            fig.suptitle(
                self.subject_name
                + " at slice "
                + str(self.index[2].start)
                + r", $(i_x,i_y)$ = ("
                + str(self.index[0].start)
                + ","
                + str(self.index[1].start)
                + ")"
            )

            mask_ax = fig.add_axes([0.78, 0.86, 0.095, 0.095])
            mask_ax.scatter([self.index[1].start], [self.index[0].start], color="red")
            mask_ax.imshow(self.subject_mask[:, :, self.index[2].start].T, cmap="gray")
            mask_ax.set_xticks([])
            mask_ax.set_yticks([])
            mask_ax.invert_xaxis()

            fig.tight_layout()
            plt.savefig(
                f"{self.save_dir}/{self.subject_name}_{self.index[0].start}_"
                + f"{self.index[1].start}_{self.index[2].start}_{time_point}_reconstruction.png"
            )
            plt.close(fig)

    def on_validation_epoch_end_3d(
        self, trainer: Trainer, pl_module: Autoencoder
    ) -> None:
        """Helper function to make on_validation_epoch_end more readable.

        Saves the following quantities:
            - input spectra
            - fitted spectra at each time point
            - amplitudes at each time point
            - all model parameters
        """

        # do quantification
        input_spectra = self.subject_spectra.detach()
        input_spectra = torch.swapaxes(input_spectra[:, self.subject_mask, :], 0, 1)

        # calculate and save baseline, basis signals as well as fitted and input spectra
        shape = self.subject_spectra.shape
        fitted_spectra = torch.zeros(
            size=shape, dtype=get_default_type(domain="complex", framework="torch")
        )
        baseline_signals = torch.zeros(
            size=shape, dtype=get_default_type(domain="complex", framework="torch")
        )
        basis_signals = torch.zeros(
            size=shape, dtype=get_default_type(domain="complex", framework="torch")
        )
        with torch.no_grad():
            x = input_spectra[..., self.interval_bounds[0] : self.interval_bounds[1]]
            x = pl_module.normalization(x)
            (
                basis_signals_fit,
                baseline_signals_fit,
                basis_lorentzian_dampings_fit,
                basis_frequency_shifts_fit,
                lineshape_kernel_fit,
                optionals,
            ) = pl_module.forward(x, crop_signal=False)

            # invert preprocessing
            basis_signals_fit = pl_module.normalization.invert(
                basis_signals_fit, squeeze="adaptive", scaling_only=True
            )
            baseline_signals_fit = pl_module.normalization.invert(
                baseline_signals_fit, squeeze="adaptive", scaling_only=False
            )

            fitted_spectra[:, self.subject_mask, :] = torch.swapaxes(
                basis_signals_fit + baseline_signals_fit, 0, 1
            )
            baseline_signals[:, self.subject_mask, :] = torch.swapaxes(
                baseline_signals_fit, 0, 1
            )
            basis_signals[:, self.subject_mask, :] = torch.swapaxes(
                basis_signals_fit, 0, 1
            )
            subject_spectra = self.subject_spectra.clone()

            subject_spectra = subject_spectra.detach().cpu().numpy()
            fitted_spectra = fitted_spectra.detach().cpu().numpy()
            baseline_signals = baseline_signals.detach().cpu().numpy()
            basis_signals = basis_signals.detach().cpu().numpy()
            for i in range(len(self.subject_spectra)):
                fitted_spectra[i] = self.rotation_transforms[i].invert(
                    fitted_spectra[i], squeeze="adaptive"
                )
                baseline_signals[i] = self.rotation_transforms[i].invert(
                    baseline_signals[i], squeeze="adaptive"
                )
                basis_signals[i] = self.rotation_transforms[i].invert(
                    basis_signals[i], squeeze="adaptive"
                )
                subject_spectra[i] = self.rotation_transforms[i].invert(
                    subject_spectra[i], squeeze="adaptive"
                )

        # save as nifti
        nib.save(
            nib.Nifti1Image(subject_spectra, np.eye(4)),
            f"{self.save_dir}/{self.subject_name}_input_spectra.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(fitted_spectra, np.eye(4)),
            f"{self.save_dir}/{self.subject_name}_fitted_spectra.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(basis_signals, np.eye(4)),
            f"{self.save_dir}/{self.subject_name}_basis_signals.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(baseline_signals, np.eye(4)),
            f"{self.save_dir}/{self.subject_name}_baseline_signals.nii.gz",
        )

        # calculate and save latent space representation
        z = self._fit(pl_module=pl_module, x=input_spectra).detach()
        shape = (
            self.subject_mask.shape[0],
            self.subject_mask.shape[1],
            self.subject_mask.shape[2],
            z.shape[-1],
        )
        latent_space_vectors = torch.zeros(size=shape)
        latent_space_vectors[self.subject_mask, :] = z
        nib.save(
            nib.Nifti1Image(latent_space_vectors.detach().cpu().numpy(), np.eye(4)),
            f"{self.save_dir}/{self.subject_name}_latent_space_vectors.nii.gz",
        )

        # save unpacked latent space representation
        inputs = pl_module.decoder._unpack(z)
        amplitudes, delta_f, _ = pl_module.decoder._calculate_dynamics(inputs=inputs)
        amplitudes = pl_module.normalization.invert(
            amplitudes,
            scaling_only=True,
            squeeze="adaptive",
        )

        Path(f"{self.save_dir}/basisset_parameters/").mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(inputs["amplitudes"].keys()):
            save_quantity_to_nifti(
                x=inputs["amplitudes"][name],
                mask=self.subject_mask,
                save_path=f"{self.save_dir}/basisset_parameters/{self.subject_name}_{name}_amplitude_parameters.nii.gz",
            )
            save_quantity_to_nifti(
                x=inputs["basis_lorentzian_damping"][..., i : i + 1],
                mask=self.subject_mask,
                save_path=f"{self.save_dir}/basisset_parameters/{self.subject_name}_{name}_lorentzian_damping.nii.gz",
            )
            save_quantity_to_nifti(
                x=inputs["basis_delta_f"][..., i : i + 1],
                mask=self.subject_mask,
                save_path=f"{self.save_dir}/basisset_parameters/{self.subject_name}_{name}_frequency_shift.nii.gz",
            )
            save_quantity_to_nifti(
                x=amplitudes[..., i : i + 1],
                mask=self.subject_mask,
                save_path=f"{self.save_dir}/basisset_parameters/{self.subject_name}_{name}_amplitudes.nii.gz",
            )

        save_quantity_to_nifti(
            x=inputs["delta_phi"],
            mask=self.subject_mask,
            save_path=f"{self.save_dir}/{self.subject_name}_delta_phi.nii.gz",
        )
        save_quantity_to_nifti(
            x=inputs["delta_phi_1"],
            mask=self.subject_mask,
            save_path=f"{self.save_dir}/{self.subject_name}_delta_phi_1.nii.gz",
        )
        save_quantity_to_nifti(
            x=inputs["delta_f"],
            mask=self.subject_mask,
            save_path=f"{self.save_dir}/{self.subject_name}_delta_f_parameters.nii.gz",
        )
        save_quantity_to_nifti(
            x=delta_f,
            mask=self.subject_mask,
            save_path=f"{self.save_dir}/{self.subject_name}_delta_f.nii.gz",
        )
        save_quantity_to_nifti(
            x=inputs["lineshape_kernel"].squeeze(),
            mask=self.subject_mask,
            save_path=f"{self.save_dir}/{self.subject_name}_lineshape_kernel.nii.gz",
        )
        save_quantity_to_nifti(
            x=inputs["baseline_spline_values_real"],
            mask=self.subject_mask,
            save_path=f"{self.save_dir}/{self.subject_name}_baseline_spline_values_real.nii.gz",
        )
        save_quantity_to_nifti(
            x=inputs["baseline_spline_values_imag"],
            mask=self.subject_mask,
            save_path=f"{self.save_dir}/{self.subject_name}_baseline_spline_values_imag.nii.gz",
        )


def save_quantity_to_nifti(x: Tensor, mask: Tensor, save_path: str) -> None:
    shape = (mask.shape[0], mask.shape[1], mask.shape[2], x.shape[-2], x.shape[-1])
    template = torch.zeros(shape)
    template[mask, :] = x
    nib.save(nib.Nifti1Image(template.detach().cpu().numpy(), np.eye(4)), save_path)
