import json
from typing import Any, Dict, List, Union

import numpy as np
import torch
from typing_extensions import Self


class IOConfig:
    def __init__(
        self,
        basis_path: str = None,
        training_data_path: str = None,
        test_data_path: str = None,
        logging_path: str = None,
        full_subject_path_validation: str = None,
        full_subject_path_test: str = None,
        saving_path: str = None,
        dwelltime: float = None,
        reference_frequency: float = None,
        nucleus: str = None,
    ) -> None:
        """
        Args:
            basis_path (str): Path to the basis signals (including only metabolite signals).
            training_data_path (str): Path to the training data.
            test_data_path (str): Path to the test data.
            logging_path (str): Path to save the logs and plots.
            full_subject_path_validation (str): Path to the full subject data used for validating the model.
            full_subject_path_test (str): Path to subject data used for testing the model.
            saving_path (str): Path to save the model and checkpoints.
            dwelltime (float): Dwell time of the data.
            reference_frequency (float): Reference frequency of the data.
            nucleus (str): Nucleus the resonances are originating from.
        """

        self.basis_path = basis_path
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.logging_path = logging_path
        self.full_subject_path_validation = full_subject_path_validation
        self.full_subject_path_test = full_subject_path_test
        self.saving_path = saving_path
        self.dwelltime = dwelltime
        self.reference_frequency = reference_frequency
        self.nucleus = nucleus


class PreprocessorConfig:
    def __init__(
        self,
        basis_signal_names: List[str] = None,
        normalize_signals: Union[str, float] = None,
        shift_mean_to_zero: bool = None,
        augmentation_params: List[float] = None,
        conjugate_basis: bool = None,
        rotate_signals_to_real_axis: bool = None,
        ppm_bounds: List[float] = None,
        normalization_scaling_mode: str = None,
    ) -> None:
        """
        Args:
            basis_signal_names (List[str]): Names of the basis signals (metabolites).
            normalize_signals (Union[str, float]): Normalize the signals from the datasets. Valid values are
                "frequency", "time", None and a float. If None, no normalization is done.
            shift_mean_to_zero (bool): Shift the mean of each signal to zero.
            augmentation_params (List[float]): Augmentation parameters.
            conjugate_basis (bool): Conjugate the basis signals.
            rotate_signals_to_real_axis (bool): Rotate the second point of each signal to the real axis.
            ppm_bounds (List[float]): Bounds along the ppm axis defining the spectrum that is considered by the encoder.
            normalization_scaling_mode (str): Mode for scaling the normalization. Can be "max" or "mean".
        """

        self.basis_signal_names = basis_signal_names
        self.normalize_signals = normalize_signals
        self.shift_mean_to_zero = shift_mean_to_zero
        self.augmentation_params = augmentation_params
        self.conjugate_basis = conjugate_basis
        self.rotate_signals_to_real_axis = rotate_signals_to_real_axis
        self.ppm_bounds = ppm_bounds
        self.normalization_scaling_mode = normalization_scaling_mode


class EncoderConfig:
    def __init__(self, shared_lineshape_kernel: bool = None) -> None:
        """
        Args:
            shared_lineshape_kernel (bool): Whether the lineshape kernel should be shared among time points or not.
        """

        self.shared_lineshape_kernel = shared_lineshape_kernel


class DecoderConfig:
    def __init__(
        self,
        n_splines: int = None,
        normalize_basis: bool = None,
        reference_peak_interval: List[float] = None,
        complex_output: bool = None,
        scaling_factors: dict = None,
        decoder_class: str = None,
        lineshape_kernel_size: float = None,
        dynamic_decoder_options: Dict[str, Any] = None,
        repetition_axis: List[float] = None,
        shared_baseline_spline: bool = None,
    ) -> None:
        """

        Args:
            n_splines (int): Number of splines to use for the baseline.
            normalize_basis (bool): Normalize the basis signals.
            reference_peak_interval (List[float]): Interval around the reference peak (in ppm) to normalize the basis.
            complex_output (bool): Whether the output of the decoder should also calculate the cubic spline for the
                imaginary part.
            scaling_factors (dict): Scaling factors for the basis signals, frequency shifts and damping
                (Lorentzian).
            decoder_class (str): The decoder class to use. Can be "StandardModelDecoder" or "LCModelDecoder".
            lineshape_kernel_size (float): Size of the kernel for the lineshape convolution in ppm.
            dynamic_decoder_options (Dict[str, Any]): Dictionary containing extra information for the definition of the
                dynamic decoder.
            repetition_axis (List[float]): Axis for acquisition times of the repetitions in minutes.
            shared_baseline_spline (bool): Whether the baseline spline should be shared among time points or not.
        """

        self.n_splines = n_splines
        self.normalize_basis = normalize_basis
        self.reference_peak_interval = reference_peak_interval
        self.complex_output = complex_output
        self.scaling_factors = scaling_factors
        self.lineshape_kernel_size = lineshape_kernel_size
        if decoder_class not in [
            "UniversalDynamicDecoder",
            None,
        ]:
            raise ValueError(
                f"Decoder class must be 'UniversalDynamicDecoder' or None, but got {decoder_class}."
            )
        self.decoder_class = decoder_class
        self.dynamic_decoder_options = dynamic_decoder_options
        self.repetition_axis = repetition_axis
        self.shared_baseline_spline = shared_baseline_spline


class LossFunctionConfig:
    def __init__(
        self,
        baseline_spline_regularization_slope: float = None,
        baseline_spline_regularization_l2: float = None,
        baseline_spline_regularization_curvature: float = None,
        lineshape_kernel_regularization_curvature: float = None,
        sigma_gamma_l: float = None,
        sigma_epsilon_l: float = None,
        loss_function_type: str = None,
        reconstruction_scaling: Union[float, str] = None,
        dynamic_spline_regularization_curvature: float = None,
        dynamic_spline_regularization_target_quantities: List[str] = None,
        weights_time_axis: List[float] = None,
    ) -> None:
        """
        The cost function will use the complex reconstruction loss if the complex_output argument for the decoder is
        set to True. Otherwise, only the real part of the signals will be used for the reconstruction loss.

        Args:
            baseline_spline_regularization_slope (float): Regularization parameter for the spline weights penalizing
                quickly changing baselines
            baseline_spline_regularization_l2 (float): Regularization parameter for the spline weights penalizing large
                values of the spline
            baseline_spline_regularization_curvature (float): Regularization parameter for the spline weights penalizing
                the curvature of the spline.
            lineshape_kernel_regularization_curvature (float): Regularization parameter for the penalty on the curvature
                of the lineshape kernel.
            sigma_gamma_l (float): see Provencher LCModel paper. Only relevant for the LCModel loss function. Value
                should be in units of Hz.
            sigma_epsilon_l (float): see Provencher LCModel paper. Only relevant for the LCModel loss function. Value
                should be in units of Hz.
            loss_function_type (str): Type of the loss function. Can be "StandardLossFunction" or "LCModelLossFunction".
            reconstruction_scaling (Union[float, str]): Scaling factor for the reconstruction loss. If "auto",
                the scaling factor is determined automatically as in the LCModel paper.
            dynamic_spline_regularization_curvature (float): Regularization parameter for the dynamic spline. Only
                relevant for dynamic cost functions.
            dynamic_spline_regularization_target_quantities (List[str]): List of quantities to be considered for the
                dynamic spline regularization. Can contain "amplitudes" and "delta_f".
            weights_time_axis (List[float]): Weights for each time step.
        """

        self.baseline_spline_regularization_slope = baseline_spline_regularization_slope
        self.baseline_spline_regularization_l2 = baseline_spline_regularization_l2
        self.baseline_spline_regularization_curvature = (
            baseline_spline_regularization_curvature
        )
        self.lineshape_kernel_regularization_curvature = (
            lineshape_kernel_regularization_curvature
        )
        self.sigma_gamma_l = sigma_gamma_l
        self.sigma_epsilon_l = sigma_epsilon_l
        if loss_function_type not in [
            "StandardLossFunction",
            "LCModelLossFunction",
            "DynamicLCModelLossFunction",
            None,
        ]:
            raise ValueError(
                f"Loss function type must be 'StandardLossFunction', 'LCModelLossFunction',"
                f" 'DynamicLCModelLossFunction' or None, but got {loss_function_type}."
            )
        self.loss_function_type = loss_function_type
        if (
            reconstruction_scaling is not None
            and not isinstance(reconstruction_scaling, float)
            and not reconstruction_scaling == "auto"
        ):
            raise ValueError(
                f"Reconstruction scaling must be a float, 'auto' or None, but got {reconstruction_scaling}."
            )
        self.reconstruction_scaling = reconstruction_scaling
        self.dynamic_spline_regularization_curvature = (
            dynamic_spline_regularization_curvature
        )
        self.dynamic_spline_regularization_target_quantities = (
            dynamic_spline_regularization_target_quantities
        )
        self.weights_time_axis = weights_time_axis


class TrainingConfig:
    def __init__(
        self,
        batch_size: int = None,
        learning_rate: float = None,
        weight_decay: float = None,
        epochs: int = None,
        dropout: float = None,
        reduce_learning_rate: float = None,
        validation_split: float = None,
        n_workers: int = None,
        early_stopping: int = None,
        load_from_checkpoint: str = None,
        milestones: List[Union[int, float]] = None,
    ) -> None:
        """
        Args:
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            weight_decay (float): Weight decay parameter of the Adam optimizer.
            epochs (int): Number of epochs.
            dropout (float): Dropout rate.
            reduce_learning_rate (float): Factor to reduce the learning rate by. AKA "gamma"
            validation_split (float): Fraction of the training data to use as validation data.
            n_workers (int): Number of workers to use for the data loader.
            early_stopping (int): Patience for early stopping. If early_stopping <= 0, early stopping is disabled.
            load_from_checkpoint (str): Name of the checkpoint to be loaded (at the default location in
                the saving_path). If None is provided, no model will be loaded.
            milestones (List[Union[int, float]]): Milestones for the learning rate scheduler. (Integer) numbers >= 1 are
                interpreted as epochs, floats < 1 are interpreted as fractions of the total number of epochs.
        """

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dropout = dropout
        self.reduce_learning_rate = reduce_learning_rate
        self.validation_split = validation_split
        self.n_workers = n_workers
        self.early_stopping = early_stopping
        self.load_from_checkpoint = load_from_checkpoint
        self.milestones = milestones


class TestConfig:
    def __init__(
        self,
        conjugate_full_subject_signals: bool = None,
        full_subject_name: Union[str, List[str]] = None,
        plot_at_index: List[List[int]] = None,
        plot_at_time_point: int = None,
        b0_correction_path: str = None,
    ) -> None:
        """
        Args:
            conjugate_full_subject_signals (bool): Whether the time-domain signals from the test subject should be
                conjugated.
            full_subject_name (str): Name of the full subject to be tested.
            plot_at_index (List[List[int]]): List of fixed indices [i,j,k] during plotting using the SubjectQuantifier
                At these indices, the spectra fill be plotted (in addition to the metabolic maps).
            plot_at_time_point (int): time point to plot the metabolic maps (int between 0 and 7).
            b0_correction_path (str): Path to B0 maps. If None is provided, no correction will be done.
        """

        self.conjugate_full_subject_signals = conjugate_full_subject_signals
        self.full_subject_name = full_subject_name
        self.plot_at_index = plot_at_index
        self.plot_at_time_point = plot_at_time_point
        self.b0_correction_path = b0_correction_path


class ValidationConfig:
    def __init__(
        self,
        plot_at_index: List[List[int]] = None,
        plot_ratio_denominator: str = None,
        plot_ratio_numerator: List[str] = None,
        plot_ratio_slice: int = None,
        maximum_plot_ratio: List[float] = None,
        conjugate_full_subject_signals: bool = None,
        full_subject_name: str = None,
        plot_at_time_point: int = None,
        every_n_epochs: int = None,
        b0_correction_path: str = None,
        divide_by_mean: bool = None,
    ) -> None:
        """

        Args:
            plot_at_index (List[List[int]]): List of fixed indices [i,j,k] during plotting using the SubjectQuantifier
                Callback. If all spectra along a certain axis should be plotted, set the corresponding index to None.
                For each set of indices, one Quantifier will be created.
            plot_ratio_denominator (str): The plots from the SubjectQuantifier Callback are ratios between certain
                metabolites. This parameter specifies the denominator of the ratio.
            plot_ratio_numerator (List[str]): The plots from the SubjectQuantifier Callback are ratios between certain
                metabolites. This parameter specifies the numerator(s) of the ratio(s).
            plot_ratio_slice (int): slice index to be used to create the metabolic maps.
            maximum_plot_ratio (List[float]): Maximum value(s) of the ratio(s) to be plotted.
            conjugate_full_subject_signals (bool): Whether the time-domain signals from the test subject should be
                conjugated.
            full_subject_name (str): name of the subject to be quantified in validation.
            plot_at_time_point (int): time point to plot the metabolic maps (int between 0 and 7).
            every_n_epochs (int): Do the validation every nth epoch.
            b0_correction_path (str): Path to B0 maps. If None is provided, no correction will be done.
            divide_by_mean (bool): Whether the numerator metabolic maps should be divided by the
                denominator map or the mean of the denominator map.
        """
        if plot_ratio_numerator is not None:
            if maximum_plot_ratio is None:
                raise ValueError(
                    "If plot_ratio_numerator is provided, maximum_plot_ratio must be provided as well."
                )
            if len(plot_ratio_numerator) != len(maximum_plot_ratio):
                raise ValueError(
                    "The number of elements in plot_ratio_numerator and maximum_plot_ratio must be the same."
                )
        self.plot_at_index = plot_at_index
        self.plot_ratio_denominator = plot_ratio_denominator
        self.plot_ratio_numerator = plot_ratio_numerator
        self.maximum_plot_ratio = maximum_plot_ratio
        self.conjugate_full_subject_signals = conjugate_full_subject_signals
        self.full_subject_name = full_subject_name
        self.plot_ratio_slice = plot_ratio_slice
        self.plot_at_time_point = plot_at_time_point
        self.every_n_epochs = every_n_epochs
        self.b0_correction_path = b0_correction_path
        self.divide_by_mean = divide_by_mean


class PytorchConfig:
    def __init__(
        self,
        num_threads: int = None,
        float32_matmul_precision: str = None,
        default_type: str = None,
        device: str = None,
    ) -> None:
        """
        Args:
            num_threads (int): Number of threads for parallel processing.
            float32_matmul_precision (str): Precision of the matrix multiplication.
            device (str): Device to run all tensor operations on. Can be 'cpu' or f'cuda:{gpu_id}'.
        """
        self.device = torch.device(device) if device is not None else device
        self.num_threads = num_threads
        self.float32_matmul_precision = float32_matmul_precision
        if default_type not in ["single", "double", None]:
            raise ValueError(
                f"Default type must be 'single' or 'double', but got {default_type}"
            )
        if default_type == "single":
            self.default_type = torch.float32
        elif default_type == "double":
            self.default_type = torch.float64
        else:
            self.default_type = None


class WandbConfig:

    def __init__(self, project_name: str = None, key: str = None) -> None:
        """
        Args:
            project_name (str): Name of the Wandb project.
            key (str): Your wandb API key.
        """
        self.project_name = project_name
        self.key = key


class Configuration:
    """Class to hold all the configuration parameters for the pipeline."""

    def __init__(
        self,
        io_config: IOConfig = IOConfig(),
        preprocessor_config: PreprocessorConfig = PreprocessorConfig(),
        encoder_config: EncoderConfig = EncoderConfig(),
        decoder_config: DecoderConfig = DecoderConfig(),
        loss_function_config: LossFunctionConfig = LossFunctionConfig(),
        training_config: TrainingConfig = TrainingConfig(),
        test_config: TestConfig = TestConfig(),
        validation_config: ValidationConfig = ValidationConfig(),
        pytorch_config: PytorchConfig = PytorchConfig(),
        wandb_config: WandbConfig = WandbConfig(),
    ) -> None:
        self.io_config = io_config
        self.preprocessor_config = preprocessor_config
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.loss_function_config = loss_function_config
        self.training_config = training_config
        self.test_config = test_config
        self.validation_config = validation_config
        self.pytorch_config = pytorch_config
        self.wandb_config = wandb_config

    @classmethod
    def from_json(cls, json_file: str) -> Self:
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        f.close()
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> Self:
        io_config = IOConfig(**config_dict["io_config"])
        preprocessor_config = PreprocessorConfig(**config_dict["preprocessor_config"])
        encoder_config = EncoderConfig(**config_dict["encoder_config"])
        decoder_config = DecoderConfig(**config_dict["decoder_config"])
        loss_function_config = LossFunctionConfig(**config_dict["loss_function_config"])
        training_config = TrainingConfig(**config_dict["training_config"])
        test_config = TestConfig(**config_dict["test_config"])
        validation_config = ValidationConfig(**config_dict["validation_config"])
        pytorch_config = PytorchConfig(**config_dict["pytorch_config"])
        wandb_config = WandbConfig(**config_dict["wandb_config"])
        return cls(
            io_config=io_config,
            preprocessor_config=preprocessor_config,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            loss_function_config=loss_function_config,
            test_config=test_config,
            training_config=training_config,
            validation_config=validation_config,
            pytorch_config=pytorch_config,
            wandb_config=wandb_config,
        )


class SolverConfiguration:
    """Class to hold all the configuration parameters for fitting using the Solver class."""

    def __init__(
        self,
        max_iter: int = None,
        patience: int = None,
        delta: float = None,
        loss_function_config: LossFunctionConfig = LossFunctionConfig(),
        x0: Union[np.ndarray, List[float]] = None,
        input_size: int = None,
        learning_rate: float = None,
        device: str = None,
    ) -> None:
        """
        Args:
            max_iter (int): Maximum number of iterations.
            patience (int): Patience for the convergence criterion. The optimization will   terminate if the improvement
                is less than delta for patience consecutive iterations.
            delta (float): Relative improvement (in percent) needed to classify an optimization step as sufficiently
                improving.
            loss_function_config (LossFunctionConfig): Definition of the cost function used for the optimization.
            x0 (Union[np.ndarray, List[float]], float): Initial guess to be used for the fitting. If a float is provided
                , a constant vector with corresponding components are used as initial guess.
            input_size (int): Size of the input data.
            learning_rate (float): Learning rate of the optimizer.
            device (torch.device): device to be used for the optimization
        """

        self.max_iter = max_iter
        self.patience = patience
        self.delta = delta
        self.loss_function_config = loss_function_config
        self.x0 = x0
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.device = device

    @classmethod
    def from_dict(cls, config_dict: dict) -> Self:
        return cls(
            max_iter=config_dict["max_iter"],
            patience=config_dict["patience"],
            delta=config_dict["delta"],
            loss_function_config=LossFunctionConfig(
                **config_dict["loss_function_config"]
            ),
            x0=config_dict["x0"],
            input_size=config_dict["input_size"],
            learning_rate=config_dict["learning_rate"],
            device=config_dict["device"],
        )

    @classmethod
    def from_json(cls, json_file: str) -> Self:
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        f.close()
        return cls.from_dict(config_dict)
