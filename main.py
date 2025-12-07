import argparse

import torch

from dldf.config import Configuration
from dldf.engine import Engine
from wandb import login


def do_config_replacements(config: Configuration, args) -> Configuration:
    """Replace the config values with the ones provided in the arguments to facilitate quantification."""
    if args.model_checkpoint is not None:
        config.training_config.load_from_checkpoint = args.model_checkpoint
    if args.full_subject_path_test is not None:
        config.io_config.full_subject_path_test = args.full_subject_path_test
    if args.full_test_subject_name is not None:
        config.test_config.full_subject_name = args.full_test_subject_name
    if args.logging_path is not None:
        config.io_config.logging_path = args.logging_path
    return config


def main(args):
    config = Configuration.from_json(args.config)
    config = do_config_replacements(config, args)
    login(key=config.wandb_config.key)
    torch.set_default_device(
        config.pytorch_config.device
    )  # set the default device for all tensors
    if config.pytorch_config.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(
            config.pytorch_config.float32_matmul_precision
        )  # WandB suggested this
    if config.pytorch_config.num_threads is not None:
        torch.set_num_threads(
            config.pytorch_config.num_threads
        )  # set the number of threads for parallel processing
    if config.pytorch_config.default_type is not None:
        torch.set_default_dtype(
            config.pytorch_config.default_type
        )  # set the default floating point type for torch tensors
    engine = Engine(config)

    if args.mode == "train":
        engine.train()
    if args.mode == "test":
        engine.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to JSON file that contains the pipeline configuration",
    )
    parser.add_argument(
        "--mode",
        required=False,
        type=str,
        help="Mode of the pipeline ('test','train')",
        default="train",
    )

    # optional config replacements
    parser.add_argument(
        "--model_checkpoint",
        required=False,
        type=str,
        help="Optional path to a model checkpoint replacing the one defined in the config file",
        default=None,
    )
    parser.add_argument(
        "--full_subject_path_test",
        required=False,
        type=str,
        help="Optional path to the test subject replacing the one defined in the config file",
        default=None,
    )
    parser.add_argument(
        "--full_test_subject_name",
        required=False,
        type=str,
        help="Optional name of train subject replacing the one defined in the config file",
        default=None,
    )
    parser.add_argument(
        "--logging_path",
        required=False,
        type=str,
        help="Optional path to the logging directory replacing the one defined in the config file",
        default=None,
    )

    args = parser.parse_args()
    main(args)
