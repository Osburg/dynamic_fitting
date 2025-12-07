from typing import Type, Union

import numpy as np
import torch


def get_default_type(
    domain: str,
    framework: str = "torch",
) -> Union[
    Type[torch.complex64],
    Type[torch.complex128],
    Type[torch.float32],
    Type[torch.float64],
]:
    """Returns the standard complex or real type.

    Args:
        domain (str): The domain for which the default type should be returned. Can be "complex" or "real".
        framework (str): The framework for which the default type should be returned. Can be "torch" or "numpy".
    """
    if framework not in ["torch", "numpy"]:
        raise ValueError(f"Unknown framework: {framework}")
    if domain not in ["complex", "real"]:
        raise ValueError(f"Unknown domain: {domain}")

    if framework == "torch":
        if domain == "real":
            return torch.get_default_dtype()
        if domain == "complex":
            if torch.get_default_dtype() == torch.float32:
                return torch.complex64
            if torch.get_default_dtype() == torch.float64:
                return torch.complex128

    if framework == "numpy":
        if domain == "real":
            if torch.get_default_dtype() == torch.float32:
                return np.float32
            if torch.get_default_dtype() == torch.float64:
                return np.float64
        if domain == "complex":
            if torch.get_default_dtype() == torch.float32:
                return np.complex64
            if torch.get_default_dtype() == torch.float64:
                return np.complex128
