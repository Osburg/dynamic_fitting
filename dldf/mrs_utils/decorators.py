from typing import Callable

import numpy as np
import torch
from torch import Tensor


def torch_as_numpy_function(torch_function: Callable) -> Callable:
    """Turns a function operating on torch tensors to a function operationg on numpy arrays."""

    def torch_as_numpy(x: np.ndarray) -> np.ndarray:
        x = torch.tensor(x, device="cpu", requires_grad=False)
        return torch_function(x).numpy()

    return torch_as_numpy


def numpy_as_torch_function(numpy_function: Callable) -> Callable:
    """Turns a function operating on numpy arrays to a function operationg on torch tensors."""

    def numpy_as_torch(x: Tensor) -> Tensor:
        return torch.tensor(
            numpy_function(x.detach().cpu().numpy()), dtype=x.dtype, device=x.device
        )

    return numpy_as_torch
