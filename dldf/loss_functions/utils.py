from typing import Type, Union

from dldf.loss_functions import (
    ComplexLCModelLossFunction,
    DynamicComplexLCModelLossFunction,
    DynamicLCModelLossFunction,
    LCModelLossFunction,
)


def get_loss_function_class(
    loss_function_type: str, complex_decoder_output: bool
) -> Union[
    Type[LCModelLossFunction],
    Type[ComplexLCModelLossFunction],
    Type[DynamicLCModelLossFunction],
    Type[DynamicComplexLCModelLossFunction],
]:
    """Get the loss function class based on the loss function and decoder output type.

    Args:
        loss_function_type (str): The loss function type. Can be "StandardLossFunction", "LCModelLossFunction" or
            "DynamicLCModelLossFunction".
        complex_decoder_output (bool): Whether the decoder output is complex.

    Returns:
        class: The decoder class.
    """
    if loss_function_type == "LCModelLossFunction":
        if complex_decoder_output:
            return ComplexLCModelLossFunction
        else:
            return LCModelLossFunction
    elif loss_function_type == "DynamicLCModelLossFunction":
        if complex_decoder_output:
            return DynamicComplexLCModelLossFunction
        else:
            return DynamicLCModelLossFunction
    else:
        raise ValueError(f"Unknown loss function type: {loss_function_type}")
