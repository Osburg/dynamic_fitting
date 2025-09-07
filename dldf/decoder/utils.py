from typing import Type, Union

from dldf.decoder import UniversalDynamicDecoder


def get_decoder_class(
    decoder_type: str,
) -> Union[Type[UniversalDynamicDecoder],]:
    """Get the decoder class based on the decoder type.

    Args:
        decoder_type (str): The decoder type.

    Returns:
        class: The decoder class.
    """
    if decoder_type == "UniversalDynamicDecoder":
        return UniversalDynamicDecoder
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
