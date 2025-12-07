import subprocess
from typing import Dict, List, Union

import nilearn
import numpy as np
import torch
from nibabel import Nifti1Image
from numpy.fft import fft as np_fft
from numpy.fft import fftn
from numpy.fft import fftshift as np_fftshift
from numpy.fft import ifft as np_ifft
from numpy.fft import ifftn
from numpy.fft import ifftshift as np_ifftshift
from scipy.io import savemat
from scipy.ndimage import zoom
from scipy.signal.windows import hamming
from torch import Tensor
from torch.fft import fft as torch_fft
from torch.fft import fftshift as torch_fftshift
from torch.fft import ifft as torch_ifft
from torch.fft import ifftshift as torch_ifftshift


def resolve_metabolite_name(name: str, resolve_glx: bool = True) -> List[str]:
    """Resolves common abbreviation to the corresponding list of metabolites.

    Args:
        name (str): Metabolite name.
        resolve_glx (bool): Whether to resolve Glx to Glu and Gln.

    Returns:
        int: Index of the metabolite.
    """

    if name == "tCho":
        return ["GPC", "PCh"]
    elif name == "tNAA":
        return ["NAA", "NAAG"]
    elif name == "tCr":
        return ["Cr", "PCr"]
    elif name == "Glx":
        if resolve_glx:
            return ["Glu", "Gln"]
        else:
            return ["Glx"]
    else:
        return [name]


def b0_correction(
    fid: Union[np.ndarray, Tensor],
    b0_shift: Union[np.ndarray, Tensor],
    time_axis: Union[np.ndarray, Tensor],
) -> Union[np.ndarray, Tensor]:
    """Corrects an FID for a given B0 shift.

    Args:
        fid (Union[np.ndarray, Tensor]): Input FIDs of shape (n_fids, signal_length).
        b0_shift (float): B0 shift in Hz of shape (n_fids, 1).
        time_axis (Union[np.ndarray, Tensor]): Time axis of the FIDs.

    Returns:
        Union[np.ndarray, Tensor]: Corrected FID.
    """
    if not isinstance(fid, (np.ndarray, Tensor)):
        raise ValueError("fid must be a numpy array or a torch tensor.")
    if not isinstance(time_axis, (np.ndarray, Tensor)):
        raise ValueError("time_axis must be a numpy array or a torch tensor.")
    if not isinstance(b0_shift, (np.ndarray, Tensor)):
        raise ValueError("b0_shift must be a numpy array or a torch tensor.")

    if isinstance(fid, np.ndarray):
        fid *= np.exp(-1j * 2 * np.pi * time_axis * b0_shift)
    else:
        fid *= torch.exp(-1j * 2 * np.pi * time_axis * b0_shift)

    return fid


def ifft(
    x: Union[Tensor, np.ndarray], dim: int = -1, norm: str = "backward"
) -> Union[Tensor, np.ndarray]:
    """Applies the inverse Fourier transform to the input data.

    Args:
        x (Union[Tensor, np.ndarray]): Input data.
        dim (int): Dimension to apply the inverse Fourier transform to.
        norm (str): Normalization to apply. Can be None, 'ortho', 'forward', or 'backward'. Default is 'backward'.

    Returns:
        Union[Tensor, np.ndarray]: Output data.
    """
    if isinstance(x, Tensor):
        return torch_ifft(x, dim=dim, norm=norm)
    elif isinstance(x, np.ndarray):
        return np_ifft(x, axis=dim, norm=norm)
    else:
        raise ValueError("data must be a numpy array or a torch tensor.")


def fft(
    x: Union[Tensor, np.ndarray], dim: int = -1, norm: str = "backward"
) -> Union[Tensor, np.ndarray]:
    """Applies the Fourier transform to the input data.

    Args:
        x (Union[Tensor, np.ndarray]): Input data.
        dim (int): Dimension to apply the Fourier transform to.
        norm (str): Normalization to apply. Can be None, 'ortho', 'forward', or 'backward'. Default is 'backward'.

    Returns:
        Union[Tensor, np.ndarray]: Output data.
    """

    if isinstance(x, Tensor):
        return torch_fft(x, dim=dim, norm=norm)
    elif isinstance(x, np.ndarray):
        return np_fft(x, axis=dim, norm=norm)
    else:
        raise ValueError("data must be a numpy array or a torch tensor.")


def fftshift(x: Union[Tensor, np.ndarray], dim: int = -1) -> Union[Tensor, np.ndarray]:
    """Shifts the zero frequency component to the center of the spectrum.

    Args:
        x (Union[Tensor, np.ndarray]): Input data.
        dim (int): Dimension to apply the shift to.

    Returns:
        Union[Tensor, np.ndarray]: Output data.
    """
    if isinstance(x, Tensor):
        return torch_fftshift(x, dim=dim)
    elif isinstance(x, np.ndarray):
        return np_fftshift(x, axes=dim)
    else:
        raise ValueError("data must be a numpy array or a torch tensor.")


def ifftshift(x: Union[Tensor, np.ndarray], dim: int = -1) -> Union[Tensor, np.ndarray]:
    """Shifts the zero frequency component to the center of the spectrum.

    Args:
        x (Union[Tensor, np.ndarray]): Input data.
        dim (int): Dimension to apply the shift to.

    Returns:
        Union[Tensor, np.ndarray]: Output data.
    """

    if isinstance(x, Tensor):
        return torch_ifftshift(x, dim=dim)
    elif isinstance(x, np.ndarray):
        return np_ifftshift(x, axes=dim)
    else:
        raise ValueError("data must be a numpy array or a torch tensor.")


def resize_nifti_image(
    image: Nifti1Image, new_shape: np.ndarray, interpolation: str = "continuous"
) -> Nifti1Image:
    """Resizes an image to a new shape.

    Args:
        image (nibabel.Nifti1Image): Input image.
        new_shape (tuple): New shape of the image.
        interpolation (str): Interpolation method. Can be 'continuous' or 'nearest'.

    Returns:
        nibabel.Nifti1Image: Resized image.
    """
    input_shape = np.asarray(image.shape, dtype=np.float16)
    ras_image = nilearn.image.reorder_img(image, resample=interpolation)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape / output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return nilearn.image.resample_img(
        ras_image,
        target_affine=new_affine,
        target_shape=output_shape,
        interpolation=interpolation,
    )


def imresize3(
    volume: np.ndarray, new_shape: List[int], method: str = "nearest"
) -> np.ndarray:
    """
    Resize a 3D volume to a new new_shape. Congruent with the matlab function imresize3.

    Parameters:
    volume (np.ndarray): The input 3D volume to be resized.
    new_shape (tuple of int): The target shape as (new_depth, new_height, new_width).
    method (str): Interpolation method; options are 'linear', 'nearest', or 'cubic'.
                  - 'linear' for trilinear interpolation
                  - 'nearest' for nearest-neighbor interpolation
                  - 'cubic' for tricubic interpolation

    Returns:
        np.ndarray: The resized 3D volume.
    """
    # Calculate zoom factors for each dimension
    depth_factor = new_shape[0] / volume.shape[0]
    height_factor = new_shape[1] / volume.shape[1]
    width_factor = new_shape[2] / volume.shape[2]

    # Mapping methods to order parameter in scipy.ndimage.zoom
    order_dict = {"nearest": 0, "linear": 1, "cubic": 3}

    # Resize volume
    resized_volume = zoom(
        volume, (depth_factor, height_factor, width_factor), order=order_dict[method]
    )

    return resized_volume


def hamming_filter(
    x: np.ndarray,
    axes: List[int],
    filter_width: int = 100,
    transform_to_kspace: bool = True,
    invert_filter: bool = False,
) -> np.ndarray:
    """
    Apply a Hamming filter to (k-space) data in `x`.

    Translated from MATLAB code by Bernhard Strasser.

    Args:
    x (np.ndarray): Input array to which the filter should be applied.
    axes List(int): Dimensions along which to apply the filter.
    filter_width (float): Width of the filter in percentage. Default is 100%.
    transform_to_kspace (bool): If False, FFT is applied before and after filtering.
    invert_filter (bool): If True, the filter is inverted.

    Returns:
    x (np.ndarray): Filtered output array.
    """

    shape = x.shape

    # FFT to k-space if necessary
    if transform_to_kspace:
        x = np_fftshift(fftn(np_ifftshift(x, axes=axes), axes=axes), axes=axes)

    # Outer product approach
    filter = np.ones(shape)
    for dim in axes:
        hamming_1D = hamming(int(np.ceil(filter_width / 100 * shape[dim])), sym=True)
        hamming_1D = np.concatenate(
            [
                hamming_1D[: int(np.ceil(len(hamming_1D) / 2))],
                np.ones(shape[dim] - len(hamming_1D)),
                hamming_1D[int(np.ceil(len(hamming_1D) / 2)) :],
            ]
        )

        hamming_1D = hamming_1D.reshape(
            [1] * dim + [len(hamming_1D)] + [1] * (len(shape) - dim - 1)
        )
        for i, size in enumerate(shape):
            if i != dim:
                hamming_1D = hamming_1D.repeat(size, axis=i)
        filter *= hamming_1D

    # Invert the Hamming Filter if needed
    if invert_filter:
        filter = np.where(filter == 0, 0, 1 / filter)

    # Apply Hamming Filter
    x *= filter

    # Transform back to Image Space if necessary
    if transform_to_kspace:
        x = np_fftshift(ifftn(np_ifftshift(x, axes=axes), axes=axes), axes=axes)

    return x


def savemat73(path: str, data: Dict) -> None:
    """Saves a dict as mat7.3 file.

    Args:
        data (Dict): Dictionary to be saved as .mat file.
        path (str): Path to the .mat file.
    """

    savemat(path, data)
    bash_commands = (
        "matlab -nodisplay -nosplash -nodesktop -r \"try, load('{0}'); save('{0}', '-v7.3'); "
        'catch, exit(1); end; exit(0);"'
    ).format(path)
    exit_code = subprocess.run(bash_commands, shell=True)
    if exit_code.returncode != 0:
        raise ValueError("Error in saving the .mat file.")
