import numpy as np
import scipy.io as sio
from overrides import override
from torch import Tensor, from_numpy
from torch.utils.data import TensorDataset
from typing_extensions import Self


class MRSContainer(TensorDataset):
    """Class to store mrs(i) data"""

    def __init__(
        self,
        data: np.ndarray,
        ground_truth: np.ndarray = None,
        metabolite_maps: np.ndarray = None,
        **kwargs
    ) -> None:
        """Initializes the Container object which inherits from the Dataset class

        Args:
            data (np.ndarray): The data to be stored in the container.
                The samples should be stored along the 0th dimension.
                The time points should be stored along the 1st dimension.
                If multiple repetitions are stored, the repetitions should be stored along the 2nd dimension.
                If multiple signals are stored, the 2nd and the 1st dimensions will be swapped internally.
            ground_truth (np.ndarray): The ground truth data to be stored in the container. Provide none for
                unsupervised learning.
            metabolite_maps (np.ndarray): The metabolic maps to be stored in the container, which can work as ground
                truths as well.
            **kwargs: Additional keyword arguments to be stored in the container
                device (str): The device to store the data on. Default is "cpu"
                dwelltime (float): The dwell time of the data. Default is None
                reference_frequency (float): The reference frequency of the data. Default is None
                transform (callable): The transform to be applied to the data. Default is lambda x: x.
                This can for example be used to include an augmentation step.

        """

        if "device" in kwargs:
            self.device = kwargs.get("device")
        else:
            self.device = "cpu"

        if "dwelltime" in kwargs:
            self.dwelltime = kwargs.get("dwelltime")
        else:
            self.dwelltime = None

        if "reference_frequency" in kwargs:
            self.reference_frequency = kwargs.get("reference_frequency")
        else:
            self.reference_frequency = None

        if "transform" in kwargs:
            self.transform = kwargs.get("transform")
        else:
            self.transform = lambda x: x

        self.n_signals = data.shape[0]
        self.signal_length = data.shape[1]
        if len(data.shape) > 2:
            self.n_repetitions = data.shape[2]
        else:
            self.n_repetitions = 1

        # Swap the 1st and 2nd dimensions if multiple repetitions are stored
        if len(data.shape) > 2:
            data = data.swapaxes(1, 2)

        tensors = [from_numpy(data.astype("complex64")).to(self.device)]
        if ground_truth is not None:
            if len(ground_truth.shape) > 2:
                ground_truth = ground_truth.swapaxes(1, 2)
            tensors.append(from_numpy(ground_truth.astype("complex64")).to(self.device))

        if metabolite_maps is not None:
            if len(metabolite_maps.shape) > 2:
                metabolite_maps = metabolite_maps.swapaxes(1, 2)
            tensors.append(
                from_numpy(metabolite_maps.astype("complex64")).to(self.device)
            )

        super().__init__(*tensors)

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return tuple(self.transform(tensor[idx]) for tensor in self.tensors)

    @classmethod
    def from_matlab(cls, path: str, **kwargs) -> Self:
        data = sio.loadmat(path).get("data").T
        return cls(data, **kwargs)

    @classmethod
    def from_npz(cls, path: str, **kwargs) -> Self:
        data = np.load(path)
        data = data["data"]
        return cls(data, **kwargs)

    @classmethod
    def from_npy(cls, path: str, **kwargs) -> Self:
        data = np.load(path)
        return cls(data, **kwargs)

    def to_npy(self, path: str) -> None:
        np.save(path, self.tensors[0].detach().cpu().numpy())

    def to_npz(self, path: str) -> None:
        np.savez(path, data=self.tensors[0].detach().cpu().numpy())

    def to_matlab(self, path: str) -> None:
        dict = {
            "__header__": b"MAT-file",
            "__version__": "1.0",
            "__globals__": [],
            "data": self.tensors[0].detach().cpu().numpy().T,
        }
        sio.savemat(path, dict)
