import json
import re
import warnings
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from typing_extensions import Self

from dldf.mrs_utils.axis import Axis
from dldf.mrs_utils.constants import GAMMA_D2, GAMMA_H1


class Basisset:
    """Class to store metabolite basis sets."""

    def __init__(
        self,
        fids: np.ndarray,
        metabolite_names: List[str],
        conjugate_basis: bool = False,
        dwelltime: float = None,
        reference_frequency: float = None,
        nucleus: str = "H1",
    ) -> None:
        """Initializes the Basisset object.

        Args:
            fids (np.ndarray): 2D numpy array The data to be stored in the container.
                The samples should be stored along the first dimension.
            metabolite_names (List[str]): List of metabolite names.
            conjugate_basis (bool): Optional. If True, the basis set is conjugated.
            dwelltime (float): Optional. The dwell time of the FID signals.
            reference_frequency (float): Optional. The reference frequency of the data (i.e. water frequency at B0).
            nucleus (str): Nucleus used for this basis. Can be 'H1' or 'D2'. Default is 'H1'.
        """
        if not len(metabolite_names) == fids.shape[1]:
            raise ValueError(
                "The number of metabolite names should be equal to the number of FID signals."
            )
        self.fids: np.ndarray = np.array(fids, dtype=np.complex64)
        self.spectra = np.array(
            [
                np.fft.fftshift(np.fft.fft(self.fids, axis=0)[:, i])
                for i in range(self.fids.shape[1])
            ],
            dtype=np.complex64,
        ).T
        self.metabolite_names: List[str] = metabolite_names

        if nucleus not in ["H1", "D2"]:
            raise ValueError("The nucleus must be either 'H1' or 'D2'.")
        self.nucleus = nucleus
        self.gamma = GAMMA_H1 if nucleus == "H1" else GAMMA_D2

        self.dwelltime = dwelltime
        self.reference_frequency = reference_frequency
        self.conjugate_basis = False
        if conjugate_basis:
            self.conjugate()

    def __len__(self) -> int:
        return self.fids.shape[1]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.fids[:, idx]

    def by_key(self, name: str, domain: str = "frequency") -> np.ndarray:
        """Returns the signal of a metabolite by name.

        Args:
            name (str): The name of the metabolite.
            domain (str): Optional. The domain of the signal. Can be "frequency" or "time". Defaults to "frequency".
        """
        if domain not in ["frequency", "time"]:
            raise ValueError("Domain must be 'frequency' or 'time'.")

        if name not in self.metabolite_names:
            raise ValueError(f"Metabolite {name} not found in the basis set.")

        if domain == "frequency":
            return self.spectra[:, self.metabolite_names.index(name)]
        else:
            return self.fids[:, self.metabolite_names.index(name)]

    def __setitem__(self, idx: int, value: np.ndarray) -> None:
        self.fids[:, idx] = value
        self.spectra[:, idx] = np.fft.fftshift(np.fft.fft(value))

    def __iter__(self):
        return self.fids.T.__iter__()

    def __next__(self):
        return self.fids.T.__next__()

    def get_name_from_index(self, idx: int) -> str:
        """Returns the metabolite name corresponding to the index."""
        return self.metabolite_names[idx]

    def conjugate(self) -> None:
        """Conjugates the basis set."""
        self.conjugate_basis = not self.conjugate_basis
        self.fids = np.conj(self.fids)
        self.spectra = np.array(
            [
                np.fft.fftshift(
                    np.fft.fft(self.fids.astype(np.complex128), axis=0)[:, i]
                )
                for i in range(self.fids.shape[1])
            ],
            dtype=np.complex64,
        ).T

    @classmethod
    def from_matlab(
        cls,
        path: str,
        metabolite_names: List[str] = None,
        conjugate_basis: bool = False,
        dwelltime: float = None,
        reference_frequency: float = None,
        nucleus: str = "H1",
    ) -> Self:
        """
        Args:
            path (str): Path to the .mat file containing the fid signals and the metabolite names.
            metabolite_names (List[str]): Optional. List of metabolite names. If None is
                provided, the names will be taken from the field "metabolite_names" of the input file.
            conjugate_basis (bool): Optional. If True, the basis set will be conjugated.
            dwelltime (float): Optional. The dwell time of the FID signals.
            reference_frequency (float): Optional. The reference frequency of the data (i.e. water frequency
                at B0). Defaults to None.
            nucleus (str): Optional. Nucleus used for this basis. Can be 'H1' or 'D2'. Default is 'H1'.

        Returns:
            A Basisset object.
        """
        fids = sio.loadmat(path).get("data")
        if metabolite_names is None:
            metabolite_names = sio.loadmat(path).get("metabolite_names")
        return cls(
            fids=fids,
            metabolite_names=list(metabolite_names),
            conjugate_basis=conjugate_basis,
            dwelltime=dwelltime,
            reference_frequency=reference_frequency,
            nucleus=nucleus,
        )

    @classmethod
    def from_npz(
        cls,
        path: str,
        metabolite_names: List[str],
        conjugate_basis: bool = False,
        dwelltime: float = None,
        reference_frequency: float = None,
        nucleus: str = "H1",
    ) -> Self:
        """
        Args:
            path (str): Path to the npz file containing the fid .
            metabolite_names (List[str]): List of metabolite names.
            conjugate_basis (bool): Optional. If True, the basis set will be conjugated.
            dwelltime (float): Optional. The dwell time of the FID signals.
            reference_frequency (float): Optional. The reference frequency of the data (i.e. water frequency at B0).
            nucleus (str): Optional. Nucleus used for this basis. Can be 'H1' or 'D2'. Default is 'H1'.

        Returns:
            A Basisset object.
        """
        data = np.load(path)
        data = data["data"].T
        return cls(
            fids=data,
            metabolite_names=metabolite_names,
            conjugate_basis=conjugate_basis,
            reference_frequency=reference_frequency,
            dwelltime=dwelltime,
            nucleus=nucleus,
        )

    @classmethod
    def from_RAW(
        cls,
        path_list: List[str],
        metabolite_names: List[str] = None,
        conjugate_basis: bool = True,
        dwelltime: float = None,
        reference_frequency: float = None,
        nucleus: str = "H1",
    ) -> Self:
        """
        Args:
            path_list (List[str]): List of paths to the .raw files.
            metabolite_names List[str]: Optional. List of metabolite names.
                if no names are provided, the names will be taken from the field
                "ID" of the input files.
            conjugate_basis (bool): Optional. If True, the basis set will be conjugated.
            dwelltime (float): Optional. The dwell time of the FID signals.
            reference_frequency (float): Optional. The reference frequency of the data (i.e. water frequency at B0).
            nucleus (str): Optional. Nucleus used for this basis. Can be 'H1' or 'D2'. Default is 'H1'.

        Returns:
            A Basisset object.
        """
        if metabolite_names is not None:
            assert len(metabolite_names) == len(path_list)

        out = {}
        for path in path_list:
            metabolite = io_readlcmraw(path)
            out.update(metabolite)

        fids = np.array([out[metab] for metab in list(out.keys())]).T
        if metabolite_names is None:
            metabolite_names = list(out.keys())
        conjugate_basis = False

        return cls(
            fids=fids,
            metabolite_names=metabolite_names,
            conjugate_basis=conjugate_basis,
            dwelltime=dwelltime,
            reference_frequency=reference_frequency,
            nucleus=nucleus,
        )

    @classmethod
    def from_jMRUI(
        cls,
        path_list: List[str],
        metabolite_names: List[str] = None,
        conjugate_basis: bool = True,
        dwelltime: float = None,
        reference_frequency: float = None,
        nucleus: str = "H1",
        acquisition_delay_index: int = None,
    ) -> Self:
        """
        Args:
            path_list (List[str]): List of paths to the .txt files.
            metabolite_names List[str]: Optional. List of metabolite names.
                if no names are provided, the names will be taken from the field
                "Filename" of the input files.
            conjugate_basis (bool): Optional. If True, the basis set will be conjugated.
            dwelltime (float): Optional. The dwell time of the FID signals.
            reference_frequency (float): Optional. The reference frequency of the data (i.e. water frequency at B0).
            nucleus (str): Optional. Nucleus used for this basis. Can be 'H1' or 'D2'. Default is 'H1'.
            acquisition_delay_index (int): Optional. Index pointing at the first entry after the acquisition delay

        Returns:
            A Basisset object.
        """
        if metabolite_names is not None:
            assert len(metabolite_names) == len(path_list)

        # read header
        dwelltimes = []
        metabolite_names_ = []
        for path in path_list:
            with open(path, "r") as f:
                lines = f.readlines()
                header = lines[:22]
                for line in header:
                    if line.startswith("SamplingInterval"):
                        dwelltimes.append(float(line.split(":")[1]) * 1e-3)
                    if line.startswith("Filename"):
                        metabolite_names_.append(line.split(":")[:-4])
        if metabolite_names is None:
            metabolite_names = metabolite_names_
        if not np.allclose(dwelltimes, dwelltimes[0], rtol=1e-3):
            raise RuntimeError("The dwell times of the files are not equal.")
        if dwelltime is None:
            dwelltime = dwelltimes[0]

        # load FID signals
        fids = {}
        for i, path in enumerate(path_list):
            data = np.loadtxt(path, skiprows=22)
            fids[metabolite_names[i]] = (
                data[acquisition_delay_index:, 0]
                + 1j * data[acquisition_delay_index:, 1]
            )

        fids = np.array([fids[metab] for metab in list(fids.keys())]).T
        conjugate_basis = False

        return cls(
            fids=fids,
            metabolite_names=metabolite_names,
            conjugate_basis=conjugate_basis,
            dwelltime=dwelltime,
            reference_frequency=reference_frequency,
            nucleus=nucleus,
        )

    @classmethod
    def from_BASIS(
        cls, path: str, conjugate_basis: bool = True, nucleus: str = "H1"
    ) -> Self:
        """
        Args:
            path (str): Path to the .basis file containing the basis.
            conjugate_basis (bool): Optional. If True, the basis set will be conjugated.
            nucleus (str): Optional. Nucleus used for this basis. Can be 'H1' or 'D2'. Default is 'H1'.
        """
        out = io_readlcmraw_basis(path, conjugate=False)

        metabolite_names = list(out.keys())
        fids = np.array([out[metab]["fids"] for metab in metabolite_names]).T
        # assuming the same dwelltime for all metabolites
        dwelltime = out[metabolite_names[0]]["dwelltime"]
        reference_frequency = out[metabolite_names[0]]["txfrq"] * 1e-6

        return cls(
            fids=fids,
            metabolite_names=metabolite_names,
            conjugate_basis=conjugate_basis,
            dwelltime=dwelltime,
            reference_frequency=reference_frequency,
            nucleus=nucleus,
        )

    def to_npz(self, path: str) -> None:
        np.savez(path, data=self.fids.T, metabolite_names=self.metabolite_names)

    def to_matlab(self, path: str) -> None:
        dict = {
            "__header__": b"MAT-file",
            "__version__": "1.0",
            "__globals__": [],
            "data": self.fids,
            "metabolite_names": self.metabolite_names,
        }
        sio.savemat(path, dict)

    def to_json(self, path: str) -> None:
        dict = {
            "fids_real": self.fids.real.tolist(),
            "fids_imag": self.fids.imag.tolist(),
            "metabolite_names": list(self.metabolite_names),
            "dwelltime": self.dwelltime,
            "reference_frequency": self.reference_frequency,
            "nucleus": self.nucleus,
        }
        with open(path, "w") as f:
            json.dump(dict, f)

    @classmethod
    def from_json(cls, path: str) -> Self:
        with open(path, "r") as f:
            dict = json.load(f)
        fids_real = np.array(dict["fids_real"])
        fids_imag = np.array(dict["fids_imag"])
        fids = fids_real + 1j * fids_imag
        return cls(
            fids=np.array(fids, dtype=np.complex64),
            metabolite_names=dict["metabolite_names"],
            dwelltime=dict["dwelltime"],
            reference_frequency=dict["reference_frequency"],
            nucleus=dict["nucleus"],
        )

    def normalize(self, reference_peak_interval: List[float]) -> None:
        """Normalizes the basis set such that the reference peak has the same height for all elements of the basis.
        The reference peak must be clearly separated from the rest of the spectral components.

        Args:
            reference_peak_interval (List[float]): An interval on the spectral axis that contains the reference peak and
                only the reference peak.
        """
        ppm_axis = Axis.from_time_axis(
            time=np.linspace(
                0, self.dwelltime * (self.fids.shape[0] - 1), self.fids.shape[0]
            ),
            b0=self.reference_frequency / self.gamma,
            nucleus=self.nucleus,
        )
        idx_low = ppm_axis.to_index(reference_peak_interval[0], "ppm")
        idx_high = ppm_axis.to_index(reference_peak_interval[1], "ppm")
        peak_heights = np.max(np.abs(self.spectra[idx_low:idx_high, :]), axis=0)

        if not np.allclose(
            peak_heights, peak_heights[0], rtol=3e-2
        ):  # 3 percent tolerance
            warnings.warn(
                "The peak heights of the basis set are not equal and will be normalized."
            )
            for i in range(len(self)):
                self.fids[:, i] = self.fids[:, i] / peak_heights[i] * peak_heights[0]
            self.spectra = np.array(
                [
                    np.fft.fftshift(np.fft.fft(self.fids, axis=0)[:, i])
                    for i in range(self.fids.shape[1])
                ],
                dtype=np.complex64,
            ).T

    def by_index(self, idx: int, domain: str = "frequency") -> np.ndarray:
        """Returns the data corresponding to the index.

        Args:
            idx (int): The index of the metabolite.
            domain (str): Optional. The data to be returned. Either "time" or "frequency".
                Defaults to "time".

        Returns:
            np.ndarray: The FID signal/spectrum corresponding to the index.
        """
        if idx >= len(self.metabolite_names):
            raise ValueError("The index is out of range.")
        if domain == "time":
            return self.fids[:, idx]
        elif domain == "frequency":
            return self.spectra[:, idx]
        else:
            raise ValueError("The data type should be either 'fids' or 'spectra'.")

    def plot(self, save_path: str, ppm_bounds: List[int]) -> None:
        """Plots the basis set and saves the plot to the specified path.

        Args:
            save_path (str): The path to save the plot.
            ppm_bounds (List[int]): The bounds of the ppm axis.
        """
        time_axis = np.linspace(
            0, self.dwelltime * (self.fids.shape[0] - 1), self.fids.shape[0]
        )
        ppm_axis = Axis.from_time_axis(
            time=time_axis,
            b0=self.reference_frequency / self.gamma,
            nucleus=self.nucleus,
        )

        fig = plt.figure(figsize=(10, np.max([1.1 * len(self), 3.0])))
        ax = fig.add_subplot(111)
        for i in range(len(self)):
            signal = np.abs(self.spectra[:, i])
            ax.plot(ppm_axis._ppm, signal - 1.5 * i, color="black")
            ax.text(
                ppm_bounds[1] + 0.35,
                -1.5 * i,
                self.metabolite_names[i],
                fontsize=11,
                verticalalignment="center",
            )
        ax.set_xlabel("chemical shift [ppm]")
        ax.set_yticks([])
        ax.set_xticks(np.arange(np.ceil(ppm_bounds[0]), np.ceil(ppm_bounds[1]), 0.5))
        ax.vlines(
            np.arange(np.ceil(ppm_bounds[0]), np.ceil(ppm_bounds[1]), 0.5),
            ymin=-1.5 * len(self) + 1,
            ymax=2,
            color="grey",
            linestyles="dotted",
        )
        if ppm_bounds is not None:
            ax.set_xlim(ppm_bounds[0], ppm_bounds[1])
        ax.set_title("Basis and macromolecule signals")
        for spine in fig.gca().spines.values():
            spine.set_visible(False)
        ax.invert_xaxis()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)

    def get_index_from_name(self, name: str) -> List[int]:
        """
        Returns the indices corresponding to a given metabolite (group) name

        Args:
            name (str): metabolite (group) name
        """
        if name is None:
            return None
        elif name == "tCho":
            if "tCho" not in self.metabolite_names:
                indices = [self.metabolite_names.index(n) for n in ["GPC", "PCh"]]
            else:
                indices = [self.metabolite_names.index("tCho")]
        elif name[0] == "t":
            indices = [
                self.metabolite_names.index(n)
                for n in self.metabolite_names
                if name[1:] in n
            ]
        elif name == "Glx":
            if "Glx" not in self.metabolite_names:
                indices = [self.metabolite_names.index(n) for n in ["Glu", "Gln"]]
            else:
                indices = [self.metabolite_names.index("Glx")]
        else:
            indices = [self.metabolite_names.index(name) if name is not None else None]
        return indices


def io_readlcmraw(filename: str) -> dict:
    """
    Read a LCModel .raw file and extract the FID signal from it.

    Parameters:
    filename (str): The path to the .raw file.

    Returns:
    dict: A dictionary containing the metabolite name as key and the FID signal as value.
    """
    out = {}

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        if "ID" in line and "NMID" not in line:
            metabolite_name = line.split(sep="=")[1]
            metabolite_name = metabolite_name.split(".RAW")[0]

        if "$END" in line:
            # Read the FID
            real = []
            imag = []
            while line.strip() and i < len(lines) - 1:
                i += 1
                line = lines[i]
                vals = line.split()
                real += [float(vals[0])]
                imag += [float(vals[1])]

            fid = np.array(real) + 1j * np.array(imag)
            out[metabolite_name] = fid

        i += 1

    return out


####################################################################################
# The following code is translated from MATLAB to Python from the repo        ######
# plotLCM by schorschinho (https://github.com/schorschinho/plotLCMBasis)      ######
# using chatGPT.                                                              ######
####################################################################################


def get_num_from_string(s):
    s = re.sub(r"[;=]", " ", s)
    s = re.sub(r"[^\d.\-eE]", " ", s)
    s = re.sub(r"(?i)e(?![+-])", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def remove_white_spaces(s):
    return re.sub(r"\s+", "", s)


def lcmodel_rng(dix):
    a = 16807.0
    b15 = 32768.0
    b16 = 65536.0
    p = 2147483647.0

    xhi = dix / b16
    xhi = xhi - (xhi % 1.0)
    xalo = (dix - xhi * b16) * a
    leftlo = xalo / b16
    leftlo = leftlo - (leftlo % 1.0)
    fhi = xhi * a + leftlo
    k = fhi / b15
    k = k - (k % 1.0)
    dix = (((xalo - leftlo * b16) - p) + (fhi - k * b15) * b16) + k
    if dix < 0.0:
        dix += p
    randomresult = dix * 4.656612875e-10
    return randomresult, dix


def io_readlcmraw_basis(filename, conjugate=True):
    """
    Read .BASIS file and extract the FID signals and metabolite names.

    Args:
        filename (str): The path to the .basis file.
        conjugate (bool, optional): Whether to conjugate the FID signals. Defaults to True.

    Returns:
        dict: A dictionary containing the extracted information from the .basis file.

    Raises:
        FileNotFoundError: If the specified file does not exist.

    """
    out = {}

    with open(filename, "r") as f:
        lines = f.readlines()

    linewidth = None
    hzpppm = None
    te = None
    dwelltime = None
    Bo = None
    linewidth = None
    spectralwidth = None
    centerFreq = None
    metabName = None

    i = 0
    while i < len(lines):
        line = lines[i]

        if linewidth is None:
            while "FWHMBA" not in line:
                i += 1
                line = lines[i]
            linewidth = float(get_num_from_string(line))

        if hzpppm is None:
            while "HZPPPM" not in line:
                i += 1
                line = lines[i]
            hzpppm = float(get_num_from_string(line))
            Bo = hzpppm / 42.577
            linewidth = linewidth * hzpppm

        if te is None:
            while "ECHOT" not in line:
                i += 1
                line = lines[i]
            te = float(get_num_from_string(line))

        if dwelltime is None:
            while "BADELT" not in line:
                i += 1
                line = lines[i]
            dwelltime = float(get_num_from_string(line))
            spectralwidth = 1 / dwelltime

        if centerFreq is None:
            while "PPMSEP" not in line:
                i += 1
                line = lines[i]
                if "METABO" in line and "METABO_" not in line:
                    break
            if "PPMSEP" in line:
                centerFreq = float(get_num_from_string(line))
            else:
                centerFreq = []

        if metabName is None:
            while not ("METABO" in line and "METABO_" not in line):
                i += 1
                line = lines[i]
            metabName = re.search(
                r"METABO\s*=\s*['\"]?([-_+A-Za-z0-9]+)['\"]?", line
            ).group(1)

        if "$END" in line:
            i += 1
            line = lines[i]
            RF = []
            while not any(x in line for x in ["$NMUSED", "$BASIS"]) and line.strip():
                RF += [float(val) for val in line.split()]
                i += 1
                if i < len(lines):
                    line = lines[i]
                else:
                    break

            specs = np.array(RF[0::2]) + 1j * np.array(RF[1::2])

            if dwelltime < 0:
                dix = 1499
                for rr in range(len(specs)):
                    randomresult, dix = lcmodel_rng(dix)
                    specs[rr] = -specs[rr] * np.exp(-20 * randomresult + 10)

            if conjugate:
                # osburg: here the conjugation property of the fourier transform is used?
                specs = np.flipud(np.fft.fftshift(np.conj(specs)))
            else:
                specs = np.fft.fftshift(specs)

            vectorsize = len(specs)
            sz = (vectorsize, 1)
            if vectorsize % 2 == 0:
                fids = np.fft.ifft(np.fft.ifftshift(specs))
            else:
                fids = np.fft.ifft(np.roll(np.fft.ifftshift(specs), 1))

            f = np.linspace(
                (-1 + 1 / sz[0]) * spectralwidth / 2,
                (1 - 1 / sz[0]) * spectralwidth / 2,
                vectorsize,
            )
            ppm = f / (Bo * 42.577) + 4.68
            t = np.arange(dwelltime, vectorsize * dwelltime + dwelltime, dwelltime)
            txfrq = hzpppm * 1e6
            metabName = remove_white_spaces(metabName)
            if metabName == "-CrCH2":
                metabName = "CrCH2"
            if metabName == "2HG":
                metabName = "bHG"

            out[metabName] = {
                "fids": fids,
                "specs": specs,
                "sz": (vectorsize, 1, 1, 1),
                "n": vectorsize,
                "spectralwidth": abs(spectralwidth),
                "Bo": Bo,
                "te": te,
                "tr": [],
                "dwelltime": abs(1 / spectralwidth),
                "linewidth": linewidth,
                "ppm": ppm,
                "t": t,
                "txfrq": txfrq,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "seq": "",
                "sim": "",
                "dims": {"t": 1, "coils": 0, "averages": 0, "subSpecs": 0, "extras": 0},
                "averages": 1,
                "flags": {
                    "writtentostruct": 1,
                    "gotparams": 1,
                    "leftshifted": 1,
                    "filtered": 0,
                    "zeropadded": 0,
                    "freqcorrected": 0,
                    "phasecorrected": 0,
                    "averaged": 1,
                    "addedrcvrs": 1,
                    "subtracted": 1,
                    "writtentotext": 1,
                    "downsampled": 0,
                    "isFourSteps": 0,
                },
            }

            centerFreq = None
            metabName = None

        i += 1

    return out
