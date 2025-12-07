import argparse
import json
import os
from typing import List

import numpy as np
from scipy.interpolate import interp1d

from dldf.mrs_utils.basisset import Basisset


def filter_basis(
    basis: Basisset,
    names_filtered: List[str],
    n_samples: int,
    dwelltime: float = None,
    n_zero_filling: int = None,
    reference_frequency: float = None,
    normalization: float = None,
) -> Basisset:
    """Reads a basis from a LCModel .basis file and filters it to only include the metabolites in names_filtered.
    The basis signals are interpolated and sampled with the user defined dwelltime and number of samples.

    Args:
        basis (Basisset): Basisset object containing the basis signals
        names_filtered (List[str]): List of metabolites to include in the basis
        dwelltime (float): Dwelltime for the output data. The basis signals are interpolated to this dwelltime.
        n_samples (int): Number of samples taken from the basis signal. The basis will be sampled to this length and
            then zero-filled to .
        n_zero_filling (int): Number of samples for the output data. The basis signals are interpolated to this number
            of samples.
        reference_frequency (float): reference frequency of the basisset (i.e. water frequency at B0 used for the basis)
        normalization (float): Normalization factor for the basis signals. If None, the basis signals are not scaled.
    """

    t = np.linspace(
        0.0, basis.dwelltime * (basis.fids.shape[0] - 1), basis.fids.shape[0]
    )

    dwelltime_new = basis.dwelltime
    if dwelltime is not None:
        dwelltime_new = dwelltime

    reference_frequency_new = basis.reference_frequency
    if reference_frequency is not None:
        reference_frequency_new = reference_frequency

    n_zero_filling_new = basis.fids.shape[0]
    if n_samples is not None:
        n_zero_filling_new = n_zero_filling

    t_new = np.linspace(0.0, dwelltime_new * (n_samples - 1), n_samples)
    fids_new = []
    for i in range(basis.fids.shape[1]):
        if basis.metabolite_names[i] in names_filtered:
            f = interp1d(t, basis.fids[:, i], kind="linear")
            basis_signal = np.zeros(n_zero_filling_new, dtype=np.complex64)
            basis_signal[:n_samples] = f(t_new)
            fids_new.append(basis_signal)
    fids_new = np.array(fids_new).T

    if normalization is not None:
        fids_new *= normalization
        print("Basis scaled by", normalization)

    basis_new = Basisset(
        fids_new,
        names_filtered,
        dwelltime=dwelltime_new,
        reference_frequency=reference_frequency_new,
        nucleus=basis.nucleus,
    )
    return basis_new


if __name__ == "__main__":
    """
    Extracts and processes a basis from a LCModel .basis files. The configuration is defined by a .json file containing
    the following keys:
    - dwelltime: Dwelltime for the output data. The basis signals are interpolated to this dwelltime.
    - signal_length: Length of the signals
    - datasets: Dictionary containing information about preparing the test/training data. Not relevant for this script.
    - basis: Dictionary containing information about preparing the basisset. The following keys are required:
        - basis_root_folder: Path to the directory containing the .basis file.
        - basis_file_name: Name of the .basis file.
        - metabolite_names: List of metabolites to include in the basis.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to JSON file that contains the configuration file for preparing the basis",
    )
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as f:
        config = json.load(f)

    # determine length of the original signals
    n_samples = config["n_samples"]
    save_path = config["basis"]["basis_root_folder"]
    os.makedirs(save_path, exist_ok=True)
    names_met = config["basis"]["metabolite_names"]
    dwelltime = config["dwelltime"]
    n_zero_filling = config["n_zero_filling"]
    normalization = config["basis"]["normalization"]
    nucleus = config["basis"]["nucleus"]

    if isinstance(config["basis"]["basis_file_name"], list):
        basis_paths = [
            config["basis"]["basis_root_folder"] + "/" + name
            for name in config["basis"]["basis_file_name"]
        ]
        if config["basis"]["basis_file_name"][0].split(".")[-1] == "RAW":
            basis = Basisset.from_RAW(
                path_list=basis_paths,
                metabolite_names=names_met,
                conjugate_basis=config["basis"]["conjugate_basis"],
                dwelltime=dwelltime,
                nucleus=nucleus,
                reference_frequency=config["basis"]["reference_frequency"],
            )
        else:
            basis = Basisset.from_jMRUI(
                path_list=basis_paths,
                metabolite_names=names_met,
                conjugate_basis=config["basis"]["conjugate_basis"],
                dwelltime=None,
                nucleus=nucleus,
                reference_frequency=config["basis"]["reference_frequency"],
                acquisition_delay_index=config["basis"]["acquisition_delay_index"],
            )
    else:
        basis_path = (
            config["basis"]["basis_root_folder"]
            + "/"
            + config["basis"]["basis_file_name"]
        )
        basis = Basisset.from_BASIS(
            basis_path,
            conjugate_basis=config["basis"]["conjugate_basis"],
            nucleus=nucleus,
        )

    basis_met = filter_basis(
        basis=basis,
        names_filtered=names_met,
        n_samples=n_samples,
        dwelltime=dwelltime,
        n_zero_filling=n_zero_filling,
        reference_frequency=None,
        normalization=normalization,
    )
    basis_met.to_matlab(save_path + "/basisset_met.mat")
    basis_met.plot(
        save_path + "/basisset_met.png",
        ppm_bounds=config["basis"]["ppm_bounds_for_plotting"],
    )
    print(
        f"Generated metabolite basissets with a dwell time of {dwelltime} s, a signal length of",
        f"{n_samples} zero-filled to a length of {n_zero_filling}. The reference frequency",
        f"is {basis.reference_frequency} Hz.",
    )
    print(
        f"The following metabolites are included in the metabolite basisset: {basis_met.metabolite_names}"
    )
