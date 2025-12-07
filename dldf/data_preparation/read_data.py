import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import mat73
import nibabel as nib
import numpy as np
import tqdm
from numpy.fft import fft, fftshift


def read_mat(
    root_folder: str, folder_names: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reads all .mat files within the folders listed in folder_names

    Args:
        root_folder (str): Root folder containing the subfolders with the .mat files
        folder_names (List[str]): List of subfolders containing the .mat files

    Returns:
        Tuple of size two containing:
            - The masked signals from the .mat files
            - The brain masks used to mask the signals

    """

    print("Read raw data from .mat files.")

    # Initialize a list to store data from each folder
    csi = []
    mask = []
    dwelltimes = []  # dwelltimes for consistency check
    larmor_frequencies = []  # larmor_frequencies for consistency check
    ground_truth = []
    metabolite_maps = []
    # Iterate through folders
    for folder_name in tqdm.tqdm(folder_names):
        folder_path = os.path.join(root_folder, folder_name)

        # Iterate through files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith("CombinedCSI.mat"):
                file_path = os.path.join(folder_path, file_name)
                mat_file = mat73.loadmat(file_path)

                if "xtMeta" not in mat_file["csi"].keys():
                    mat_file["csi"]["xtMeta"] = mat_file["csi"]["Data"]

                mat_file["csi"]["maps"] = -100 * np.ones(
                    shape=(
                        mat_file["csi"]["Data"].shape[0],
                        mat_file["csi"]["Data"].shape[1],
                        mat_file["csi"]["Data"].shape[2],
                        4,  # 4 metabolites (Glc, Glx, Lac, water=
                        mat_file["csi"]["Data"].shape[4],
                    ),
                    dtype=np.float32,
                )

                for i, name in enumerate(["Glc", "Glx", "Lac", "water"]):
                    for j in range(8):
                        amplitudes = nib.load(
                            folder_path + f"/maps/{j+1}/Orig/{name}_amp_map.nii"
                        ).get_fdata()
                        amplitudes = np.flip(np.flip(amplitudes, axis=1), axis=0)
                        mat_file["csi"]["maps"][:, :, :, i, j] = amplitudes

                csi.append(np.array(mat_file["csi"]["Data"], dtype=np.complex64))
                ground_truth.append(
                    np.array(mat_file["csi"]["xtMeta"], dtype=np.complex64)
                )
                mask.append(np.array(mat_file["mask"], dtype=np.float32))

                maps = np.array(mat_file["csi"]["maps"])
                maps[maps <= 0.0] = -100.0
                metabolite_maps.append(maps)

                dwelltimes.append(mat_file["csi"]["Par"]["Dwelltimes"])
                larmor_frequencies.append(mat_file["csi"]["Par"]["LarmorFreq"])

    # Consistency Check
    if len(np.unique(dwelltimes)) > 1:
        raise ValueError(
            "Dwelltimes are not consistent taking the values",
            np.unique(dwelltimes),
            "ns.",
        )
    print(f"The loaded data has consistent dwelltimes {dwelltimes[0]}")
    print(
        "The larmor frequencies of the loaded data reach from",
        np.min(larmor_frequencies),
        "to",
        np.max(larmor_frequencies),
        "Hz .",
    )

    gathered_data = []
    for csi_, mask_, ground_truth_, metabolite_maps_ in zip(
        csi, mask, ground_truth, metabolite_maps
    ):
        mask_ = np.array(mask_, dtype=bool)
        gathered_data.append(np.array(csi_[mask_, ...], dtype=np.complex64))

    return gathered_data


def create_dataset(
    dataset: np.ndarray,
    outfile: str,
    signal_length: int,
    conjugate_data: bool = True,
) -> None:
    """Creates a dataset from the data files and saves it to the save_folder

    Args:
        dataset (np.ndarray): numpy array containing the data
        outfile (str): File to save the dataset to
        signal_length (int): Length of the output signal. Signals are truncated if they are longer than signal_length
            and zero-filles otherwise.
        conjugate_data (bool): If True, the data is conjugated (in the time domain) before saving. Default is True.
    """
    dataset = np.array(dataset, dtype=np.complex64)

    if conjugate_data:
        dataset = np.conj(dataset)

    if dataset.shape[1] <= signal_length:
        dataset_ = np.zeros(
            shape=(dataset.shape[0], signal_length, dataset.shape[-1]),
            dtype=np.complex64,
        )
        dataset_[:, : dataset.shape[1], :] = dataset
        dataset = dataset_
    else:
        dataset = dataset[:, :signal_length, :]

    dataset = fftshift(fft(dataset, axis=1), axes=1)

    np.save(outfile, dataset, allow_pickle=True)
    print("Created a dataset with of shape", np.array(dataset).shape, ".")


if __name__ == "__main__":
    """
    Prepares data from .mat files. The configuration is defined by a .json file containing the following keys:
    - dwelltime: Can be ignored here.
    - n_zero_filling: Length of signals after zero-filling
    - n_samples: Can be ignored here
    - datasets: Dictionary containing the following keys:
        - data_root_folder: Root folder containing the subfolders with the .mat files
        - data_subfolders: List of subfolders containing the .mat files
    - basis: Dictionary containing information about preparing the basisset. Not relevant for this script.
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

    # save as numpy arrays
    signal_length = config["n_zero_filling"]
    root_folder = config["datasets"]["data_root_folder"]
    save_folder = root_folder + "/npy_saves"
    Path(save_folder).mkdir(parents=False, exist_ok=True)
    folder_names = config["datasets"]["data_subfolders"]
    conjugate_data = config["datasets"]["conjugate_data"]
    dwelltime = config["dwelltime"]

    all_data = read_mat(root_folder, folder_names)

    training_data = np.vstack(all_data[:-2])

    test_data = np.vstack(all_data[-2:])

    # create and save test and training datasets Tfrom the saved numpy arrays
    print("Saving datasets in folder", save_folder)
    create_dataset(
        dataset=training_data,
        outfile=save_folder + "/training_dataset.npy",
        signal_length=signal_length,
        conjugate_data=conjugate_data,
    )

    create_dataset(
        dataset=test_data,
        outfile=save_folder + "/test_dataset.npy",
        signal_length=signal_length,
        conjugate_data=conjugate_data,
    )
