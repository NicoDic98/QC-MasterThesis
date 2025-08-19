# import argparse
#
# parser = argparse.ArgumentParser(description='Short sample app')
# parser.add_argument('--id', action="store", dest='id', default=40)
# args = parser.parse_args()
from datetime import datetime

import h5py

from misc import data_folder, default_id


def print_info(f: str):
    file = f.split("/")[-1].split(".")[0]
    dataset_name = f.split("/")[-2]
    temp = file.split("_")

    parameters = {}
    for parameter in temp[1:]:
        key, value = parameter.split("=")
        parameters[key] = float(value)

    h5_filename = f"{data_folder}{datetime.strptime(dataset_name, '%Y-%m-%d_%H-%M-%S').strftime('%Y-%m-%U')}-{default_id}.hdf5"

    print(f"{temp[0]} for (Info for '{f}' from '{h5_filename}'):")
    with h5py.File(h5_filename, "r") as f:
        group = f[dataset_name]
        for key, value in group.attrs.items():
            print(f"\t{key}: {value}")

        for key, value in parameters.items():
            print(f"\t{key}: {value}")


print_info("results/plots/2025-08-19_16-54-37/EnergyGapED_r=1.00.pdf")
