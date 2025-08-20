from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from combine_data import combine_data
from exact_diagonalization import EDParameters
from hamiltonians import HamiltonianParameters
from misc import plots_folder, data_folder


def retrieve_dataset_dependency(dataset: h5py.Dataset, parameters: dict[str, int], dependency_names: list[str]):
    selected_indices = []
    dependencies = [np.array([])] * len(dependency_names)
    for dim in dataset.dims:
        if dim.label in parameters.keys():
            selected_indices.append(parameters[dim.label])
        elif dim.label in dependency_names:
            for i, dependency_name in enumerate(dependency_names):
                if dependency_name == dim.label:
                    dependencies[i] = np.array(dim[dim.label])
                    selected_indices.append(list(range(len(dependencies[i]))))
                    break
        elif dim.label == EDParameters.EigenValueAxis:  # This is always at the end
            pass
        else:
            raise ValueError(f"You needed to specify an index for dimension {dim.label}")

    for dependency_name, dependency in zip(dependency_names, dependencies):
        if len(dependency) == 0:
            raise ValueError(f"No dimension is labeled as {dependency_name} dimension")
    values = dataset[*selected_indices]
    return values, dependencies


def create_filename(group: h5py.Group, parameters: dict[str, int], plot_name: str):
    output_filename = plots_folder
    output_filename += str(group.name).split("/")[-1]
    Path(output_filename).mkdir(parents=True, exist_ok=True)

    output_filename += "/"
    output_filename += plot_name

    for key, value in parameters.items():
        print(f"\t{key}: {group[key][value]}")
        output_filename = f"{output_filename}_{key}={group[key][value]:.2f}"

    output_filename += ".pdf"
    return output_filename


def plot_energy_gap_ed(group: h5py.Group, parameters: dict[str, int]):
    print(f"Plotting energy gap for {group.name}:")
    for key, value in group.attrs.items():
        print(f"\t{key}: {value}")

    energies, dep = retrieve_dataset_dependency(group[EDParameters.EigenValues], parameters,
                                                [HamiltonianParameters.Mass])

    output_filename = create_filename(group, parameters, "EnergyGapED")

    masses = dep[0]
    energies.sort(-1)
    energy_gap = energies[:, 1] - energies[:, 0]

    fig, ax = plt.subplots()
    ax.plot(masses, energy_gap)
    ax.set(xlabel='Mass', ylabel='Energy gap')
    ax.set_title("Energy gap")
    plt.savefig(output_filename)


def plot_energies_ed(group: h5py.Group, parameters: dict[str, int], n_plot: int):
    print(f"Plotting energies for {group.name}:")
    for key, value in group.attrs.items():
        print(f"\t{key}: {value}")

    energies, dep = retrieve_dataset_dependency(group[EDParameters.EigenValues], parameters,
                                                [HamiltonianParameters.Mass])

    output_filename = create_filename(group, parameters, "EnergiesED")

    masses = dep[0]
    energies.sort(-1)

    fig, ax = plt.subplots()
    ax.plot(masses, energies[:, :n_plot])
    ax.set(xlabel='Mass', ylabel='Energies')
    ax.set_title("Energies")
    plt.savefig(output_filename)


combine_data()
h5_file = f"{data_folder}{datetime.now().strftime('%Y-%m-%U')}.hdf5"
with h5py.File(h5_file, "r") as f:
    for name in f:
        my_group = f[name]
        my_parameters = {
            HamiltonianParameters.WilsonParameter: 0
        }
        plot_energy_gap_ed(my_group, my_parameters)
        plot_energies_ed(my_group, my_parameters, n_plot=2)
