from datetime import datetime
from pathlib import Path

import h5py
from matplotlib import pyplot as plt

from combine_data import combine_data
from solver.exact_diagonalization import EDParameters
from h5_interface import H5Loader
from hamiltonians import HamiltonianParameters
from misc import plots_folder, data_folder


def create_filename(group: h5py.Group, parameters: dict[str, int], plot_name: str):
    output_filename = plots_folder
    output_filename += str(group.name).split("/")[-1]
    Path(output_filename).mkdir(parents=True, exist_ok=True)

    output_filename += "/"
    output_filename += plot_name

    for key, value in parameters.items():
        output_filename = f"{output_filename}_{key}={group[key][value]:.2f}"

    with open(output_filename + ".info", "w") as f:
        message = f"{plot_name} for {group.name}:"
        print(message, file=f)
        print(message)
        for key, value in group.attrs.items():
            message = f"\t{key}: {value}"
            print(message, file=f)
            print(message)
        for key, value in parameters.items():
            message = f"\t{key}: {group[key][value]}"
            print(message, file=f)
            print(message)

    return output_filename + ".png"


def plot_energy_gap_ed(group: h5py.Group, parameters: dict[str, int]):
    output_filename = create_filename(group, parameters, "EnergyGapED")
    h5_loader = H5Loader(group, EDParameters.EigenValues)
    energies, dep = h5_loader.retrieve_dependency(parameters, [HamiltonianParameters.Mass])

    masses = dep[0]
    energies.sort(-1)
    energy_gap = energies[:, 1] - energies[:, 0]

    fig, ax = plt.subplots()
    ax.plot(masses, energy_gap)
    ax.set(xlabel='Mass', ylabel='Energy gap')
    ax.set_title("Energy gap")
    plt.savefig(output_filename)


def plot_energies_ed(group: h5py.Group, parameters: dict[str, int], n_plot: int):
    output_filename = create_filename(group, parameters, "EnergiesED")
    h5_loader = H5Loader(group, EDParameters.EigenValues)
    energies, dep = h5_loader.retrieve_dependency(parameters, [HamiltonianParameters.Mass])

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
        print(name)
    my_group = f[name]
    my_parameters = {
        HamiltonianParameters.WilsonParameter: 0
    }
    plot_energy_gap_ed(my_group, my_parameters)
    plot_energies_ed(my_group, my_parameters, n_plot=2)
