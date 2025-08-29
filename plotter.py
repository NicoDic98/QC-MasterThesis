import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from combine_data import combine_data
from solver.base import GlobalParameters
from solver.exact_diagonalization import EDParameters, ED
from h5_interface import H5Loader
from hamiltonian.base import HamiltonianParameters
from misc import plots_folder, data_folder
from solver.variational_quantum_eigensolver import VQE, VQEParameters


def custom_json(obj):
    if isinstance(obj, h5py.Group):
        temp = dict(obj)
        temp["Attributes"] = dict(obj.attrs)
        return temp
    if isinstance(obj, h5py.Dataset):
        temp = {
            "Dataset": str(obj),
            "Attributes": dict(obj.attrs),
        }
        return temp
    return str(obj)


def create_filename(group: h5py.Group, parameters: dict[str, int], plot_name: str):
    output_filename = plots_folder
    output_filename += str(group.name).split("/")[-1]
    Path(output_filename).mkdir(parents=True, exist_ok=True)

    output_filename += "/"
    output_filename += plot_name

    for key, value in parameters.items():
        output_filename = f"{output_filename}_{key}={group[key][value]:.2f}"

    with open(output_filename + ".json", "w") as finfo:
        json.dump(group, finfo, sort_keys=True, indent=4, default=custom_json)
        # todo: update to use pprint recursion with only group attributes + dataset attributes of the dataset of interest

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


def plot_energies(group: h5py.Group, parameters: dict[str, int], n_plot: int):
    solver = group.attrs[GlobalParameters.SolverName]
    output_filename = create_filename(group, parameters, f"Energies{solver}")
    if solver == ED.__name__:
        h5_loader = H5Loader(group, EDParameters.EigenValues)
        energies, dep = h5_loader.retrieve_dependency(parameters, [HamiltonianParameters.Mass])
        masses = dep[0]
        energies.sort(-1)
        energies = energies[:, :n_plot]
    elif solver == VQE.__name__:
        h5_loader = H5Loader(group, VQEParameters.Hamiltonian)
        energies, dep = h5_loader.retrieve_dependency(parameters, [HamiltonianParameters.Mass])
        h5_loader = H5Loader(group, VQEParameters.NIterations)
        n_iterations, dep = h5_loader.retrieve_dependency(parameters, [HamiltonianParameters.Mass])
        print(n_iterations)
        masses = dep[0]
        energies = energies[np.arange(len(masses)), n_iterations]
    else:
        raise ValueError(f"Unknown solver {solver}")

    fig, ax = plt.subplots()
    ax.plot(masses, energies)
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
    }
    # plot_energy_gap_ed(my_group, my_parameters)
    plot_energies(my_group, my_parameters, n_plot=2)
