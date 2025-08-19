from datetime import datetime
from fileinput import filename
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from exact_diagonalization import EDParameters
from hamiltonians import HamiltonianType, HamiltonianParameters


def plot_energy_gap_ed(group: h5py.Group, parameters: dict[str, int]):
    print("Plotting energy gap for:")
    for key, value in group.attrs.items():
        print(f"\t{key}: {value}")

    output_filename = "results/plots/"
    output_filename += str(group.name).split("/")[-1]
    Path(output_filename).mkdir(parents=True, exist_ok=True)

    output_filename += "/"
    output_filename += "energy_gap_ed"

    for key, value in parameters.items():
        print(f"\t{key}: {group[key][value]}")
        output_filename = f"{output_filename}_{key}={group[key][value]:.2f}"

    output_filename += ".pdf"

    selected_indices = []
    masses = np.array([])
    for dim in group[EDParameters.EigenValues].dims:
        if dim.label in parameters.keys():
            selected_indices.append(parameters[dim.label])
        elif dim.label == HamiltonianParameters.Mass:
            masses = np.array(dim[dim.label])
            selected_indices.append(list(range(len(masses))))
        elif dim.label == EDParameters.EigenValueAxis: #This is always at the end
            pass
        else:
            raise ValueError(f"You needed to specify an index for dimension {dim.label}")
    if len(masses) == 0:
        raise ValueError("No dimension is labeled as mass dimension")
    energies = group[EDParameters.EigenValues][*selected_indices]

    energy_gap = energies[:, 1] - energies[:, 0]

    fig, ax = plt.subplots()
    ax.plot(masses, energy_gap)
    ax.set(xlabel='Mass', ylabel='Energy gap')
    ax.set_title("Energy gap for r=1")
    plt.savefig(output_filename)

parent_folder = "results/data/"
with h5py.File(f"{parent_folder}{datetime.now().strftime('%Y-%m-%U')}-{42}.hdf5", "r") as f:
    for name in f:
        print(name)
    plot_energy_gap_ed(f[list(f.keys())[-1]], {
        HamiltonianParameters.WilsonParameter: 0
    })
