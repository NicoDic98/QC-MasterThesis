from datetime import datetime

import h5py
from matplotlib import pyplot as plt

from exact_diagonalization import EDParameters
from hamiltonians import HamiltonianType
from misc import pprint_h5


def plot_energy_gap_ed(group: h5py.Group):
    print("Plotting energy gap for:")
    pprint_h5(group)
    filename = str(group.name).split("/")[-1]
    print([dim.label for dim in group[EDParameters.EigenValues].dims])
    fig, ax = plt.subplots()
    # ax.plot(ms, sorted_energies[:, 1] - sorted_energies[:, 0])
    ax.set(xlabel='Mass', ylabel='Energy gap')
    ax.set_title("Energy gap for r=1")
    plt.savefig(f'{filename}_energy_gap.pdf')

with h5py.File(f"{datetime.now().strftime('%Y-%m-%U')}-{42}.hdf5", "r") as f:
    plot_energy_gap_ed(f["2025-08-19_14-29-11"])
print(HamiltonianType.__name__)