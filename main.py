import argparse
from datetime import datetime

import h5py
import numpy as np
from matplotlib import pyplot as plt

from hamiltonians import FreeWilson2D, HamiltonianType
from exact_diagonalization import ED
from misc import pprint_h5

# Define the parser
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--id', action="store", dest='id', default=42)
args = parser.parse_args()

def plot_energy_gap(ms: np.ndarray, sorted_energies: np.ndarray, name_suffix: str = '_2x2'):
    fig, ax = plt.subplots()
    ax.plot(ms, sorted_energies[:,1]-sorted_energies[:,0])
    ax.set(xlabel='Mass', ylabel='Energy gap')
    ax.set_title("Energy gap for r=1")
    plt.savefig(f'energy_gap{name_suffix}.pdf')

def plot_energies(ms: np.ndarray, sorted_energies: np.ndarray, n_plot: int, name_suffix: str = '_2x2'):
    fig, ax = plt.subplots()
    ax.plot(ms, sorted_energies[:,:n_plot])
    ax.set(xlabel='Mass', ylabel='Energies')
    ax.set_title("Energies for r=1")
    plt.savefig(f'energies{name_suffix}.pdf')


masses = np.linspace(-6, 2, 501)
print(FreeWilson2D.__name__)
with h5py.File(f"{datetime.now().strftime('%Y-%m-%U')}-{args.id}.hdf5", "w") as f:
    pprint_h5(f)
    my_ed = ED(FreeWilson2D.build_hamiltonian, f)
    my_ed.run({"n_x": [2],
               "n_y": [2],
               "mass": masses.tolist(),
               "r": [1., 3.14]},
              HamiltonianType.ZeroChargePenalty)
    pprint_h5(f)
# plot_energy_gap(masses, energies)
# plot_energies(masses, energies, 2)
# attr: System-parameters, ProcessId


