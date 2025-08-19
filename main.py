import argparse
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt

from hamiltonians import FreeWilson2D, HamiltonianType, HamiltonianParameters
from exact_diagonalization import ED
from misc import pprint_h5, GlobalParameters, data_folder, default_id

# Define the parser
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--id', action="store", dest='id', default=default_id)
args = parser.parse_args()


def plot_energies(ms: np.ndarray, sorted_energies: np.ndarray, n_plot: int, name_suffix: str = '_2x2'):
    fig, ax = plt.subplots()
    ax.plot(ms, sorted_energies[:, :n_plot])
    ax.set(xlabel='Mass', ylabel='Energies')
    ax.set_title("Energies for r=1")
    plt.savefig(f'energies{name_suffix}.pdf')


masses = np.linspace(-6, 2, 51)
Path(data_folder).mkdir(parents=True, exist_ok=True)
with h5py.File(f"{data_folder}{datetime.now().strftime('%Y-%m-%U')}-{args.id}.hdf5", "w") as f:
    pprint_h5(f)
    f.attrs[GlobalParameters.ProcessId] = args.id
    my_ed = ED(FreeWilson2D.build_hamiltonian, f)
    my_ed.run({HamiltonianParameters.XExtend: [2],
               HamiltonianParameters.YExtend: [2],
               HamiltonianParameters.Mass: masses.tolist(),
               HamiltonianParameters.WilsonParameter: [1., 3.14]},
              HamiltonianType.ZeroChargePenalty)
    pprint_h5(f)
# attr: ProcessId
