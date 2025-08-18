import numpy as np
from matplotlib import pyplot as plt

from hamiltonians import FreeWilson2D, HamiltonianType
from exact_diagonalization import EDSolver

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
energies = np.array([EDSolver(FreeWilson2D(2, 2, m, 1)).solve(HamiltonianType.ZeroChargePenalty) for m in masses])
plot_energy_gap(masses, energies)
plot_energies(masses, energies, 2)


