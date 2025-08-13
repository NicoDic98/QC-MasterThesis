import numpy as np
from matplotlib import pyplot as plt

from hamiltonians import FreeWilson2D
from scipy.sparse.linalg import eigsh

def calc_energies(m: float, n_eigv = 10):
    print(f'Calculating energies for mass {m:2.3f}')
    free_wilson = FreeWilson2D(2, 2, m, 1)
    h_operator = free_wilson.hamiltonian().simplify()
    h_sparse_matrix = h_operator.to_matrix(sparse=True)
    eigen_values, eigen_vectors = eigsh(h_sparse_matrix, k=n_eigv)
    eigen_values : np.ndarray
    eigen_vectors : np.ndarray
    eigen_values.sort()
    return eigen_values

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
energies = np.array([calc_energies(m, 10) for m in masses])
print(masses.shape, energies.shape)
plot_energy_gap(masses, energies)
plot_energies(masses, energies, 2)



