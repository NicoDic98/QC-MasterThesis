import numpy as np
from matplotlib import pyplot as plt

from hamiltonians import FreeWilson2D
from scipy.sparse.linalg import eigsh

def calc_energies(m: float, n_eigv = 10):
    print(f'Calculating energies for mass {m:2.3f}')
    free_wilson = FreeWilson2D(2, 2, m, 1)
    h_operator = free_wilson.zero_charge_projected_hamiltonian().simplify()
    h_sparse_matrix = h_operator.to_matrix(sparse=True)
    eigen_values, eigen_vectors = eigsh(h_sparse_matrix, k=n_eigv, which='LM') #'SM' if using penalty and 'LM' if using projection
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


# masses = np.linspace(-6, 2, 501)
# test = FreeWilson2D(2, 2, masses[0], 1)
#
# zero_c_size = test.size_of_zero_charge_sector()
# print(f"Number of pauli strings:\n"
#       f"Full hamiltonian: {len(test.full_hamiltonian().to_list())}\n"
#       f"Zero-charge penalized hamiltonian: {len(test.zero_charge_penalized_hamiltonian().to_list())}\n"
#       f"Zero-charge projected hamiltonian: {len(test.zero_charge_projected_hamiltonian().to_list())}\n"
#       f"Zero-charge projector: {len(test.zero_charge_projector().to_list())}\n"
#       f"Zero-charge penalty: {len(test.zero_charge_penalty_term().to_list())}")
#
# print(f"Number of non-commuting pauli strings:\n"
#       f"Full hamiltonian: {len(test.full_hamiltonian().group_commuting())}\n"
#       f"Zero-charge penalized hamiltonian: {len(test.zero_charge_penalized_hamiltonian().group_commuting())}\n"
#       f"Zero-charge projected hamiltonian: {len(test.zero_charge_projected_hamiltonian().group_commuting())}\n"
#       f"Zero-charge projector: {len(test.zero_charge_projector().group_commuting())}\n"
#       f"Zero-charge penalty: {len(test.zero_charge_penalty_term().group_commuting())}")

# print(f"Size of the zero charge sector: {zero_c_size}")
# energies = np.array([calc_energies(m, zero_c_size+2) for m in masses])
# plot_energy_gap(masses, energies)
# plot_energies(masses, energies, 2)

# pen_operator = test.zero_charge_projector().simplify()
# pen_matrix = pen_operator.to_matrix()
# pen_list = pen_matrix.diagonal()
# zeros = np.argwhere(np.isclose(pen_list, np.zeros_like(pen_list)))[:,0]
# print(pen_matrix.min(), pen_matrix.max())
# print(zeros)
# print(len(zeros))
# counter = np.zeros(9)
# for i in zeros:
#     bin_i = np.binary_repr(i)
#     n_ones = bin_i.count("1")
#     counter[n_ones] += 1
# print(counter)

