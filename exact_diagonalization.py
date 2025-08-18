from typing import Literal

import numpy as np

from hamiltonians import BaseHamiltonian, HamiltonianType
from scipy.sparse.linalg import eigsh


class EDSolver:
    def __init__(self, hamiltonian: BaseHamiltonian):
        self.hamiltonian = hamiltonian

    def solve(self, hamiltonian_type: HamiltonianType = HamiltonianType.Full, n_eigv: int = None):
        print(f"Calculating energies for {self.hamiltonian}")
        if hamiltonian_type == HamiltonianType.Full:
            h_operator = self.hamiltonian.full_hamiltonian()
            which = "SM"
        elif hamiltonian_type == HamiltonianType.ZeroChargePenalty:
            h_operator = self.hamiltonian.zero_charge_penalized_hamiltonian()
            which = "SM"
        elif hamiltonian_type == HamiltonianType.ZeroChargeProjection:
            h_operator = self.hamiltonian.zero_charge_projected_hamiltonian()
            which = "LM"
        else:
            raise NotImplementedError
        which: Literal["SM", "LM"]
        if n_eigv is None:
            n_eigv = self.hamiltonian.size_of_zero_charge_sector() + 2
        h_sparse_matrix = h_operator.to_matrix(sparse=True)
        eigen_values, eigen_vectors = eigsh(h_sparse_matrix, k=n_eigv,
                                            which=which)  # 'SM' if using penalty and 'LM' if using projection
        eigen_values: np.ndarray
        eigen_vectors: np.ndarray
        eigen_values.sort()
        return eigen_values
