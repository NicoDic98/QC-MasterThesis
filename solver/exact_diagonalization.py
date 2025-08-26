from enum import StrEnum
from typing import Literal, Callable

import h5py
import numpy as np
from scipy.sparse.linalg import eigsh

from hamiltonians import BaseHamiltonian, HamiltonianType
from solver.base import BaseSolver


class EDParameters(StrEnum):
    EigenValues = "EigenValues"
    EigenVectors = "EigenVectors"
    Which = "Which"
    EigenValueAxis = "EigenValueAxis"
    EigenVectorAxis = "EigenVectorAxis"


class ED(BaseSolver):

    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group):
        super().__init__(hamiltonian_factory, save_group)

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType, n_eigv: int = None):
        local_group, h5_saver, test_hamiltonian = self.initialize_run(parameters_dict_list, hamiltonian_type)

        if n_eigv is None:
            n_eigv = test_hamiltonian.size_of_zero_charge_sector() + 2

        eigenvector_dim = test_hamiltonian.size_of_hamiltonian(hamiltonian_type)

        which: Literal["SM", "LM"]
        # 'SM' if using penalty and 'LM' if using projection/full
        if hamiltonian_type == HamiltonianType.Full:
            which = "LM"
        elif hamiltonian_type == HamiltonianType.ZeroChargePenalty:
            which = "SM"
        elif hamiltonian_type == HamiltonianType.ZeroChargeProjection:
            which = "LM"

        local_group.attrs[EDParameters.Which] = which

        h5_saver.create_dataset_with_dim_labels(EDParameters.EigenValues,
                                                [n_eigv],
                                                [EDParameters.EigenValueAxis])
        h5_saver.create_dataset_with_dim_labels(EDParameters.EigenVectors,
                                                [eigenvector_dim, n_eigv],
                                                [EDParameters.EigenVectorAxis, EDParameters.EigenValueAxis],
                                                np.complex128)

        for parameters, non_singular_index in zip(h5_saver.parameters_list_dict, h5_saver.non_singular_indices_list):
            hamiltonian = self.hamiltonian_factory(**parameters)
            print(f"Calculating energies for {hamiltonian}")
            h_operator = hamiltonian.hamiltonian_op(hamiltonian_type)
            h_sparse_matrix = h_operator.to_matrix(sparse=True)
            eigen_values, eigen_vectors = eigsh(h_sparse_matrix, k=n_eigv, which=which)
            local_group[EDParameters.EigenValues][*non_singular_index, :] = eigen_values
            local_group[EDParameters.EigenVectors][*non_singular_index] = eigen_vectors
