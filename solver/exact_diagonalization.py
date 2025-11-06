from enum import StrEnum
from typing import Literal, Callable

import h5py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from hamiltonian.free_wilson import BaseHamiltonian
from hamiltonian.base import HamiltonianType
from solver.base import BaseSolver


class EDParameters(StrEnum):
    EigenValues = "EigenValues"
    EigenVectors = "EigenVectors"
    Which = "Which"
    CompressMatrix = "CompressMatrix"
    EigenValueAxis = "EigenValueAxis"
    EigenVectorAxis = "EigenVectorAxis"


class ED(BaseSolver):

    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group):
        super().__init__(hamiltonian_factory, save_group)

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType, n_eigv: int = None,
            compress_matrix=False):
        local_group, h5_saver, test_hamiltonian = self.initialize_run(parameters_dict_list, hamiltonian_type)

        if n_eigv is None:
            n_eigv = test_hamiltonian.size_of_zero_charge_sector() + 2
            if compress_matrix:
                n_eigv -= 4

        if compress_matrix:
            eigenvector_dim = test_hamiltonian.size_of_zero_charge_sector()
        else:
            eigenvector_dim = test_hamiltonian.size_of_hamiltonian(hamiltonian_type)

        which: Literal["SM", "LM"]
        # 'SM' if using penalty and 'LM' if using projection/full
        if hamiltonian_type == HamiltonianType.Full:
            which = "LM"
        elif hamiltonian_type == HamiltonianType.ZeroChargePenalty:
            which = "SM"
        elif hamiltonian_type == HamiltonianType.ZeroChargeProjection:
            which = "LM"

        if compress_matrix:
            which = "LM"

        local_group.attrs[EDParameters.Which] = which
        local_group.attrs[EDParameters.CompressMatrix] = compress_matrix

        h5_saver.create_dataset_with_dim_labels(EDParameters.EigenValues,
                                                [n_eigv],
                                                [EDParameters.EigenValueAxis],
                                                np.float64)
        if not compress_matrix:
            h5_saver.create_dataset_with_dim_labels(EDParameters.EigenVectors,
                                                    [eigenvector_dim, n_eigv],
                                                    [EDParameters.EigenVectorAxis, EDParameters.EigenValueAxis],
                                                    np.complex128)

        for parameters, non_singular_index in zip(h5_saver.parameters_list_dict, h5_saver.non_singular_indices_list):
            hamiltonian = self.hamiltonian_factory(**parameters)
            print(f"Calculating energies for {hamiltonian}")
            h_operator = hamiltonian.hamiltonian_op(hamiltonian_type)
            # noinspection PyTypeChecker
            h_sparse_matrix = h_operator.to_matrix(sparse=True)
            h_sparse_matrix: csr_matrix
            h_sparse_matrix.eliminate_zeros()

            if compress_matrix:
                temp = h_sparse_matrix.tocoo()

                final_dat = []
                row_ind = []
                col_ind = []
                zero_charge_size = hamiltonian.size_of_zero_charge_sector()
                my_n = h_operator.num_qubits

                my_ind_map = []
                id_temp = 0
                for i in range(temp.shape[0]):
                    bin_i = np.binary_repr(i, my_n)
                    i_ones = bin_i.count("1")
                    if i_ones == (my_n // 2):
                        my_ind_map.append(id_temp)
                        id_temp += 1
                    else:
                        my_ind_map.append(-1)
                # print("Index map done")

                for i, j, dat in zip(temp.row, temp.col, temp.data):
                    bin_i = np.binary_repr(i, my_n)
                    bin_j = np.binary_repr(j, my_n)
                    i_ones = bin_i.count("1")
                    j_ones = bin_j.count("1")
                    if (i_ones == (my_n // 2)) and (j_ones == (my_n // 2)):
                        row_ind.append(my_ind_map[i])
                        col_ind.append(my_ind_map[j])
                        final_dat.append(dat)
                # print("Data map done")

                h_sparse_matrix = csr_matrix((final_dat, (row_ind, col_ind)),
                                             shape=(zero_charge_size, zero_charge_size))
                # print("Operator done")
            temp = eigsh(h_sparse_matrix, k=n_eigv, which=which,
                                                return_eigenvectors=(not compress_matrix))
            if compress_matrix:
                eigen_values = temp
            else:
                eigen_values, eigen_vectors = temp
            local_group[EDParameters.EigenValues][*non_singular_index, :] = eigen_values
            if not compress_matrix:
                local_group[EDParameters.EigenVectors][*non_singular_index] = eigen_vectors
