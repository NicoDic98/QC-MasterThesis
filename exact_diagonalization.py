import itertools
from enum import StrEnum
from typing import Literal, Callable

import h5py
import numpy as np
from datetime import datetime

from hamiltonians import BaseHamiltonian, HamiltonianType
from scipy.sparse.linalg import eigsh

from misc import GlobalParameters


class EDParameters(StrEnum):
    EigenValues = "EigenValues"
    EigenVectors = "EigenVectors"
    Which = "Which"
    EigenValueAxis = "EigenValueAxis"
    EigenVectorAxis = "EigenVectorAxis"


class ED:
    which: Literal["SM", "LM"]

    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group):
        self.hamiltonian_factory = hamiltonian_factory
        self.save_group = save_group
        self.which = "SM"

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType, n_eigv: int = None):
        local_group = self.save_group.create_group(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        parameters_list_dict = [dict(zip(parameters_dict_list.keys(), paired_parameters)) for paired_parameters in
                                itertools.product(*[parameters_dict_list[key] for key in parameters_dict_list.keys()])]
        temp = self.hamiltonian_factory(**(parameters_list_dict[0]))

        if n_eigv is None:
            n_eigv = temp.size_of_zero_charge_sector() + 2

        if hamiltonian_type == HamiltonianType.Full:
            self.which = "SM"  # 'SM' if using penalty and 'LM' if using projection
            eigenvector_dim = temp.size_of_full_hamiltonian()
        elif hamiltonian_type == HamiltonianType.ZeroChargePenalty:
            self.which = "SM"
            eigenvector_dim = temp.size_of_zero_charge_penalized_hamiltonian()
        elif hamiltonian_type == HamiltonianType.ZeroChargeProjection:
            self.which = "LM"
            eigenvector_dim = temp.size_of_zero_charge_projected_hamiltonian()

        local_group.attrs[GlobalParameters.SystemName] = type(temp).__name__
        local_group.attrs[GlobalParameters.SolverName] = type(self).__name__
        local_group.attrs[GlobalParameters.LastModified] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        local_group.attrs[HamiltonianType.__name__] = hamiltonian_type.name
        local_group.attrs[EDParameters.Which] = self.which

        non_singular_keys = []
        for key, value in parameters_dict_list.items():
            if len(value) == 1:
                local_group.attrs[key] = value[0]
            elif len(value) > 1:
                local_group[key] = value
                local_group[key].make_scale(key)
                non_singular_keys.append(key)
            else:
                raise NotImplementedError

        non_singular_indices_list = list(
            itertools.product(*[range(len(parameters_dict_list[key])) for key in non_singular_keys]))

        eigen_value_dims = [len(parameters_dict_list[key]) for key in non_singular_keys]
        eigen_value_dims.append(n_eigv)
        local_group.create_dataset(EDParameters.EigenValues, eigen_value_dims, dtype=np.float64)
        for i, key in enumerate(non_singular_keys):
            local_group[EDParameters.EigenValues].dims[i].attach_scale(local_group[key])
            local_group[EDParameters.EigenValues].dims[i].label = key
        local_group[EDParameters.EigenValues].dims[len(non_singular_keys)].label = EDParameters.EigenValueAxis

        eigen_vector_dims = [len(parameters_dict_list[key]) for key in non_singular_keys]
        eigen_vector_dims.append(eigenvector_dim)
        eigen_vector_dims.append(n_eigv)
        local_group.create_dataset(EDParameters.EigenVectors, eigen_vector_dims, dtype=np.complex128)
        for i, key in enumerate(non_singular_keys):
            local_group[EDParameters.EigenVectors].dims[i].attach_scale(local_group[key])
            local_group[EDParameters.EigenVectors].dims[i].label = key
        local_group[EDParameters.EigenVectors].dims[len(non_singular_keys)].label = EDParameters.EigenVectorAxis
        local_group[EDParameters.EigenVectors].dims[len(non_singular_keys) + 1].label = EDParameters.EigenValueAxis

        for parameters, non_singular_index in zip(parameters_list_dict, non_singular_indices_list):
            hamiltonian = self.hamiltonian_factory(**parameters)
            eigen_values, eigen_vectors = self.solve(hamiltonian, hamiltonian_type, n_eigv)
            local_group[EDParameters.EigenValues][*non_singular_index, :] = eigen_values
            local_group[EDParameters.EigenVectors][*non_singular_index] = eigen_vectors
        print(np.array(local_group[EDParameters.EigenValues]))
        print(np.array(local_group[EDParameters.EigenVectors]))

    def solve(self, hamiltonian: BaseHamiltonian, hamiltonian_type: HamiltonianType, n_eigv: int = 2):
        print(f"Calculating energies for {hamiltonian}")
        if hamiltonian_type == HamiltonianType.Full:
            h_operator = hamiltonian.full_hamiltonian()
        elif hamiltonian_type == HamiltonianType.ZeroChargePenalty:
            h_operator = hamiltonian.zero_charge_penalized_hamiltonian()
        elif hamiltonian_type == HamiltonianType.ZeroChargeProjection:
            h_operator = hamiltonian.zero_charge_projected_hamiltonian()
        h_sparse_matrix = h_operator.to_matrix(sparse=True)
        eigen_values, eigen_vectors = eigsh(h_sparse_matrix, k=n_eigv, which=self.which)
        eigen_values: np.ndarray
        eigen_vectors: np.ndarray
        eigen_values.sort()
        return eigen_values, eigen_vectors
