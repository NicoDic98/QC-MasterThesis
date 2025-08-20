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

    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group):
        self.hamiltonian_factory = hamiltonian_factory
        self.save_group = save_group

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType, n_eigv: int = None):
        local_group = self.save_group.create_group(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        parameters_list_dict = [dict(zip(parameters_dict_list.keys(), paired_parameters)) for paired_parameters in
                                itertools.product(*[parameters_dict_list[key] for key in parameters_dict_list.keys()])]
        temp = self.hamiltonian_factory(**(parameters_list_dict[0]))

        if n_eigv is None:
            n_eigv = temp.size_of_zero_charge_sector() + 2

        eigenvector_dim = temp.size_of_hamiltonian(hamiltonian_type)

        which: Literal["SM", "LM"]
        # 'SM' if using penalty and 'LM' if using projection/full
        if hamiltonian_type == HamiltonianType.Full:
            which = "LM"
        elif hamiltonian_type == HamiltonianType.ZeroChargePenalty:
            which = "SM"
        elif hamiltonian_type == HamiltonianType.ZeroChargeProjection:
            which = "LM"

        local_group.attrs[GlobalParameters.SystemName] = type(temp).__name__
        local_group.attrs[GlobalParameters.SolverName] = type(self).__name__
        local_group.attrs[GlobalParameters.ProcessId] = local_group.file.attrs[GlobalParameters.ProcessId]
        local_group.attrs[GlobalParameters.LastModified] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        local_group.attrs[HamiltonianType.__name__] = hamiltonian_type.name
        local_group.attrs[EDParameters.Which] = which

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

        parameter_dims = [len(parameters_dict_list[key]) for key in non_singular_keys]

        local_group.create_dataset(EDParameters.EigenValues, parameter_dims + [n_eigv],
                                   dtype=np.float64)
        for i, key in enumerate(non_singular_keys):
            local_group[EDParameters.EigenValues].dims[i].attach_scale(local_group[key])
            local_group[EDParameters.EigenValues].dims[i].label = key
        local_group[EDParameters.EigenValues].dims[len(non_singular_keys)].label = EDParameters.EigenValueAxis

        local_group.create_dataset(EDParameters.EigenVectors, parameter_dims + [eigenvector_dim, n_eigv],
                                   dtype=np.complex128)
        for i, key in enumerate(non_singular_keys):
            local_group[EDParameters.EigenVectors].dims[i].attach_scale(local_group[key])
            local_group[EDParameters.EigenVectors].dims[i].label = key
        local_group[EDParameters.EigenVectors].dims[len(non_singular_keys)].label = EDParameters.EigenVectorAxis
        local_group[EDParameters.EigenVectors].dims[len(non_singular_keys) + 1].label = EDParameters.EigenValueAxis

        for parameters, non_singular_index in zip(parameters_list_dict, non_singular_indices_list):
            hamiltonian = self.hamiltonian_factory(**parameters)
            print(f"Calculating energies for {hamiltonian}")
            h_operator = hamiltonian.hamiltonian_op(hamiltonian_type)
            h_sparse_matrix = h_operator.to_matrix(sparse=True)
            eigen_values, eigen_vectors = eigsh(h_sparse_matrix, k=n_eigv, which=which)
            local_group[EDParameters.EigenValues][*non_singular_index, :] = eigen_values
            local_group[EDParameters.EigenVectors][*non_singular_index] = eigen_vectors
