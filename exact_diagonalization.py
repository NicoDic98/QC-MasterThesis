import itertools
from typing import Literal, Callable

import h5py
import numpy as np
from datetime import datetime

from hamiltonians import BaseHamiltonian, HamiltonianType
from scipy.sparse.linalg import eigsh


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

        non_singular_keys = []
        for key, value in parameters_dict_list.items():
            if len(value) == 1:
                local_group[key] = value[0]
            else:
                non_singular_keys.append(key)

        temp_key_list = parameters_dict_list.keys()
        paired_parameters_list = list(itertools.product(*[parameters_dict_list[key] for key in temp_key_list]))
        parameters_list_dict = [dict(zip(temp_key_list, paired_parameters)) for paired_parameters in
                                paired_parameters_list]

        temp = self.hamiltonian_factory(**(parameters_list_dict[0]))
        if n_eigv is None:
            n_eigv = temp.size_of_zero_charge_sector() + 2

        local_group.attrs['SystemName'] = temp.__name__
        local_group.attrs['HamiltonianType'] = hamiltonian_type.name
        local_group.attrs['SolverName'] = self.__name__
        local_group.attrs['Last-Modified'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        if hamiltonian_type == HamiltonianType.Full:
            self.which = "SM"
        elif hamiltonian_type == HamiltonianType.ZeroChargePenalty:
            self.which = "SM"
        elif hamiltonian_type == HamiltonianType.ZeroChargeProjection:
            self.which = "LM"
        local_group.attrs['ED-Which'] = self.which

        print(parameters_list_dict)
        for parameters in parameters_list_dict:
            hamiltonian = self.hamiltonian_factory(**parameters)
            self.solve(hamiltonian, hamiltonian_type, n_eigv)

    def solve(self, hamiltonian: BaseHamiltonian, hamiltonian_type: HamiltonianType, n_eigv: int = 2):
        print(f"Calculating energies for {hamiltonian}")
        if hamiltonian_type == HamiltonianType.Full:
            h_operator = hamiltonian.full_hamiltonian()
        elif hamiltonian_type == HamiltonianType.ZeroChargePenalty:
            h_operator = hamiltonian.zero_charge_penalized_hamiltonian()
        elif hamiltonian_type == HamiltonianType.ZeroChargeProjection:
            h_operator = hamiltonian.zero_charge_projected_hamiltonian()
        h_sparse_matrix = h_operator.to_matrix(sparse=True)
        eigen_values, eigen_vectors = eigsh(h_sparse_matrix, k=n_eigv,
                                            which=self.which)  # 'SM' if using penalty and 'LM' if using projection
        eigen_values: np.ndarray
        eigen_vectors: np.ndarray
        eigen_values.sort()
        return eigen_values, eigen_vectors
