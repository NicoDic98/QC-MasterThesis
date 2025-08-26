from datetime import datetime
from enum import Enum, auto
from typing import Callable

import h5py

from h5_interface import H5Saver
from hamiltonian.free_wilson import BaseHamiltonian
from hamiltonian.base import HamiltonianType
from misc import GlobalParameters


class EstimatorType(Enum):
    Statevector = auto()
    Aer = auto()
    Hardware = auto()


class BaseSolver:
    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group):
        self.hamiltonian_factory = hamiltonian_factory
        self.save_group = save_group

    def initialize_run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType):
        local_group = self.save_group.create_group(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        h5_saver = H5Saver(local_group, parameters_dict_list)

        test_hamiltonian = self.hamiltonian_factory(**(h5_saver.parameters_list_dict[0]))

        local_group.attrs[GlobalParameters.SystemName] = type(test_hamiltonian).__name__
        local_group.attrs[GlobalParameters.SolverName] = type(self).__name__
        local_group.attrs[GlobalParameters.ProcessId] = local_group.file.attrs[GlobalParameters.ProcessId]
        local_group.attrs[GlobalParameters.LastModified] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        local_group.attrs[HamiltonianType.__name__] = hamiltonian_type.name
        return local_group, h5_saver, test_hamiltonian

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType):
        local_group, h5_saver, test_hamiltonian = self.initialize_run(parameters_dict_list, hamiltonian_type)
