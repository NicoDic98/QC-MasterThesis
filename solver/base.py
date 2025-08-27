from datetime import datetime
from enum import Enum, auto, StrEnum
from typing import Callable

import h5py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp

from h5_interface import H5Saver
from hamiltonian.free_wilson import BaseHamiltonian
from hamiltonian.base import HamiltonianType
from misc import GlobalParameters


class EstimatorType(Enum):
    Statevector = auto()
    Aer = auto()
    Hardware = auto()

class QCPrefix(StrEnum):
    Backend = "Backend/"
    Estimator = "Estimator/"
    Optimizer = "Optimizer/"
    Data = "Data/"
    MetaData = "MetaData/"


def cost_func(params: np.ndarray, ansatz:QuantumCircuit, hamiltonian: SparsePauliOp, estimator: BaseEstimatorV2):
    pub = (ansatz, hamiltonian, [params])
    # noinspection PyTypeChecker
    job = estimator.run(pubs=[pub])
    pub_result = job.result()[0]
    energy = pub_result.data.evs[0]


    return energy


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
        self.initialize_run(parameters_dict_list, hamiltonian_type)
