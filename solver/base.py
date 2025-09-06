from datetime import datetime
from enum import Enum, auto, StrEnum
from typing import Callable

import h5py
import qiskit
import qiskit_ibm_runtime
import qiskit_aer

from h5_interface import H5Saver, save_dict_as_attribute
from hamiltonian.base import HamiltonianType
from hamiltonian.free_wilson import BaseHamiltonian


class GlobalParameters(StrEnum):
    SystemName = "SystemName"
    SolverName = "SolverName"
    LastModified = "Last-Modified"
    ProcessId = "ProcessId"
    VersionInfo = "VersionInfo"
    QiskitVersion = "QiskitVersion"
    QiskitIBMRuntimeVersion = "QiskitIBMRuntimeVersion"
    QiskitAerVersion = "QiskitAerVersion"


class SimulatorType(Enum):
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
        local_group = self.save_group.create_group(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                                                   + f"-{self.save_group.file.attrs[GlobalParameters.ProcessId]}")
        h5_saver = H5Saver(local_group, parameters_dict_list)

        test_hamiltonian = self.hamiltonian_factory(**(h5_saver.parameters_list_dict[0]))

        local_group.attrs[GlobalParameters.SystemName] = type(test_hamiltonian).__name__
        local_group.attrs[GlobalParameters.SolverName] = type(self).__name__
        local_group.attrs[GlobalParameters.ProcessId] = local_group.file.attrs[GlobalParameters.ProcessId]
        local_group.attrs[GlobalParameters.LastModified] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        local_group.attrs[HamiltonianType.__name__] = hamiltonian_type.name
        version_dict = {
            GlobalParameters.QiskitVersion: qiskit.version.get_version_info(),
            GlobalParameters.QiskitIBMRuntimeVersion: qiskit_ibm_runtime.version.get_version_info(),
            GlobalParameters.QiskitAerVersion: qiskit_aer.version.get_version_info(),
        }
        save_dict_as_attribute(local_group, version_dict, GlobalParameters.VersionInfo)
        return local_group, h5_saver, test_hamiltonian

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType):
        self.initialize_run(parameters_dict_list, hamiltonian_type)
