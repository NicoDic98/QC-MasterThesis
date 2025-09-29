from dataclasses import asdict
from datetime import datetime
from enum import Enum, auto, StrEnum
from typing import Callable, Any

import h5py
import numpy as np
import qiskit
import qiskit_ibm_runtime
import qiskit_aer
from qiskit import generate_preset_pass_manager, QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator, PrimitiveResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.options.utils import UnsetType

from h5_interface import H5Saver, save_dict_as_attribute
from hamiltonian.base import HamiltonianType
from hamiltonian.free_wilson import BaseHamiltonian
from labels import VQEParameters
from misc import fill_defaults_in_dict
from solver.circuits import BaseAnsatz


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


class BaseCostFunction:
    def __init__(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, estimator: BaseEstimatorV2,
                 group: h5py.Group, current_non_singular_index: tuple[
                int]):  # todo add functionality to accept arbitrary additional operators
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.estimator = estimator
        self.group = group
        self.current_non_singular_index = current_non_singular_index
        self.iteration = 0

    def evaluate(self, params: np.ndarray) -> PrimitiveResult:
        pub = (self.ansatz, self.hamiltonian, [params])
        # noinspection PyTypeChecker
        job = self.estimator.run(pubs=[pub])
        return job.result()


class BaseVQE(BaseSolver):
    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group,
                 ansatz: BaseAnsatz):
        super().__init__(hamiltonian_factory, save_group)
        self.ansatz = ansatz
        self.cost_function = BaseCostFunction

    def setup_estimator(self, simulator_type: SimulatorType,
                        simulator_options: dict[str, Any],
                        preset_pass_manager_options: dict[str, Any],
                        estimator_options: EstimatorOptions) -> tuple[BaseEstimatorV2, PassManager]:
        """

        :param simulator_type:
        :param simulator_options:
        https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html#aersimulator
        :param preset_pass_manager_options:
        https://quantum.cloud.ibm.com/docs/en/guides/defaults-and-configuration-options
        :param estimator_options:
        https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-estimator-options
        https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorEstimator
        :return:
        """
        if simulator_type == SimulatorType.Statevector:
            estimator_options_default = EstimatorOptions()
            # estimator_options_default.default_precision = 0.0
            estimator_options_default.simulator.seed_simulator = 42

            preset_pass_manager_options_default = {
                "seed_transpiler": 42,
                "optimization_level": 3,
                "approximation_degree": 1.0
            }

            if simulator_options:
                raise UserWarning("Simulator options are ignored when using Statevector estimator")

            fill_defaults_in_dict(preset_pass_manager_options, preset_pass_manager_options_default)

            if isinstance(estimator_options.default_precision, UnsetType):
                precision = 0.0
            else:
                precision = estimator_options.default_precision
            if isinstance(estimator_options.simulator.seed_simulator, UnsetType):
                estimator_options.simulator.seed_simulator = estimator_options_default.simulator.seed_simulator

            pm = generate_preset_pass_manager(**preset_pass_manager_options)

            estimator = StatevectorEstimator(default_precision=precision,
                                             seed=estimator_options.simulator.seed_simulator)

        elif simulator_type == SimulatorType.Aer:
            simulator_options_defaults = {

            }

            preset_pass_manager_options_default = {
                "seed_transpiler": 42,
                "optimization_level": 3,
                "approximation_degree": 1.0
            }

            estimator_options_default = EstimatorOptions()
            estimator_options_default.seed_estimator = 42
            estimator_options_default.simulator.seed_simulator = 42

            fill_defaults_in_dict(simulator_options, simulator_options_defaults)

            fill_defaults_in_dict(preset_pass_manager_options, preset_pass_manager_options_default)

            if isinstance(estimator_options.seed_estimator, UnsetType):
                estimator_options.seed_estimator = estimator_options_default.seed_estimator
            if isinstance(estimator_options.simulator.seed_simulator, UnsetType):
                estimator_options.simulator.seed_simulator = estimator_options_default.simulator.seed_simulator

            backend = AerSimulator(**simulator_options)

            pm = generate_preset_pass_manager(backend=backend,
                                              **preset_pass_manager_options)

            estimator = Estimator(mode=backend, options=estimator_options)

        elif simulator_type == SimulatorType.Hardware:
            if simulator_options:
                raise UserWarning("Simulator options are ignored when using Hardware estimator")

            preset_pass_manager_options_default = {
                "seed_transpiler": 42,
                "optimization_level": 3,
                "approximation_degree": 1.0
            }

            estimator_options_default = EstimatorOptions()
            estimator_options_default.seed_estimator = 42
            estimator_options_default.simulator.seed_simulator = 42

            fill_defaults_in_dict(preset_pass_manager_options, preset_pass_manager_options_default)

            if isinstance(estimator_options.seed_estimator, UnsetType):
                estimator_options.seed_estimator = estimator_options_default.seed_estimator
            if isinstance(estimator_options.simulator.seed_simulator, UnsetType):
                estimator_options.simulator.seed_simulator = estimator_options_default.simulator.seed_simulator

            # todo: choose actual hardware backend

            raise NotImplementedError

        else:
            raise NotImplementedError

        return estimator, pm

    def initialize_run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType,
                       simulator_type: SimulatorType = SimulatorType.Statevector,
                       simulator_options: dict[str, Any] = None,
                       preset_pass_manager_options: dict[str, Any] = None,
                       estimator_options: EstimatorOptions = None,
                       optimizer_options: dict[str, Any] = None):
        local_group, h5_saver, test_hamiltonian = super().initialize_run(parameters_dict_list, hamiltonian_type)

        test_hamiltonian_op = test_hamiltonian.hamiltonian_op(hamiltonian_type)
        if test_hamiltonian_op.num_qubits != self.ansatz.num_qubits:
            raise ValueError(
                f"Number of qubits does not match ansatz: {test_hamiltonian_op.num_qubits}!={self.ansatz.num_qubits}")
        self.ansatz.save_parameters(local_group)

        if simulator_options is None:
            simulator_options = {}
        if preset_pass_manager_options is None:
            preset_pass_manager_options = {}
        if estimator_options is None:
            estimator_options = EstimatorOptions()
        if optimizer_options is None:
            optimizer_options = {}

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        optimizer_options_default = {
            "method": 'cobyla',
            "bounds": None,
            "constraints": (),
            "tol": None,
            "callback": None,
            "options": {"maxiter": 100,
                        "disp": 2},
            "x0Seed": 42,
        }
        fill_defaults_in_dict(optimizer_options, optimizer_options_default)

        local_group.attrs[SimulatorType.__name__] = simulator_type.name

        save_dict_as_attribute(local_group, simulator_options, VQEParameters.SimulatorOptions)
        save_dict_as_attribute(local_group, preset_pass_manager_options, VQEParameters.PresetPassManagerOptions)
        # noinspection PyDataclass,PyTypeChecker
        save_dict_as_attribute(local_group, asdict(estimator_options), VQEParameters.EstimatorOptions)
        save_dict_as_attribute(local_group, optimizer_options, VQEParameters.OptimizerOptions)
        estimator, pm = self.setup_estimator(simulator_type, simulator_options,
                                             preset_pass_manager_options, estimator_options)
        circuit = pm.run(self.ansatz())

        rng = np.random.default_rng(seed=optimizer_options["x0Seed"])
        del optimizer_options["x0Seed"]
        x0 = 2 * np.pi * rng.random(self.ansatz.num_parameters())
        test_hamiltonian_op = test_hamiltonian_op.apply_layout(layout=circuit.layout)
        test_cost_function = self.cost_function(circuit, test_hamiltonian_op, estimator, local_group,
                                                h5_saver.non_singular_indices_list[0])
        test_full_result = test_cost_function.evaluate(x0)
        test_pub_result = test_full_result[0]

        for key, value in test_pub_result.data.items():
            for i, operator_name_suffix in enumerate([VQEParameters.HamiltonianSuffix]):
                h5_saver.create_dataset_with_dim_labels(VQEParameters.DataPrefix + key + operator_name_suffix,
                                                        [10], [VQEParameters.IterationAxis],
                                                        type(value[i]), [None])

        for key, value in test_pub_result.metadata.items():  # pub specific metadata
            h5_saver.create_dataset_with_dim_labels(VQEParameters.MetaDataPrefix + key,
                                                    [10], [VQEParameters.IterationAxis],
                                                    type(value), [None])

        for key, value in test_full_result.metadata.items():  # general metadata
            h5_saver.create_dataset_with_dim_labels(VQEParameters.MetaDataPrefix + key,
                                                    [10], [VQEParameters.IterationAxis],
                                                    type(value), [None])

        h5_saver.create_dataset_with_dim_labels(VQEParameters.CircuitParameters,
                                                [10, self.ansatz.num_parameters()],
                                                [VQEParameters.IterationAxis, VQEParameters.CircuitParameterAxis],
                                                x0.dtype,
                                                [None, self.ansatz.max_num_parameters()])
        return local_group, h5_saver, estimator, pm, x0
