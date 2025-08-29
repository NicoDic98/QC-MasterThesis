from dataclasses import asdict
from enum import StrEnum
from typing import Callable, Any

import h5py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, BaseEstimatorV2, PrimitiveResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime.options.utils import UnsetType

from h5_interface import save_dict_as_attribute
from hamiltonian.base import HamiltonianType
from hamiltonian.free_wilson import BaseHamiltonian
from misc import fill_defaults_in_dict
from solver.base import BaseSolver, SimulatorType
from solver.circuits import XXPlusYYRZAnsatz1


class VQEParameters(StrEnum):
    DataPrefix = "Data/"
    MetaDataPrefix = "MetaData/"
    SimulatorOptions = "SimulatorOptions"
    PresetPassManagerOptions = "PresetPassManagerOptions"
    EstimatorOptions = "EstimatorOptions"
    OptimizerOptions = "OptimizerOptions"
    IterationAxis = "IterationAxis"
    CircuitParameterAxis = "CircuitParameterAxis"
    CircuitParameters = "CircuitParameters"


class VQECostFunction:
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

    def __call__(self, params: np.ndarray) -> float:
        full_result = self.evaluate(params)
        pub_result = full_result[0]

        for key, value in pub_result.data.items():
            dataset = self.group[VQEParameters.DataPrefix + key]
            if not (h5py.check_string_dtype(dataset.dtype) is None):
                dataset[*self.current_non_singular_index, self.iteration] = str(value[0])  # only one pub
            else:
                dataset[*self.current_non_singular_index, self.iteration] = value[0]  # only one pub

        for key, value in pub_result.metadata.items():  # pub specific metadata
            dataset = self.group[VQEParameters.MetaDataPrefix + key]
            if not (h5py.check_string_dtype(dataset.dtype) is None):
                dataset[*self.current_non_singular_index, self.iteration] = str(value)
            else:
                dataset[*self.current_non_singular_index, self.iteration] = value

        for key, value in full_result.metadata.items():  # general metadata
            dataset = self.group[VQEParameters.MetaDataPrefix + key]
            if not (h5py.check_string_dtype(dataset.dtype) is None):
                dataset[*self.current_non_singular_index, self.iteration] = str(value)
            else:
                dataset[*self.current_non_singular_index, self.iteration] = value

        energy = pub_result.data["evs"][0]
        return energy


class VQE(BaseSolver):
    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group,
                 num_qubits: int,
                 num_layers: int, ):
        super().__init__(hamiltonian_factory, save_group)
        self.ansatz = XXPlusYYRZAnsatz1(num_qubits, num_layers)

    def setup_estimator(self, simulator_type: SimulatorType,
                        simulator_options: dict[str, Any],
                        preset_pass_manager_options: dict[str, Any],
                        estimator_options: EstimatorOptions) -> tuple[BaseEstimatorV2, QuantumCircuit]:
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
            estimator_options_default.default_precision = 0.0
            estimator_options_default.simulator.seed_simulator = 42

            preset_pass_manager_options_default = {
                "seed_transpiler": 42,
                "optimization_level": 3,
                "approximation_degree": 1.0
            }

            if simulator_options:
                raise UserWarning("Simulator options are ignored when using Statevector estimator")

            preset_pass_manager_options = fill_defaults_in_dict(preset_pass_manager_options,
                                                                preset_pass_manager_options_default)

            if isinstance(estimator_options.default_precision, UnsetType):
                estimator_options.default_precision = estimator_options_default.default_precision
            if isinstance(estimator_options.simulator.seed_simulator, UnsetType):
                estimator_options.simulator.seed_simulator = estimator_options_default.simulator.seed_simulator

            pm = generate_preset_pass_manager(**preset_pass_manager_options)

            estimator = StatevectorEstimator(default_precision=estimator_options.default_precision,
                                             seed=estimator_options.simulator.seed_simulator)
            circuit = pm.run(self.ansatz())

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

            simulator_options = fill_defaults_in_dict(simulator_options,
                                                      simulator_options_defaults)

            preset_pass_manager_options = fill_defaults_in_dict(preset_pass_manager_options,
                                                                preset_pass_manager_options_default)

            if isinstance(estimator_options.seed_estimator, UnsetType):
                estimator_options.seed_estimator = estimator_options_default.seed_estimator
            if isinstance(estimator_options.simulator.seed_simulator, UnsetType):
                estimator_options.simulator.seed_simulator = estimator_options_default.simulator.seed_simulator

            backend = AerSimulator(**simulator_options)

            pm = generate_preset_pass_manager(backend=backend,
                                              **preset_pass_manager_options)

            estimator = Estimator(mode=backend, options=estimator_options)
            circuit = pm.run(self.ansatz())

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

            preset_pass_manager_options = fill_defaults_in_dict(preset_pass_manager_options,
                                                                preset_pass_manager_options_default)

            if isinstance(estimator_options.seed_estimator, UnsetType):
                estimator_options.seed_estimator = estimator_options_default.seed_estimator
            if isinstance(estimator_options.simulator.seed_simulator, UnsetType):
                estimator_options.simulator.seed_simulator = estimator_options_default.simulator.seed_simulator

            # todo: choose actual hardware backend

            raise NotImplementedError

        else:
            raise NotImplementedError

        return estimator, circuit

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType,
            simulator_type: SimulatorType = SimulatorType.Statevector,
            simulator_options: dict[str, Any] = None,
            preset_pass_manager_options: dict[str, Any] = None,
            estimator_options: EstimatorOptions = None,
            optimizer_options: dict[str, Any] = None):
        local_group, h5_saver, test_hamiltonian = self.initialize_run(parameters_dict_list, hamiltonian_type)
        self.ansatz.save_parameters(local_group)
        test_hamiltonian_op = test_hamiltonian.hamiltonian_op(hamiltonian_type)
        if test_hamiltonian_op.num_qubits != self.ansatz.num_qubits:
            raise ValueError(
                f"Number of qubits does not match ansatz: {test_hamiltonian_op.num_qubits}!={self.ansatz.num_qubits}")

        if simulator_options is None:
            simulator_options = {}
        if preset_pass_manager_options is None:
            preset_pass_manager_options = {}
        if estimator_options is None:
            estimator_options = EstimatorOptions()
        if optimizer_options is None:
            optimizer_options = {}

        estimator, circuit = self.setup_estimator(simulator_type, simulator_options,
                                                  preset_pass_manager_options, estimator_options)

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
        optimizer_options = fill_defaults_in_dict(optimizer_options, optimizer_options_default)

        local_group.attrs[SimulatorType.__name__] = simulator_type.name

        save_dict_as_attribute(local_group, simulator_options, VQEParameters.SimulatorOptions)
        save_dict_as_attribute(local_group, preset_pass_manager_options, VQEParameters.PresetPassManagerOptions)
        # noinspection PyDataclass,PyTypeChecker
        save_dict_as_attribute(local_group, asdict(estimator_options), VQEParameters.EstimatorOptions)
        save_dict_as_attribute(local_group, optimizer_options, VQEParameters.OptimizerOptions)

        rng = np.random.default_rng(seed=optimizer_options["x0Seed"])
        del optimizer_options["x0Seed"]
        x0 = 2 * np.pi * rng.random(self.ansatz.num_parameters())
        test_hamiltonian_op = test_hamiltonian_op.apply_layout(layout=circuit.layout)
        test_cost_function = VQECostFunction(circuit, test_hamiltonian_op, estimator, local_group,
                                             h5_saver.non_singular_indices_list[0])
        test_full_result = test_cost_function.evaluate(x0)
        test_pub_result = test_full_result[0]

        for key, value in test_pub_result.data.items():
            for i, operator_name_suffix in enumerate(["/hamiltonian"]):
                h5_saver.create_dataset_with_dim_labels(VQEParameters.DataPrefix + key + operator_name_suffix,
                                                        (10,), [VQEParameters.IterationAxis],
                                                        type(value[i]), (None,))

        for key, value in test_pub_result.metadata.items():  # pub specific metadata
            h5_saver.create_dataset_with_dim_labels(VQEParameters.MetaDataPrefix + key,
                                                    (10,), [VQEParameters.IterationAxis],
                                                    type(value), (None,))

        for key, value in test_full_result.metadata.items():  # general metadata
            h5_saver.create_dataset_with_dim_labels(VQEParameters.MetaDataPrefix + key,
                                                    (10,), [VQEParameters.IterationAxis],
                                                    type(value), (None,))

        h5_saver.create_dataset_with_dim_labels(VQEParameters.CircuitParameters,
                                                (10, self.ansatz.num_parameters()),
                                                [VQEParameters.IterationAxis, VQEParameters.CircuitParameterAxis],
                                                x0.dtype,
                                                (None, self.ansatz.num_parameters()))
        """
        (1,)
        Data:
        evs [1.90917969]
        stds [0.08189292]
        MetaData:
        target_precision 0.015625
        shots 4096
        circuit_metadata {}
        """
