from enum import StrEnum
from typing import Callable

import h5py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, BaseEstimatorV2, PrimitiveResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from h5_interface import adapt_dtype_for_h5
from hamiltonian.base import HamiltonianType
from hamiltonian.free_wilson import BaseHamiltonian
from misc import fill_defaults_in_dict
from solver.base import BaseSolver, SimulatorType
from solver.circuits import XXPlusYYRZAnsatz1


class VQEParameters(StrEnum):
    Backend = "Backend"
    Estimator = "Estimator"
    Optimizer = "Optimizer"
    CircuitParameters = "CircuitParameters"
    DataPrefix = "Data/"
    MetaDataPrefix = "MetaData/"


class VQECostFunction:
    def __init__(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, estimator: BaseEstimatorV2,
                 group: h5py.Group, current_non_singular_index: tuple[int]):
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

        for key, value in pub_result.metadata.items(): # pub specific metadata
            dataset = self.group[VQEParameters.MetaDataPrefix + key]
            if not (h5py.check_string_dtype(dataset.dtype) is None):
                dataset[*self.current_non_singular_index, self.iteration] = str(value)
            else:
                dataset[*self.current_non_singular_index, self.iteration] = value

        for key, value in full_result.metadata.items(): # general metadata
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

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType,
            backend_params=None,
            simulator_type: SimulatorType = SimulatorType.Statevector, estimator_params=None,
            optimizer_params=None):
        local_group, h5_saver, test_hamiltonian = self.initialize_run(parameters_dict_list, hamiltonian_type)
        self.ansatz.save_parameters(local_group)
        test_hamiltonian_op = test_hamiltonian.hamiltonian_op(hamiltonian_type)
        if test_hamiltonian_op.num_qubits != self.ansatz.num_qubits:
            raise ValueError(
                f"Number of qubits does not match ansatz: {test_hamiltonian_op.num_qubits}!={self.ansatz.num_qubits}")

        if simulator_type == SimulatorType.Statevector:
            if backend_params is None:
                backend_params = {}
            else:
                raise UserWarning("Backend parameters are ignored when using Statevector estimator")
            estimator_params_default = {
                "seed": 42
            }
            estimator_params = fill_defaults_in_dict(estimator_params, estimator_params_default)

            # See:
            # https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorEstimator
            estimator = StatevectorEstimator(**estimator_params)

            circuit = self.ansatz()

        elif simulator_type == SimulatorType.Aer:
            backend_params_default = {}
            backend_params = fill_defaults_in_dict(backend_params, backend_params_default)
            estimator_params_default = {
                "seed_transpiler": 42,
                "optimization_level": 3,
                "approximation_degree": 1.0
            }
            estimator_params = fill_defaults_in_dict(estimator_params, estimator_params_default)

            # See:
            # https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html
            backend = AerSimulator()

            # See for more pass manager options:
            # https://quantum.cloud.ibm.com/docs/en/guides/defaults-and-configuration-options
            pm = generate_preset_pass_manager(backend=backend,
                                              **estimator_params)

            estimator = Estimator(mode=backend)

            # this will be the same as qc, if backend = AerSimulator()
            circuit = pm.run(self.ansatz())

        elif simulator_type == SimulatorType.Hardware:
            backend_params_default = {}
            backend_params = fill_defaults_in_dict(backend_params, backend_params_default)
            estimator_params_default = {
                "seed_transpiler": 42,
                "optimization_level": 3,
                "approximation_degree": 1.0
            }
            estimator_params = fill_defaults_in_dict(estimator_params, estimator_params_default)

            raise NotImplementedError

        else:
            raise NotImplementedError

        optimizer_params_default = {
            "method": 'cobyla',
            "bounds": None,
            "constraints": (),
            "tol": None,
            "callback": None,
            "options": {"maxiter": 100,
                        "disp": 2},
            "x0Seed": 42,
        }
        optimizer_params = fill_defaults_in_dict(optimizer_params, optimizer_params_default)

        local_group.attrs[SimulatorType.__name__] = simulator_type.name

        local_group.create_group(VQEParameters.Backend)
        for key, value in backend_params.items():
            local_group[VQEParameters.Backend].attrs[key] = adapt_dtype_for_h5(value)

        local_group.create_group(VQEParameters.Estimator)
        for key, value in estimator_params.items():
            local_group[VQEParameters.Estimator].attrs[key] = adapt_dtype_for_h5(value)

        local_group.create_group(VQEParameters.Optimizer)
        for key, value in optimizer_params.items():
            local_group[VQEParameters.Optimizer].attrs[key] = adapt_dtype_for_h5(value)

        rng = np.random.default_rng(seed=optimizer_params["x0Seed"])
        del optimizer_params["x0Seed"]
        x0 = 2 * np.pi * rng.random(self.ansatz.num_parameters())
        test_hamiltonian_op = test_hamiltonian_op.apply_layout(layout=circuit.layout)
        tes_cost_function = VQECostFunction(circuit, test_hamiltonian_op, estimator, local_group, h5_saver.non_singular_indices_list[0])
        test_full_result = tes_cost_function.evaluate(x0)
        # todo create resizable datasets for everything inside test_pub_result
        """
        todo: save
        per iteration:
            circuit parameters
            pub_result.data["evs"]
            pub_result.data["stds"]
            other pub_result.data.items() (in own subgroup)
            pub_result.metadat.items() and contained in that, circuit_metadata.items() (in own subgroup)
            check for non standard dtype, which should be saved as str: if not(h5py.check_string_dtype(dataset.dtype) is None):
        """
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
