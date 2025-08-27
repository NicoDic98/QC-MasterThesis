from typing import Callable, Any

import h5py
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from hamiltonian.base import HamiltonianType
from hamiltonian.free_wilson import BaseHamiltonian
from solver.base import BaseSolver, EstimatorType
from solver.circuits import XXPlusYYRZAnsatz1


class VQE(BaseSolver):
    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group,
                 num_qubits: int,
                 num_layers: int, ):
        super().__init__(hamiltonian_factory, save_group)
        self.ansatz = XXPlusYYRZAnsatz1(num_qubits, num_layers)

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType,
            estimator_type: EstimatorType = EstimatorType.Statevector, estimator_params=None,
            n_steps: int = 100):
        local_group, h5_saver, test_hamiltonian = self.initialize_run(parameters_dict_list, hamiltonian_type)
        self.ansatz.save_parameters(local_group)
        test_hamiltonian_op = test_hamiltonian.hamiltonian_op(hamiltonian_type)
        if test_hamiltonian_op.num_qubits != self.ansatz.num_qubits:
            raise ValueError(
                f"Number of qubits does not match ansatz: {test_hamiltonian_op.num_qubits}!={self.ansatz.num_qubits}")

        qc = self.ansatz()
        if estimator_type == EstimatorType.Statevector:
            if estimator_params is None:
                estimator_params = {"seed": 42}
            # See:
            # https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorEstimator
            estimator = StatevectorEstimator(**estimator_params)
        elif estimator_type == EstimatorType.Aer:
            if estimator_params is None:
                estimator_params = {"seed_transpiler": 42,
                                    "optimization_level": 3,
                                    "approximation_degree": 1.0}
            # See:
            # https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html
            backend = AerSimulator()
            # See for more pass manager options:
            # https://quantum.cloud.ibm.com/docs/en/guides/defaults-and-configuration-options
            pm = generate_preset_pass_manager(backend=backend,
                                              **estimator_params)
            # this will be the same as qc, if backend = AerSimulator()
            qc = pm.run(qc)
            hamiltonian_isa = test_hamiltonian_op.apply_layout(layout=qc.layout)
            estimator = Estimator(mode=backend)
        elif estimator_type == EstimatorType.Hardware:
            if estimator_params is None:
                estimator_params = {"seed_transpiler": 42,
                                    "optimization_level": 3,
                                    "approximation_degree": 1.0}
            raise NotImplementedError
        else:
            raise NotImplementedError

        local_group.attrs[EstimatorType.__name__] = estimator_type.name
        for key, value in estimator_params.items():
            local_group.attrs[key] = value
        """
        todo: save
        backend parameters
        estimator parameters
        per iteration:
            circuit parameters
            pub_result.data["evs"]
            pub_result.data["stds"]
            other pub_result.data.items()
            pub_result.metadat.items() and contained in that, circuit_metadata.items()
        """
