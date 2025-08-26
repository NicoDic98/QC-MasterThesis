from typing import Callable

import h5py
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from hamiltonian.base import HamiltonianType
from hamiltonian.free_wilson import BaseHamiltonian
from solver.base import BaseSolver
from solver.circuits import XXPlusYYRZAnsatz1


class VQE(BaseSolver):
    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group,
                 num_qubits: int,
                 num_layers: int, ):
        super().__init__(hamiltonian_factory, save_group)
        self.ansatz = XXPlusYYRZAnsatz1(num_qubits, num_layers)

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType, n_steps: int = 100):
        local_group, h5_saver, test_hamiltonian = self.initialize_run(parameters_dict_list, hamiltonian_type)
        self.ansatz.save_parameters(local_group)
        test_hamiltonian_op = test_hamiltonian.hamiltonian_op(hamiltonian_type)
        if test_hamiltonian_op.num_qubits != self.ansatz.num_qubits:
            raise ValueError(
                f"Number of qubits does not match ansatz: {test_hamiltonian_op.num_qubits}!={self.ansatz.num_qubits}")

        qc = self.ansatz()

        # See:
        # https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html
        backend = AerSimulator()

        # See for more pass manager options:
        # https://quantum.cloud.ibm.com/docs/en/guides/defaults-and-configuration-options
        pm = generate_preset_pass_manager(backend=backend,
                                          optimization_level=3,
                                          approximation_degree=1.00,
                                          seed_transpiler=42)

        # this will be the same as qc, if backend = AerSimulator()
        isa_circuit = pm.run(qc)

        hamiltonian_isa = test_hamiltonian_op.apply_layout(layout=isa_circuit.layout)
