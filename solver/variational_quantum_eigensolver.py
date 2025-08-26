from typing import Callable

import h5py
import matplotlib.pyplot as plt
from qiskit.circuit import Parameter
from qiskit.circuit.library import n_local, XXPlusYYGate

from hamiltonian.free_wilson import BaseHamiltonian
from hamiltonian.base import HamiltonianType
from solver.base import BaseSolver


class VQE(BaseSolver):
    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group,
                 layers: int,
                 num_qubits: int, ):
        super().__init__(hamiltonian_factory, save_group)
        if layers < 1:
            raise ValueError("Number of layers must be positive")
        initial_block = n_local(num_qubits=num_qubits,
                                rotation_blocks=[],
                                entanglement_blocks=XXPlusYYGate(Parameter('A'), 0),
                                entanglement="pairwise",
                                insert_barriers=True,
                                skip_final_rotation_layer=True,
                                reps=1,
                                parameter_prefix="phi")
        following_blocks = n_local(num_qubits=num_qubits,
                                   rotation_blocks="rz",
                                   entanglement_blocks=XXPlusYYGate(Parameter('A'), 0),
                                   entanglement="pairwise",
                                   insert_barriers=True,
                                   reps=layers - 1)  # this adds just rotations if layers - 1 = 0
        self.ansatz = initial_block.compose(following_blocks)
        self.ansatz.draw("mpl")
        plt.show()

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType, n_steps: int = 100):
        local_group, h5_saver, test_hamiltonian = self.initialize_run(parameters_dict_list, hamiltonian_type)
        test_hamiltonian_op = test_hamiltonian.hamiltonian_op(hamiltonian_type)
        if test_hamiltonian_op.num_qubits != self.ansatz.num_qubits:
            raise ValueError(
                f"Number of qubits does not match ansatz: {test_hamiltonian_op.num_qubits}!={self.ansatz.num_qubits}")
