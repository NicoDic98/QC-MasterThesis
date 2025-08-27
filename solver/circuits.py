from enum import StrEnum

import h5py
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import n_local, XXPlusYYGate


class CircuitParameters(StrEnum):
    Ansatz = "Ansatz"
    NumQubits = "NumQubits"
    NumLayers = "NumLayers"
    NumParameters = "NumParameters"


class BaseAnsatz:
    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = num_qubits
        if num_layers < 1:
            raise ValueError("Number of layers must be positive")
        self.num_layers = num_layers
        self.full_ansatz = QuantumCircuit(self.num_qubits)

    def __call__(self):
        return self.full_ansatz.copy()

    def num_parameters(self):
        return self.full_ansatz.num_parameters

    def save_parameters(self, group: h5py.Group):
        group.attrs[CircuitParameters.Ansatz] = type(self).__name__
        group.attrs[CircuitParameters.NumQubits] = self.num_qubits
        group.attrs[CircuitParameters.NumLayers] = self.num_layers
        group.attrs[CircuitParameters.NumParameters] = self.num_parameters()



class XXPlusYYRZAnsatz1(BaseAnsatz):
    def __init__(self, num_qubits: int, num_layers: int):
        super().__init__(num_qubits, num_layers)
        initial_block = n_local(num_qubits=self.num_qubits,
                                rotation_blocks=[],
                                entanglement_blocks=XXPlusYYGate(Parameter('A'), 0),
                                entanglement="pairwise",
                                insert_barriers=True,
                                skip_final_rotation_layer=True,
                                reps=1,
                                parameter_prefix="phi")
        following_blocks = n_local(num_qubits=self.num_qubits,
                                   rotation_blocks="rz",
                                   entanglement_blocks=XXPlusYYGate(Parameter('A'), 0),
                                   entanglement="pairwise",
                                   insert_barriers=True,
                                   reps=self.num_layers - 1)  # this adds just rotations if layers - 1 = 0

        self.variational_ansatz = initial_block.compose(following_blocks)
        # todo: add initialization for zero charge sector
        self.full_ansatz = self.variational_ansatz

    def save_parameters(self, group: h5py.Group):
        super().save_parameters(group)