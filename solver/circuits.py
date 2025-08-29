import inspect
from enum import StrEnum

import h5py
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import n_local, XXPlusYYGate

from h5_interface import save_dict_as_attribute, load_attribute_as_dict


class CircuitParameters(StrEnum):
    Circuit = "Circuit"
    Ansatz = "Ansatz"
    NumParameters = "NumParameters"
    AnsatzOptions = "AnsatzOptions"
    NumQubits = "num_qubits"
    NumLayers = "num_layers"


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

    def circuit_dict(self):
        ansatz_dict = {
            CircuitParameters.NumQubits: self.num_qubits,
            CircuitParameters.NumLayers: self.num_layers,
        }
        circuit_dict = {
            CircuitParameters.Ansatz: type(self).__name__,
            CircuitParameters.NumParameters: self.num_parameters(),
            CircuitParameters.AnsatzOptions: ansatz_dict,
        }
        return circuit_dict

    def save_parameters(self, group: h5py.Group):
        save_dict_as_attribute(group, self.circuit_dict(), CircuitParameters.Circuit)


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
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.x(i)
        self.full_ansatz = self.fixed_ansatz.compose(self.variational_ansatz)

    def circuit_dict(self):
        circuit_dict = super().circuit_dict()
        ansatz_dict = circuit_dict[CircuitParameters.AnsatzOptions]
        # Modify ansatz_dict
        return circuit_dict


def inheritors(my_class):
    subclasses = []
    to_be_searched = [my_class]
    while to_be_searched:
        parent = to_be_searched.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.append(child)
                to_be_searched.append(child)
    return subclasses


def rebuild_ansatz(group: h5py.Group):
    circuit_dict = load_attribute_as_dict(group[CircuitParameters.Circuit])
    for my_class in inheritors(BaseAnsatz):
        if my_class.__name__ == circuit_dict[CircuitParameters.Ansatz]:
            return my_class(**circuit_dict[CircuitParameters.AnsatzOptions])
    raise NotImplementedError(f"No matching ansatz found, for {circuit_dict[CircuitParameters.Ansatz]}")
