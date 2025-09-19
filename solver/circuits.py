from enum import StrEnum

import h5py
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import n_local, XXPlusYYGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer.library import SaveStatevector

from h5_interface import save_dict_as_attribute, load_attribute_as_dict


class CircuitParameters(StrEnum):
    Circuit = "Circuit"
    Ansatz = "Ansatz"
    NumParameters = "NumParameters"
    AnsatzOptions = "AnsatzOptions"
    NumQubits = "num_qubits"
    NumLayers = "num_layers"
    StateVectorBaseName = "psi_"


class BaseAnsatz:
    def __init__(self, num_qubits: int, num_layers: int):
        self.num_qubits = int(num_qubits)
        if num_layers < 1:
            raise ValueError("Number of layers must be positive")
        self.num_layers = int(num_layers)
        self.full_ansatz = QuantumCircuit(self.num_qubits)

    def __call__(self):
        # Remove barriers here  because otherwise the transpilation might be harmed by to many barriers
        ret = self.full_ansatz.copy()
        ret = RemoveBarriers()(ret)
        return ret

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


    def build_full_ansatz_with_save_points(self)-> tuple[QuantumCircuit, int]:
        state_vector_index = 0
        dag = circuit_to_dag(self.full_ansatz)

        for node in dag.op_nodes():
            if node.name == "barrier":
                temp = SaveStatevector(self.num_qubits,
                                       label=CircuitParameters.StateVectorBaseName + f"{state_vector_index}")
                state_vector_index += 1
                dag.substitute_node(node, temp)

        return dag_to_circuit(dag), state_vector_index


class XXPlusYYRZAnsatz1(BaseAnsatz):
    def __init__(self, num_qubits: int, num_layers: int):
        """
        Ansatz based on https://dx.doi.org/10.1103/PhysRevD.109.114508
        Factors of 2 in entangling blocks are there to match the gate definitions in this paper.
        With this the parameter range yielding unique gates is [0, 2*pi]
        :param num_qubits:
        :param num_layers:
        """
        super().__init__(num_qubits, num_layers)
        initial_block = n_local(num_qubits=self.num_qubits,
                                rotation_blocks=[],
                                entanglement_blocks=XXPlusYYGate(2 * Parameter('A'), 0),
                                entanglement="pairwise",
                                insert_barriers=True,
                                skip_final_rotation_layer=True,
                                reps=1,
                                parameter_prefix="phi")
        following_blocks = n_local(num_qubits=self.num_qubits,
                                   rotation_blocks="rz",
                                   entanglement_blocks=XXPlusYYGate(2 * Parameter('A'), 0),
                                   entanglement="pairwise",
                                   insert_barriers=True,
                                   reps=self.num_layers - 1)  # this adds just rotations if layers - 1 = 0

        self.variational_ansatz = initial_block.compose(following_blocks)
        self.variational_ansatz.barrier()
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


def rebuild_ansatz(group: h5py.Group) -> BaseAnsatz:
    circuit_dict = load_attribute_as_dict(group[CircuitParameters.Circuit])
    for my_class in inheritors(BaseAnsatz):
        if my_class.__name__ == circuit_dict[CircuitParameters.Ansatz]:
            return my_class(**circuit_dict[CircuitParameters.AnsatzOptions])
    raise NotImplementedError(f"No matching ansatz found, for {circuit_dict[CircuitParameters.Ansatz]}")
