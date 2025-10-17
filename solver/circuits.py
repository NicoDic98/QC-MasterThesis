from enum import StrEnum
from typing import Any

import h5py
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Gate, ParameterVector
from qiskit.circuit.library import n_local, XXPlusYYGate, RZGate, RYGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp
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
    def __init__(self, num_qubits: int):
        self.num_qubits = int(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        self.full_ansatz = QuantumCircuit(self.num_qubits)

    def __call__(self):
        # Remove barriers here  because otherwise the transpilation might be harmed by to many barriers
        ret = self.full_ansatz.copy()
        ret = RemoveBarriers()(ret)
        return ret

    def num_parameters(self):
        return self.full_ansatz.num_parameters

    def max_num_parameters(self):
        return self.num_parameters()

    def circuit_dict(self) -> dict[str, Any]:
        ansatz_dict = {
            CircuitParameters.NumQubits: self.num_qubits,
        }
        circuit_dict = {
            CircuitParameters.Ansatz: type(self).__name__,
            CircuitParameters.AnsatzOptions: ansatz_dict,
        }
        return circuit_dict

    def save_parameters(self, group: h5py.Group):
        save_dict_as_attribute(group, self.circuit_dict(), CircuitParameters.Circuit)

    def build_full_ansatz_with_save_points(self) -> tuple[QuantumCircuit, int]:
        state_vector_index = 0
        dag = circuit_to_dag(self.full_ansatz)

        for node in dag.op_nodes():
            if node.name == "barrier":
                temp = SaveStatevector(self.num_qubits,
                                       label=CircuitParameters.StateVectorBaseName + f"{state_vector_index}")
                state_vector_index += 1
                dag.substitute_node(node, temp)

        return dag_to_circuit(dag), state_vector_index


class BaseVQEAnsatz(BaseAnsatz):
    def __init__(self, num_qubits: int, num_layers: int):
        super().__init__(num_qubits)
        if num_layers < 1:
            raise ValueError("Number of layers must be positive")
        self.num_layers = int(num_layers)

    def circuit_dict(self):
        circuit_dict = super().circuit_dict()
        # Modify circuit_dict
        circuit_dict[CircuitParameters.NumParameters] = self.num_parameters()

        ansatz_dict = circuit_dict[CircuitParameters.AnsatzOptions]
        # Modify ansatz_dict
        ansatz_dict[CircuitParameters.NumLayers] = self.num_layers
        return circuit_dict


class XXPlusYYRZAnsatz1(BaseVQEAnsatz):
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
        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.x(i)
        self.full_ansatz = self.fixed_ansatz.compose(self.variational_ansatz)


class BaseADAPTVQEAnsatz(BaseAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.operator_pool = []
        self.gate_pool = []

    def max_num_parameters(self):
        """
        Returns the maximum number of parameters in the ansatz. None means unlimited.
        :return: maximum number of parameters
        """
        return None

    def set_ansatz(self, operator_indices: list[int]):
        self.full_ansatz = self.fixed_ansatz.copy()


class XXPlusYYRZAdaptAnsatz1(BaseADAPTVQEAnsatz):
    gate_pool: list[Gate | QuantumCircuit]

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        # self.fixed_ansatz.h(0)
        # self.fixed_ansatz.cx(0,2)
        # self.fixed_ansatz.s(0)
        #
        # self.fixed_ansatz.x(2)
        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.h(i)
        for i in range(self.num_qubits):
            op = SparsePauliOp.from_sparse_list([("Z", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)

        self.gate_pool.append(RZGate(Parameter('A')))

        for i in range(self.num_qubits - 1):
            op = SparsePauliOp.from_sparse_list([("YX", [i, i + 1], -0.5j),
                                                 ("XY", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)

        # print(self.operator_pool)
        qc = QuantumCircuit(2)
        b = Parameter('B')
        qc.z(1)
        qc.s(0)
        qc.h(1)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 1)
        qc.ry(b, 0)
        qc.rz(b, 1)
        qc.cx(0, 1)
        qc.h(0)
        qc.sdg(1)
        qc.sdg(0)
        qc.h(1)
        qc.z(1)
        # qc.draw("mpl")
        # plt.show()

        self.gate_pool.append(qc)

    def set_ansatz(self, operator_indices: list[int]):
        qc = self.fixed_ansatz.copy()
        my_params = ParameterVector("A", len(operator_indices))
        for i, oi in enumerate(operator_indices):
            if oi < self.num_qubits:
                gate = self.gate_pool[0].copy()
                gate.params[0] = my_params[i]
                qc.append(gate, [oi])
            else:
                gate = self.gate_pool[1].to_gate(label="$R_{XY+YX}$",
                                                 parameter_map={self.gate_pool[1].parameters[0]: my_params[i]})
                qc.append(gate, [(oi - self.num_qubits), (oi - self.num_qubits) + 1])
        self.full_ansatz = qc


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
