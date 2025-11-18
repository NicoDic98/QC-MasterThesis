from enum import StrEnum
from typing import Any, Callable

import h5py
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Gate, ParameterVector
from qiskit.circuit.library import n_local, XXPlusYYGate, RZGate, RYGate, RXGate, XXMinusYYGate
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
    operator_pool: list[SparsePauliOp]
    gate_pool: list[Callable[[Parameter], Gate]]
    operator_gate_map: list[tuple[int, list[int]]]

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.operator_pool = []
        self.gate_pool = []
        self.operator_gate_map = []

    def max_num_parameters(self):
        """
        Returns the maximum number of parameters in the ansatz. None means unlimited.
        :return: maximum number of parameters
        """
        return None

    def set_ansatz(self, operator_indices: list[int]):
        qc = self.fixed_ansatz.copy()
        my_params = ParameterVector("A", len(operator_indices))
        last_gate_on_qubit = [-1] * self.num_qubits
        for i, oi in enumerate(operator_indices):
            gi, qbits = self.operator_gate_map[oi]
            ok = False
            for qbit in qbits:
                if last_gate_on_qubit[qbit] != oi:
                    ok = True
            if not ok:
                return 1
            gate = self.gate_pool[gi](my_params[i])
            qc.append(gate, qbits)
            for qbit in qbits:
                last_gate_on_qubit[qbit] = oi
        self.full_ansatz = qc
        return 0

    def get_operator_info(self, operator_index: int, all_qbits=False):
        """
        :param all_qbits: Whether to return all qbits
        :param operator_index: Operator index
        :return: Gate index, First/All qbit(s), Gate name
        """
        gi, qbits = self.operator_gate_map[operator_index]
        gate = self.gate_pool[gi](Parameter("A"))
        if not all_qbits:
            qbits = qbits[0]
        return gi, qbits, gate.label


def build_r_yx_xy(b: Parameter, plus=True):
    if plus:
        name = "$R_{YX+XY}$"
    else:
        name = "$R_{YX-XY}$"
    qc = QuantumCircuit(2, name=name)
    qc.z(1)
    qc.s(0)
    qc.h(1)
    qc.h(0)
    qc.s(1)
    qc.cx(0, 1)
    if plus:
        qc.ry(b, 0)
    else:
        qc.ry(-b, 0)
    qc.rz(b, 1)
    qc.cx(0, 1)
    qc.h(0)
    qc.sdg(1)
    qc.sdg(0)
    qc.h(1)
    qc.z(1)
    return qc.to_gate(label=name)


class HardwareAdaptAnsatz1(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)
        for i in range(self.num_qubits):
            op = SparsePauliOp.from_sparse_list([("X", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((0, [i]))
            op = SparsePauliOp.from_sparse_list([("Y", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((1, [i]))
            op = SparsePauliOp.from_sparse_list([("Z", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((2, [i]))

        self.gate_pool.append(lambda a: RXGate(a, label=f"$R_X$"))
        self.gate_pool.append(lambda a: RYGate(a, label=f"$R_Y$"))
        self.gate_pool.append(lambda a: RZGate(a, label=f"$R_Z$"))

        for i in range(self.num_qubits - 1):
            op = SparsePauliOp.from_sparse_list([("YX", [i, i + 1], -0.5j),
                                                 ("XY", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((3, [i, i + 1]))

        self.gate_pool.append(lambda a: build_r_yx_xy(a))


class HardwareAdaptAnsatz2(HardwareAdaptAnsatz1):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.x(i)


class HardwareAdaptAnsatz3(HardwareAdaptAnsatz1):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.s(i)

        for i in range(self.num_qubits - 1):
            op = SparsePauliOp.from_sparse_list([("XX", [i, i + 1], -0.5j),
                                                 ("YY", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((4, [i, i + 1]))

        self.gate_pool.append(lambda a: XXPlusYYGate(2 * a, beta=0, label="$R_{XX+YY}$"))


class HardwareAdaptAnsatz4(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)
            self.fixed_ansatz.s(i)

        for i in range(self.num_qubits):
            op = SparsePauliOp.from_sparse_list([("Z", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((0, [i]))

        self.gate_pool.append(lambda a: RZGate(a, label=f"$R_Z$"))

        for i in range(self.num_qubits - 1):
            op = SparsePauliOp.from_sparse_list([("XX", [i, i + 1], -0.5j),
                                                 ("YY", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((1, [i, i + 1]))
            op = SparsePauliOp.from_sparse_list([("YX", [i, i + 1], -0.5j),
                                                 ("XY", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((2, [i, i + 1]))

        self.gate_pool.append(lambda a: XXPlusYYGate(2 * a, beta=0, label="$R_{XX+YY}$"))
        self.gate_pool.append(lambda a: build_r_yx_xy(a))


class HardwareAdaptAnsatz5(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)

        for i in range(self.num_qubits):
            op = SparsePauliOp.from_sparse_list([("X", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((0, [i]))
            op = SparsePauliOp.from_sparse_list([("Y", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((1, [i]))
            op = SparsePauliOp.from_sparse_list([("Z", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((2, [i]))

        self.gate_pool.append(lambda a: RXGate(a, label=f"$R_X$"))
        self.gate_pool.append(lambda a: RYGate(a, label=f"$R_Y$"))
        self.gate_pool.append(lambda a: RZGate(a, label=f"$R_Z$"))

        for i in range(self.num_qubits - 1):
            op = SparsePauliOp.from_sparse_list([("XX", [i, i + 1], -0.5j),
                                                 ("YY", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((3, [i, i + 1]))

            op = SparsePauliOp.from_sparse_list([("XX", [i, i + 1], -0.5j),
                                                 ("YY", [i, i + 1], 0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((4, [i, i + 1]))

            op = SparsePauliOp.from_sparse_list([("YX", [i, i + 1], -0.5j),
                                                 ("XY", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((5, [i, i + 1]))

            op = SparsePauliOp.from_sparse_list([("YX", [i, i + 1], -0.5j),
                                                 ("XY", [i, i + 1], 0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((6, [i, i + 1]))

        self.gate_pool.append(lambda a: XXPlusYYGate(2 * a, beta=0, label="$R_{XX+YY}$"))
        self.gate_pool.append(lambda a: XXMinusYYGate(2 * a, beta=0, label="$R_{XX-YY}$"))
        self.gate_pool.append(lambda a: build_r_yx_xy(a))
        self.gate_pool.append(lambda a: build_r_yx_xy(a, False))


class HardwareAdaptAnsatz6(HardwareAdaptAnsatz5):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.s(i)


class HardwareAdaptAnsatz7(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)

        for i in range(self.num_qubits):
            op = SparsePauliOp.from_sparse_list([("X", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((0, [i]))
            op = SparsePauliOp.from_sparse_list([("Y", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((1, [i]))
            op = SparsePauliOp.from_sparse_list([("Z", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((2, [i]))

        self.gate_pool.append(lambda a: RXGate(a, label=f"$R_X$"))
        self.gate_pool.append(lambda a: RYGate(a, label=f"$R_Y$"))
        self.gate_pool.append(lambda a: RZGate(a, label=f"$R_Z$"))

        for i in range(self.num_qubits - 1):
            op = SparsePauliOp.from_sparse_list([("XX", [i, i + 1], -0.5j),
                                                 ("YY", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((3, [i, i + 1]))

            op = SparsePauliOp.from_sparse_list([("YX", [i, i + 1], -0.5j),
                                                 ("XY", [i, i + 1], 0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((4, [i, i + 1]))

        self.gate_pool.append(lambda a: XXPlusYYGate(2 * a, beta=0, label="$R_{XX+YY}$"))
        self.gate_pool.append(lambda a: build_r_yx_xy(a, False))


class HardwareAdaptAnsatz8(HardwareAdaptAnsatz7):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.s(i)


class HardwareAdaptAnsatz9(HardwareAdaptAnsatz7):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)


class HardwareAdaptAnsatz10(HardwareAdaptAnsatz7):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        if (self.num_qubits % 4) != 0:
            raise NotImplementedError

        self.fixed_ansatz.h(0)

        for i in range(self.num_qubits // 2, self.num_qubits):
            self.fixed_ansatz.x(i)

        for i in range(1, self.num_qubits):
            self.fixed_ansatz.cx(0, i)


class HardwareAdaptAnsatz11(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        if (self.num_qubits % 4) != 0:
            raise NotImplementedError

        self.fixed_ansatz.h(0)

        for i in range(self.num_qubits // 2, self.num_qubits):
            self.fixed_ansatz.x(i)

        for i in range(1, self.num_qubits):
            self.fixed_ansatz.cx(0, i)

        for i in range(self.num_qubits):
            op = SparsePauliOp.from_sparse_list([("X", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((0, [i]))
            op = SparsePauliOp.from_sparse_list([("Y", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((1, [i]))
            op = SparsePauliOp.from_sparse_list([("Z", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((2, [i]))

        self.gate_pool.append(lambda a: RXGate(a, label=f"$R_X$"))
        self.gate_pool.append(lambda a: RYGate(a, label=f"$R_Y$"))
        self.gate_pool.append(lambda a: RZGate(a, label=f"$R_Z$"))

        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                op = SparsePauliOp.from_sparse_list([("XX", [i, j], -0.5j),
                                                     ("YY", [i, j], -0.5j)], num_qubits=self.num_qubits)
                self.operator_pool.append(op)
                self.operator_gate_map.append((3, [i, j]))

                op = SparsePauliOp.from_sparse_list([("YX", [i, j], -0.5j),
                                                     ("XY", [i, j], 0.5j)], num_qubits=self.num_qubits)
                self.operator_pool.append(op)
                self.operator_gate_map.append((4, [i, j]))

        self.gate_pool.append(lambda a: XXPlusYYGate(2 * a, beta=0, label="$R_{XX+YY}$"))
        self.gate_pool.append(lambda a: build_r_yx_xy(a, False))


class HardwareAdaptAnsatz12(HardwareAdaptAnsatz5):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        if (self.num_qubits % 4) != 0:
            raise NotImplementedError

        self.fixed_ansatz.h(0)

        for i in range(self.num_qubits // 2, self.num_qubits):
            self.fixed_ansatz.x(i)

        for i in range(1, self.num_qubits):
            self.fixed_ansatz.cx(0, i)


class HardwareAdaptAnsatz13(HardwareAdaptAnsatz5):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        if (self.num_qubits % 4) != 0:
            raise NotImplementedError

        self.fixed_ansatz.h(0)

        for i in range(1, self.num_qubits):
            self.fixed_ansatz.cx(0, i)

        for i in range(0, self.num_qubits, 4):
            self.fixed_ansatz.x(i)
        for i in range(1, self.num_qubits, 4):
            self.fixed_ansatz.x(i)


class HardwareAdaptAnsatz14(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.h(i)
            self.fixed_ansatz.cx(i, i + 1)

        for i in range(self.num_qubits):
            op = SparsePauliOp.from_sparse_list([("X", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((0, [i]))
            op = SparsePauliOp.from_sparse_list([("Y", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((1, [i]))
            op = SparsePauliOp.from_sparse_list([("Z", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((2, [i]))

        self.gate_pool.append(lambda a: RXGate(a, label=f"$R_X$"))
        self.gate_pool.append(lambda a: RYGate(a, label=f"$R_Y$"))
        self.gate_pool.append(lambda a: RZGate(a, label=f"$R_Z$"))

        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                op = SparsePauliOp.from_sparse_list([("XX", [i, j], -0.5j),
                                                     ("YY", [i, j], -0.5j)], num_qubits=self.num_qubits)
                self.operator_pool.append(op)
                self.operator_gate_map.append((3, [i, j]))

                op = SparsePauliOp.from_sparse_list([("XX", [i, j], -0.5j),
                                                     ("YY", [i, j], 0.5j)], num_qubits=self.num_qubits)
                self.operator_pool.append(op)
                self.operator_gate_map.append((4, [i, j]))

                op = SparsePauliOp.from_sparse_list([("YX", [i, j], -0.5j),
                                                     ("XY", [i, j], -0.5j)], num_qubits=self.num_qubits)
                self.operator_pool.append(op)
                self.operator_gate_map.append((5, [i, j]))

                op = SparsePauliOp.from_sparse_list([("YX", [i, j], -0.5j),
                                                     ("XY", [i, j], 0.5j)], num_qubits=self.num_qubits)
                self.operator_pool.append(op)
                self.operator_gate_map.append((6, [i, j]))

        self.gate_pool.append(lambda a: XXPlusYYGate(2 * a, beta=0, label="$R_{XX+YY}$"))
        self.gate_pool.append(lambda a: XXMinusYYGate(2 * a, beta=0, label="$R_{XX-YY}$"))
        self.gate_pool.append(lambda a: build_r_yx_xy(a))
        self.gate_pool.append(lambda a: build_r_yx_xy(a, False))


class HardwareAdaptAnsatz15(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)

        for i in range(self.num_qubits):
            op = SparsePauliOp.from_sparse_list([("Z", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((0, [i]))

        self.gate_pool.append(lambda a: RZGate(a, label=f"$R_Z$"))

        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                op = SparsePauliOp.from_sparse_list([("XX", [i, j], -0.5j),
                                                     ("YY", [i, j], -0.5j)], num_qubits=self.num_qubits)
                self.operator_pool.append(op)
                self.operator_gate_map.append((1, [i, j]))

                op = SparsePauliOp.from_sparse_list([("YX", [i, j], -0.5j),
                                                     ("XY", [i, j], 0.5j)], num_qubits=self.num_qubits)
                self.operator_pool.append(op)
                self.operator_gate_map.append((2, [i, j]))

        self.gate_pool.append(lambda a: XXPlusYYGate(2 * a, beta=0, label="$R_{XX+YY}$"))
        self.gate_pool.append(lambda a: build_r_yx_xy(a, False))


class HardwareAdaptAnsatz16(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)

        for i in range(self.num_qubits):
            op = SparsePauliOp.from_sparse_list([("Z", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((0, [i]))

        self.gate_pool.append(lambda a: RZGate(a, label=f"$R_Z$"))

        for i in range(self.num_qubits - 1):
            j = i + 1
            op = SparsePauliOp.from_sparse_list([("XX", [i, j], -0.5j),
                                                 ("YY", [i, j], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((1, [i, j]))

            op = SparsePauliOp.from_sparse_list([("YX", [i, j], -0.5j),
                                                 ("XY", [i, j], 0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((2, [i, j]))

        self.gate_pool.append(lambda a: XXPlusYYGate(2 * a, beta=0, label="$R_{XX+YY}$"))
        self.gate_pool.append(lambda a: build_r_yx_xy(a, False))


class HardwareAdaptAnsatz17(HardwareAdaptAnsatz15):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.h(i)
            self.fixed_ansatz.cx(i, i + 1)


class HardwareAdaptAnsatz18(HardwareAdaptAnsatz15):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.x(i)


class HardwareAdaptAnsatz19(HardwareAdaptAnsatz16):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.x(i)


class YXPlusXYRYAdaptAnsatz1(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)
        for i in range(self.num_qubits):
            op = SparsePauliOp.from_sparse_list([("Y", [i], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((0, [i]))

        self.gate_pool.append(lambda a: RYGate(a, label=f"$R_Y$"))

        for i in range(self.num_qubits - 1):
            op = SparsePauliOp.from_sparse_list([("YX", [i, i + 1], -0.5j),
                                                 ("XY", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((1, [i, i + 1]))

        self.gate_pool.append(lambda a: build_r_yx_xy(a))


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


def rebuild_ansatz(group: h5py.Group) -> BaseAnsatz | BaseVQEAnsatz | BaseADAPTVQEAnsatz:
    circuit_dict = load_attribute_as_dict(group[CircuitParameters.Circuit])
    for my_class in inheritors(BaseAnsatz):
        if my_class.__name__ == circuit_dict[CircuitParameters.Ansatz]:
            return my_class(**circuit_dict[CircuitParameters.AnsatzOptions])
    raise NotImplementedError(f"No matching ansatz found, for {circuit_dict[CircuitParameters.Ansatz]}")
