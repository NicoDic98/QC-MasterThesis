from enum import StrEnum
import functools
from itertools import product
from typing import Any, Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Gate, ParameterVector
from qiskit.circuit.library import n_local, XXPlusYYGate, RZGate, RYGate, RXGate, XXMinusYYGate, RXXGate, RYYGate, \
    RZZGate, PauliEvolutionGate, PauliProductRotationGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp, Pauli
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
        self.last_parameter_index_on_qubit = [-1] * self.num_qubits

    def max_num_parameters(self):
        """
        Returns the maximum number of parameters in the ansatz. None means unlimited.
        :return: maximum number of parameters
        """
        return None

    def set_ansatz(self, operator_indices: list[int], exit_on_duplicate=True, update_last_parameter_on_qubit=False):
        qc = self.fixed_ansatz.copy()
        my_params = ParameterVector("A", len(operator_indices))
        last_gate_on_qubit = [-1] * self.num_qubits
        if update_last_parameter_on_qubit:
            self.last_parameter_index_on_qubit = [-1] * self.num_qubits
        for i, oi in enumerate(operator_indices):
            gi, qbits = self.operator_gate_map[oi]
            ok = False
            for qbit in qbits:
                if last_gate_on_qubit[qbit] != oi:
                    ok = True
            if (not ok) and exit_on_duplicate:
                return 1
            gate = self.gate_pool[gi](my_params[i])
            qc.append(gate, qbits)
            if update_last_parameter_on_qubit:
                for qbit in qbits:
                    last_gate_on_qubit[qbit] = oi
                    self.last_parameter_index_on_qubit[qbit] = i
            else:
                for qbit in qbits:
                    last_gate_on_qubit[qbit] = oi
        self.full_ansatz = qc
        return 0

    def get_previous_parameter_indices(self, oi: int):
        gi, qbits = self.operator_gate_map[oi]
        temp = []
        for qbit in qbits:
            temp.append(self.last_parameter_index_on_qubit[qbit])
        temp = np.unique(temp)
        temp = temp[temp >= 0]
        return temp.tolist()

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
        gname = gate.label
        if gname is not None:
            if gname.endswith(r"\right)$"):
                gname = gname.rpartition(r"\left(")[0]
        return gi, qbits, gname


def build_r_yx_xy(b: Parameter | float, plus=True, val_str=""):
    if plus:
        name = "$R_{YX+XY}$"
    else:
        name = "$R_{YX-XY}$"

    if val_str == "":
        if np.issubdtype(type(b), np.floating):
            val_str = f"{b:.2f}"
        else:
            val_str = f"{b}"
    name = name[:-1] + r"\left(" + val_str + r"\right)$"

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


def build_r_xzy_m_yzx(b: Parameter | float, n: int = 0, ret_gate=True, val_str=""):
    name = "$R_{YX-XY}^{(" + str(n) + ")}$"
    if val_str == "":
        if np.issubdtype(type(b), np.floating):
            val_str = f"{b:.2f}"
        else:
            val_str = f"{b}"
    name = name[:-1] + r"\left(" + val_str + r"\right)$"

    if n < 0:
        raise ValueError("n cannot be negative")
    n_floor = n // 2
    n_ceil = (n + 1) // 2

    qc = QuantumCircuit(n + 2, name=name)

    for j in range(0, n_floor):
        gate = build_r_yx_xy(np.pi / 2, val_str=r"\frac{\pi}{2}")
        qc.append(gate, [n - j, n - j + 1])
    for j in range(0, n_ceil):
        gate = build_r_yx_xy(np.pi / 2, val_str=r"\frac{\pi}{2}")
        qc.append(gate, [j, j + 1])

    if n % 2:
        # Odd
        gate = build_r_yx_xy(((-1) ** n_ceil) * b)
    else:
        # Even
        gate = build_r_yx_xy(((-1) ** n_ceil) * b, False)

    qc.append(gate, [n_ceil, n_ceil + 1])

    for j in range(n_floor - 1, -1, -1):
        gate = build_r_yx_xy(-np.pi / 2, val_str=r"-\frac{\pi}{2}")
        qc.append(gate, [n - j, n - j + 1])
    for j in range(n_ceil - 1, -1, -1):
        gate = build_r_yx_xy(-np.pi / 2, val_str=r"-\frac{\pi}{2}")
        qc.append(gate, [j, j + 1])

    if ret_gate:
        return qc.to_gate(label=name)
    else:
        return qc


def combine_gates(gates: list[Gate], qbits: list[list[int]], n: int, name: str):
    qc = QuantumCircuit(n, name=name)
    for gate, qbit in zip(gates, qbits):
        qc.append(gate, qbit)
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


class HardwareAdaptAnsatz20(HardwareAdaptAnsatz5):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.h(i)
            self.fixed_ansatz.cx(i, i + 1)


class HardwareAdaptAnsatz21(HardwareAdaptAnsatz11):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)


class HardwareAdaptAnsatz22(BaseADAPTVQEAnsatz):
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
            op = SparsePauliOp.from_sparse_list([("XX", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((3, [i, i + 1]))

            op = SparsePauliOp.from_sparse_list([("YY", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((4, [i, i + 1]))

            op = SparsePauliOp.from_sparse_list([("ZZ", [i, i + 1], -0.5j)], num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((5, [i, i + 1]))

        self.gate_pool.append(lambda a: RXXGate(a, label="$R_{XX}$"))
        self.gate_pool.append(lambda a: RYYGate(a, label="$R_{YY}$"))
        self.gate_pool.append(lambda a: RZZGate(a, label="$R_{ZZ}$"))


class HardwareAdaptAnsatz23(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)

        pauli_strings = ["".join(tu) for tu in product(*(["XYZI"] * self.num_qubits))]
        # remove those elements which result in vanishing commutators, qiskit throws errors if the operator is 0
        pauli_strings.pop(65535)
        pauli_strings.pop(61115)
        pauli_strings.pop(48110)
        pauli_strings.pop(43690)
        helper_func = lambda param, ps, label: PauliProductRotationGate(Pauli(ps), angle=param, label=label)
        for i, pauli_string in enumerate(pauli_strings):
            op = SparsePauliOp.from_sparse_list([(pauli_string,
                                                  list(range(self.num_qubits)), -0.5j)],
                                                num_qubits=self.num_qubits)
            self.operator_pool.append(op)
            self.operator_gate_map.append((i, list(range(self.num_qubits))))
            # self.gate_pool.append(
            #     functools.partial(PauliEvolutionGate,
            #                       1j * self.operator_pool[i],
            #                       functools.Placeholder,
            #                       label="$R_{" + self.operator_pool[i].to_list()[0][0] + "}$"))
            # Can use functools.Placeholder starting from 3.14 for this for now need this:
            self.gate_pool.append(
                functools.partial(helper_func,
                                  ps=self.operator_pool[i].to_list()[0][0],
                                  label="$R_{" + self.operator_pool[i].to_list()[0][0][::-1] + "}$"))


class HardwareAdaptAnsatz24(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)

        helper_func = lambda param, ps, label: PauliProductRotationGate(Pauli(ps), angle=param, label=label)
        i = -1
        for gate_size in range(1, self.num_qubits + 1):
            pauli_strings = ["".join(tu) for tu in product(*(["XYZ"] * gate_size))]
            for pauli_string in pauli_strings:
                for start_qubit in range(self.num_qubits - (gate_size - 1)):
                    qubit_list = list(range(start_qubit, start_qubit + gate_size))
                    op = SparsePauliOp.from_sparse_list(
                        [(pauli_string, qubit_list, -0.5j)],
                        num_qubits=self.num_qubits)
                    self.operator_pool.append(op)
                    i += 1
                    self.operator_gate_map.append((len(self.gate_pool), qubit_list))

                self.gate_pool.append(
                    functools.partial(helper_func,
                                      ps=self.operator_pool[i].to_list()[0][0].replace("I",""),
                                      label="$R_{" + self.operator_pool[i].to_list()[0][0].replace("I","")[::-1] + "}$"))

class HardwareAdaptAnsatz25(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        for i in range(0, self.num_qubits):
            self.fixed_ansatz.h(i)

        helper_func = lambda param, ps, label: PauliProductRotationGate(Pauli(ps), angle=param, label=label)
        i = -1
        for gate_size in range(1, 4 + 1):
            pauli_strings = ["".join(tu) for tu in product(*(["XYZ"] * gate_size))]
            for pauli_string in pauli_strings:
                for start_qubit in range(self.num_qubits - (gate_size - 1)):
                    qubit_list = list(range(start_qubit, start_qubit + gate_size))
                    op = SparsePauliOp.from_sparse_list(
                        [(pauli_string, qubit_list, -0.5j)],
                        num_qubits=self.num_qubits)
                    self.operator_pool.append(op)
                    i += 1
                    self.operator_gate_map.append((len(self.gate_pool), qubit_list))

                self.gate_pool.append(
                    functools.partial(helper_func,
                                      ps=self.operator_pool[i].to_list()[0][0].replace("I",""),
                                      label="$R_{" + self.operator_pool[i].to_list()[0][0].replace("I","")[::-1] + "}$"))


class PhysicsAdaptAnsatz1(BaseADAPTVQEAnsatz):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        if num_qubits != 8:
            raise NotImplementedError

        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.x(i)

        # TODO: These are ordered directly according to the snake,
        #  the ordering has to be flipped such that it is consistent with the phi convention.
        # A_1(0,0) -> U_1[0,1](b)=exp(ib * A_1(0,1)):
        op = SparsePauliOp.from_sparse_list([("XY", [1, 2], 0.5j),
                                             ("YX", [1, 2], -0.5j),
                                             ("XZZY", list(range(0, 4)), 0.5j),
                                             ("YZZX", list(range(0, 4)), -0.5j),
                                             ], num_qubits=self.num_qubits)
        self.operator_pool.append(op)
        self.operator_gate_map.append((0, list(range(0, 4))))
        self.gate_pool.append(
            lambda a: combine_gates([build_r_xzy_m_yzx(a, 0), build_r_xzy_m_yzx(a, 2)],
                                    [[1, 2], list(range(0, 4))],
                                    4,
                                    "$U_1[0,1]$"))

        # A_1(0,1) -> U_1[0,0](b)=exp(ib * A_1(0,0)):
        op = SparsePauliOp.from_sparse_list([("XY", [5, 6], -0.5j),
                                             ("YX", [5, 6], 0.5j),
                                             ("XZZY", list(range(4, 8)), -0.5j),
                                             ("YZZX", list(range(4, 8)), 0.5j),
                                             ], num_qubits=self.num_qubits)
        self.operator_pool.append(op)
        self.operator_gate_map.append((1, list(range(4, 8))))
        self.gate_pool.append(
            lambda a: combine_gates([build_r_xzy_m_yzx(-a, 0), build_r_xzy_m_yzx(-a, 2)],
                                    [[1, 2], list(range(4))],
                                    4,
                                    "$U_1[0,0]$"))

        # A_2(0,0) -> U_2[0,1](b)=exp(ib * A_2(0,1)):
        op = SparsePauliOp.from_sparse_list([("XZZZZZY", list(range(1, 8)), -0.5j),
                                             ("YZZZZZX", list(range(1, 8)), 0.5j),
                                             ("XZZZZZY", list(range(0, 7)), -0.5j),
                                             ("YZZZZZX", list(range(0, 7)), 0.5j),
                                             ], num_qubits=self.num_qubits)
        self.operator_pool.append(op)
        self.operator_gate_map.append((2, list(range(0, 8))))
        self.gate_pool.append(
            lambda a: combine_gates([build_r_xzy_m_yzx(-a, 5), build_r_xzy_m_yzx(-a, 5)],
                                    [list(range(1, 8)), list(range(0, 7))],
                                    8,
                                    "$U_2[0,1]$"))

        # A_2(1,0) -> U_2[1,1](b)=exp(ib * A_2(1,1)):
        op = SparsePauliOp.from_sparse_list([("XZY", list(range(3, 6)), -0.5j),
                                             ("YZX", list(range(3, 6)), 0.5j),
                                             ("XZY", list(range(2, 5)), -0.5j),
                                             ("YZX", list(range(2, 5)), 0.5j),
                                             ], num_qubits=self.num_qubits)
        self.operator_pool.append(op)
        self.operator_gate_map.append((3, list(range(2, 6))))
        self.gate_pool.append(
            lambda a: combine_gates([build_r_xzy_m_yzx(-a, 1), build_r_xzy_m_yzx(-a, 1)],
                                    [list(range(1, 4)), list(range(0, 3))],
                                    4,
                                    "$U_2[1,1]$"))

        # A_3(0,0) -> U_3[0,1](b)=exp(ib * A_3(0,1)):
        op = SparsePauliOp.from_sparse_list([("XZY", list(range(0, 3)), -0.5j),
                                             ("YZX", list(range(0, 3)), 0.5j),
                                             ("XZY", list(range(1, 4)), -0.5j),
                                             ("YZX", list(range(1, 4)), 0.5j),
                                             ], num_qubits=self.num_qubits)
        self.operator_pool.append(op)
        self.operator_gate_map.append((4, list(range(0, 4))))
        self.gate_pool.append(
            lambda a: combine_gates([build_r_xzy_m_yzx(-a, 1), build_r_xzy_m_yzx(-a, 1)],
                                    [list(range(0, 3)), list(range(1, 4))],
                                    4,
                                    "$U_3[0,1]$"))

        # A_3(0,1) -> U_3[0,0](b)=exp(ib * A_3(0,0)):
        op = SparsePauliOp.from_sparse_list([("XZY", list(range(5, 8)), 0.5j),
                                             ("YZX", list(range(5, 8)), -0.5j),
                                             ("XZY", list(range(4, 7)), 0.5j),
                                             ("YZX", list(range(4, 7)), -0.5j),
                                             ], num_qubits=self.num_qubits)
        self.operator_pool.append(op)
        self.operator_gate_map.append((5, list(range(4, 8))))
        self.gate_pool.append(
            lambda a: combine_gates([build_r_xzy_m_yzx(a, 1), build_r_xzy_m_yzx(a, 1)],
                                    [list(range(1, 4)), list(range(0, 3))],
                                    4,
                                    "$U_3[0,0]$"))

        # A_4(0,0) -> U_4[0,1](b)=exp(ib * A_4(0,1)):
        op = SparsePauliOp.from_sparse_list([("XZZZZZZY", list(range(0, 8)), -0.5j),
                                             ("YZZZZZZX", list(range(0, 8)), 0.5j),
                                             ("XZZZZY", list(range(1, 7)), -0.5j),
                                             ("YZZZZX", list(range(1, 7)), 0.5j),
                                             ], num_qubits=self.num_qubits)
        self.operator_pool.append(op)
        self.operator_gate_map.append((6, list(range(0, 8))))
        self.gate_pool.append(
            lambda a: combine_gates([build_r_xzy_m_yzx(-a, 6), build_r_xzy_m_yzx(-a, 4)],
                                    [list(range(0, 8)), list(range(1, 7))],
                                    8,
                                    "$U_4[0,1]$"))

        # A_4(1,0) -> U_4[1,1](b)=exp(ib * A_4(1,1)):
        op = SparsePauliOp.from_sparse_list([("XZZY", list(range(2, 6)), -0.5j),
                                             ("YZZX", list(range(2, 6)), 0.5j),
                                             ("XY", list(range(3, 5)), -0.5j),
                                             ("YX", list(range(3, 5)), 0.5j),
                                             ], num_qubits=self.num_qubits)
        self.operator_pool.append(op)
        self.operator_gate_map.append((7, list(range(2, 6))))
        self.gate_pool.append(
            lambda a: combine_gates([build_r_xzy_m_yzx(-a, 2), build_r_xzy_m_yzx(-a, 0)],
                                    [list(range(0, 4)), list(range(1, 3))],
                                    4,
                                    "$U_4[1,1]$"))


class PhysicsAdaptAnsatz2(PhysicsAdaptAnsatz1):
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


class PhysicsAdaptAnsatz3(PhysicsAdaptAnsatz1):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        for i in range(0, self.num_qubits, 2):
            self.fixed_ansatz.h(i)
            self.fixed_ansatz.cx(i, i + 1)


class PhysicsAdaptAnsatz4(PhysicsAdaptAnsatz1):
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


class PhysicsAdaptAnsatz5(PhysicsAdaptAnsatz1):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        self.fixed_ansatz.x(1)
        for i in range(2, self.num_qubits, 2):
            self.fixed_ansatz.x(i)


class PhysicsAdaptAnsatz6(PhysicsAdaptAnsatz1):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        for i in range(4, self.num_qubits):
            self.fixed_ansatz.x(i)


class PhysicsAdaptAnsatz7(PhysicsAdaptAnsatz1):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        self.fixed_ansatz.x(self.num_qubits - 1)
        for i in range(0, self.num_qubits - 2, 2):
            self.fixed_ansatz.x(i)


class PhysicsAdaptAnsatz8(PhysicsAdaptAnsatz1):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.fixed_ansatz = QuantumCircuit(self.num_qubits)
        for i in range(1, self.num_qubits, 2):
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
