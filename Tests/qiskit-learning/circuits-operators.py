from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, MCXGate
import matplotlib.pyplot as plt


mcx_gate = MCXGate(3)
hadamard_gate = HGate()

qc = QuantumCircuit(4)
qc.append(hadamard_gate, [0])
qc.append(mcx_gate, [0, 1, 2, 3])
qc.draw("mpl")
plt.show()

from qiskit.circuit.library import n_local

two_local = n_local(3, "rx", "cz")
two_local.draw("mpl")
plt.show()
print(two_local.parameters)

bound_circuit = two_local.assign_parameters(
    {p: 0 for p in two_local.parameters}
)
bound_circuit.decompose().draw("mpl")
plt.show()