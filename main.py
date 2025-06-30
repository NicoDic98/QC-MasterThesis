import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator

# Create a new circuit with two qubits
qc = QuantumCircuit(2)

# Add a Hadamard gate to qubit 0
qc.h(0)

# Perform a controlled-X gate on qubit 1, controlled by qubit 0
qc.cx(0, 1)

# Return a drawing of the circuit using MatPlotLib ("mpl").
# These guides are written by using Jupyter notebooks, which
# display the output of the last line of each cell.
# If you're running this in a script, use `print(qc.draw())` to
# print a text drawing.
qc.draw("mpl")
plt.show()
# Set up six different observables.

observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]# in format "q_1, q_0"
observables = [SparsePauliOp(label) for label in observables_labels]

# Use the following code instead if you want to run on a simulator:

backend = FakeAlmadenV2()

# Convert to an ISA circuit and layout-mapped observables.
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)
isa_circuit.draw("mpl")
plt.show()

# Run:
estimator = Estimator(backend)


mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]
print(observables)
print(mapped_observables)

job = estimator.run([(isa_circuit, mapped_observables)])
result = job.result()
#
#
#

# This is the result of the entire submission.  You submitted one Pub,
# so this contains one inner result (and some metadata of its own).

job_result = job.result()
print(job_result)

# This is the result from our single pub, which had six observables,
# so contains information on all six.

pub_result = job.result()[0]

# Plot the result

values = pub_result.data.evs

errors = pub_result.data.stds

# plotting graph
plt.errorbar(observables_labels, values, errors)
plt.xlabel("Observables")
plt.ylabel("Values")
plt.show()