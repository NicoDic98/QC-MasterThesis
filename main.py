import argparse
from time import sleep
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from qiskit import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorOptions

from h5_interface import H5Saver
from labels import VQEParameters
from plotter import ResultLoader
from solver.adapt_vqe import AdaptVQE
from solver.circuits import HardwareAdaptAnsatz1, HardwareAdaptAnsatz2, HardwareAdaptAnsatz3, HardwareAdaptAnsatz4, \
    HardwareAdaptAnsatz5, HardwareAdaptAnsatz6, HardwareAdaptAnsatz7, HardwareAdaptAnsatz8, HardwareAdaptAnsatz9, \
    HardwareAdaptAnsatz10, HardwareAdaptAnsatz11, HardwareAdaptAnsatz12, HardwareAdaptAnsatz13, HardwareAdaptAnsatz14, \
    HardwareAdaptAnsatz15, HardwareAdaptAnsatz16, HardwareAdaptAnsatz17, HardwareAdaptAnsatz18, HardwareAdaptAnsatz19, \
    HardwareAdaptAnsatz20, HardwareAdaptAnsatz21, PhysicsAdaptAnsatz1, PhysicsAdaptAnsatz2, PhysicsAdaptAnsatz3, \
    PhysicsAdaptAnsatz4
from solver.exact_diagonalization import ED
from hamiltonian.free_wilson import FreeWilson2D
from hamiltonian.base import HamiltonianType, HamiltonianParameters
from misc import pprint_h5, data_folder, default_id
from solver.base import GlobalParameters, SimulatorType
from solver.variational_quantum_eigensolver import VQE


def run_ed(my_f: h5py.File):
    my_ed = ED(FreeWilson2D.build_hamiltonian, my_f)
    my_ed.run({HamiltonianParameters.XExtend: [4],
               HamiltonianParameters.YExtend: [4],
               HamiltonianParameters.Mass: np.linspace(-6, 2, 11).tolist(),
               HamiltonianParameters.WilsonParameter: [1.]},
              HamiltonianType.Full, compress_matrix=True)


def run_vqe(my_f: h5py.File):
    my_vqe = VQE(FreeWilson2D.build_hamiltonian, my_f, 8, 4)
    my_vqe.run({HamiltonianParameters.XExtend: [2],
                HamiltonianParameters.YExtend: [2],
                HamiltonianParameters.Mass: np.linspace(-6, 2, 101).tolist(),
                HamiltonianParameters.WilsonParameter: [1.]},
               HamiltonianType.ZeroChargePenalty,
               optimizer_options={
                   "options": {"maxiter": 20000, "disp": 1},
                   "x0Seed": my_f.attrs[GlobalParameters.ProcessId]
               },
               estimator_options=EstimatorOptions(seed_estimator=my_f.attrs[GlobalParameters.ProcessId]))


def run_adapt_vqe(my_f: h5py.File):
    my_adapt_vqe = AdaptVQE(FreeWilson2D.build_hamiltonian, my_f, PhysicsAdaptAnsatz4(8))
    my_adapt_vqe.run({HamiltonianParameters.XExtend: [2],
                      HamiltonianParameters.YExtend: [2],
                      HamiltonianParameters.Mass: np.linspace(-6, 2, 50).tolist(),
                      HamiltonianParameters.WilsonParameter: [1.]},
                     HamiltonianType.ZeroChargePenalty,
                     # simulator_type=SimulatorType.Aer,
                     # simulator_options={
                     #     "method": "statevector",
                     # },
                     optimizer_options={
                         "method": 'slsqp',
                         "options": {"maxiter": 20000, "disp": 0},
                         # "tol": 1e-9,
                         "x0Seed": my_f.attrs[GlobalParameters.ProcessId]
                     },
                     estimator_options=EstimatorOptions(seed_estimator=my_f.attrs[GlobalParameters.ProcessId],
                                                        # default_precision=1.
                                                        ),
                     adapt_options={
                         "max_depth": 50,
                         "gradient_hamiltonian_type": HamiltonianType.ZeroChargePenalty,
                         # "prec_cutoff": 1e-5,
                         # "parameter_prec_cutoff": -1,
                         # "allow_duplicate_gates": True,
                         # "precision": 0.01
                     })


def rerun_vqe(my_f: h5py.File, source_group: h5py.Group):
    source_loader = ResultLoader(source_group)
    energy, mass = source_loader.get_energy_mass({})
    g_name = f"{source_group.name.split('/')[-1]}-rerun-{my_f.attrs[GlobalParameters.ProcessId]}"
    my_f.create_group(g_name)
    for i in range(len(mass)):
        params = {HamiltonianParameters.Mass: i}
        params_values = source_loader.get_parameter_values_from_indices(params)
        ansatz = source_loader.get_ansatz(params, False)
        my_vqe = VQE(FreeWilson2D.build_hamiltonian, my_f[g_name], 8, 4)
        my_vqe.ansatz = ansatz
        my_vqe.run({HamiltonianParameters.XExtend: [2],
                    HamiltonianParameters.YExtend: [2],
                    HamiltonianParameters.Mass: [float(params_values[HamiltonianParameters.Mass])],
                    HamiltonianParameters.WilsonParameter: [1.]},
                   HamiltonianType.ZeroChargePenalty,
                   optimizer_options={
                       "method": 'slsqp',
                       "options": {"maxiter": 20000, "disp": 0},
                       "x0Seed": my_f.attrs[GlobalParameters.ProcessId]
                   },
                   estimator_options=EstimatorOptions(seed_estimator=my_f.attrs[GlobalParameters.ProcessId]))
        sleep(1.5)  # this avoids naming conflicts :D


def measure_observables(group: h5py.Group):
    hamiltonian_factory = FreeWilson2D.build_hamiltonian
    result_loader = ResultLoader(group)
    _, masses = result_loader.get_energy_mass({})
    h5_saver = H5Saver(group,
                       {HamiltonianParameters.XExtend: [2],
                        HamiltonianParameters.YExtend: [2],
                        HamiltonianParameters.Mass: masses.tolist(),
                        HamiltonianParameters.WilsonParameter: [1.]},
                       False
                       )

    observable_name_list = [
        VQEParameters.HamiltonianVariance,
        VQEParameters.ChargeConjugation,
        VQEParameters.ChargeConjugationVariance,
    ]
    for observable_name in observable_name_list:
        if observable_name in group:
            del group[observable_name]
        h5_saver.create_dataset_with_dim_labels(observable_name, [], [])

    c_op = SparsePauliOp.from_sparse_list([("XX", [1, 0], 0.5),  # need to flip order due to phi convention chosen
                                           ("YY", [1, 0], -0.5),
                                           ("IZ", [1, 0], -0.5),
                                           ("ZI", [1, 0], 0.5),
                                           ], num_qubits=2)

    for parameters, non_singular_index in zip(h5_saver.parameters_list_dict, h5_saver.non_singular_indices_list):
        hamiltonian = hamiltonian_factory(**parameters)
        print(f"Calculating for {hamiltonian}", flush=True)

        i = non_singular_index[0]
        params = {HamiltonianParameters.Mass: i}
        param_values = result_loader.get_parameter_values_from_indices(params)
        print([f"{x}={y:.3f}" for x, y in param_values.items()], flush=True)
        circuit = result_loader.get_circuit(params, final=True, exit_on_duplicate=False)

        h_operator = hamiltonian.hamiltonian_op(HamiltonianType.Full)
        if circuit.num_qubits % 2 != 0:
            raise NotImplementedError
        complete_c_op = c_op
        for _ in range(circuit.num_qubits // 2 - 1):
            complete_c_op = complete_c_op.tensor(c_op)

        pm = generate_preset_pass_manager()
        estimator = StatevectorEstimator()
        circuit = pm.run(circuit)
        pub = (circuit, [
            [h_operator],
            [(h_operator.compose(h_operator)).simplify()],
            [complete_c_op.simplify()]
        ])
        job = estimator.run(pubs=[pub])
        result = job.result()

        pub_result = result[0]
        h_exp = pub_result.data.evs[0][0]
        hh_exp = pub_result.data.evs[1][0]
        c_exp = pub_result.data.evs[2][0]
        h_var = hh_exp - (h_exp * h_exp)
        c_var = 1 - (c_exp * c_exp)

        print("H:", h_exp, hh_exp, h_var, flush=True)
        print("C:", c_exp, c_var, flush=True)
        dataset = group[VQEParameters.HamiltonianVariance]
        dataset[*non_singular_index] = h_var
        dataset = group[VQEParameters.ChargeConjugation]
        dataset[*non_singular_index] = c_exp
        dataset = group[VQEParameters.ChargeConjugationVariance]
        dataset[*non_singular_index] = c_var


# Define the parser
parser = argparse.ArgumentParser(description='Main app')
parser.add_argument('--id', action="store", dest='id', default=default_id, type=int)
args = parser.parse_args()

Path(data_folder).mkdir(parents=True, exist_ok=True)

h5_file = f"{data_folder}{datetime.now().strftime('%Y-%m-%U')}-{args.id}"

# with h5py.File(h5_file + ".hdf5", "w", libver='latest') as f:
#     pprint_h5(f)
#     f.attrs[GlobalParameters.ProcessId] = args.id
#     # run_ed(f)
#     # run_vqe(f)
#     run_adapt_vqe(f)
#     # with h5py.File(f"{data_folder}{"2025-11-46"}.hdf5", "r") as sf:
#     #     rerun_vqe(f, sf["2025-11-20_17-57-51-23964135"])
#     pprint_h5(f)

with h5py.File(f"{data_folder}{"2026-01-01"}.hdf5", "a") as sf:
    pprint_h5(sf)
    measure_observables(sf["2026-01-07_14-31-34-24236398"])
    pprint_h5(sf)
