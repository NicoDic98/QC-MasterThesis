import argparse
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from qiskit_ibm_runtime import EstimatorOptions

from solver.adapt_vqe import AdaptVQE
from solver.circuits import HardwareAdaptAnsatz1, HardwareAdaptAnsatz2, HardwareAdaptAnsatz3, HardwareAdaptAnsatz4, \
    HardwareAdaptAnsatz5, HardwareAdaptAnsatz6, HardwareAdaptAnsatz7, HardwareAdaptAnsatz8, HardwareAdaptAnsatz9, \
    HardwareAdaptAnsatz10, HardwareAdaptAnsatz11
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
    my_adapt_vqe = AdaptVQE(FreeWilson2D.build_hamiltonian, my_f, HardwareAdaptAnsatz11(8))
    my_adapt_vqe.run({HamiltonianParameters.XExtend: [2],
                      HamiltonianParameters.YExtend: [2],
                      HamiltonianParameters.Mass: np.linspace(-6, 2, 10).tolist(),
                      HamiltonianParameters.WilsonParameter: [1.]},
                     HamiltonianType.ZeroChargePenalty,
                     # simulator_type=SimulatorType.Aer,
                     optimizer_options={
                         "method": 'slsqp',
                         "options": {"maxiter": 20000, "disp": 0},
                         # "tol": 1e-5,
                         "x0Seed": my_f.attrs[GlobalParameters.ProcessId]
                     },
                     estimator_options=EstimatorOptions(seed_estimator=my_f.attrs[GlobalParameters.ProcessId],
                                                        # default_precision=1.
                                                        ),
                     adapt_options={
                         "max_depth": 30,
                         "gradient_hamiltonian_type": HamiltonianType.ZeroChargePenalty,
                         # "precision": 0.01
                     })


# Define the parser
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--id', action="store", dest='id', default=default_id, type=int)
args = parser.parse_args()

Path(data_folder).mkdir(parents=True, exist_ok=True)

h5_file = f"{data_folder}{datetime.now().strftime('%Y-%m-%U')}-{args.id}"

with h5py.File(h5_file + ".hdf5", "w", libver='latest') as f:
    pprint_h5(f)
    f.attrs[GlobalParameters.ProcessId] = args.id
    # run_ed(f)
    # run_vqe(f)
    run_adapt_vqe(f)
    pprint_h5(f)
