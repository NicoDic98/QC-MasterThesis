import argparse
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from qiskit_ibm_runtime import EstimatorOptions

from solver.exact_diagonalization import ED
from hamiltonian.free_wilson import FreeWilson2D
from hamiltonian.base import HamiltonianType, HamiltonianParameters
from misc import pprint_h5, data_folder, default_id
from solver.base import GlobalParameters
from solver.variational_quantum_eigensolver import VQE


def run_ed(my_f: h5py.File):
    my_ed = ED(FreeWilson2D.build_hamiltonian, my_f)
    my_ed.run({HamiltonianParameters.XExtend: [2],
               HamiltonianParameters.YExtend: [2],
               HamiltonianParameters.Mass: np.linspace(-6, 2, 1001).tolist(),
               HamiltonianParameters.WilsonParameter: [1.]},
              HamiltonianType.ZeroChargePenalty)


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


# Define the parser
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--id', action="store", dest='id', default=default_id, type=int)
args = parser.parse_args()

Path(data_folder).mkdir(parents=True, exist_ok=True)

h5_file = f"{data_folder}{datetime.now().strftime('%Y-%m-%U')}-{args.id}"

with h5py.File(h5_file + ".hdf5", "a") as f:
    pprint_h5(f)
    f.attrs[GlobalParameters.ProcessId] = args.id
    # run_ed(f)
    run_vqe(f)
    pprint_h5(f)
