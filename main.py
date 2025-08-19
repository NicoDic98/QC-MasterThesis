import argparse
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from exact_diagonalization import ED
from hamiltonians import FreeWilson2D, HamiltonianType, HamiltonianParameters
from misc import pprint_h5, GlobalParameters, data_folder, default_id

# Define the parser
parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--id', action="store", dest='id', default=default_id)
args = parser.parse_args()

masses = np.linspace(-6, 2, 501)
Path(data_folder).mkdir(parents=True, exist_ok=True)
with h5py.File(f"{data_folder}{datetime.now().strftime('%Y-%m-%U')}-{args.id}.hdf5", "a") as f:
    pprint_h5(f)
    f.attrs[GlobalParameters.ProcessId] = args.id
    my_ed = ED(FreeWilson2D.build_hamiltonian, f)
    my_ed.run({HamiltonianParameters.XExtend: [2],
               HamiltonianParameters.YExtend: [2],
               HamiltonianParameters.Mass: masses.tolist(),
               HamiltonianParameters.WilsonParameter: [1., 3.14]},
              HamiltonianType.ZeroChargePenalty)
    pprint_h5(f)
# attr: ProcessId
