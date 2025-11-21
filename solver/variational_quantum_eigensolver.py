from typing import Callable, Any

import h5py
import numpy as np
from qiskit_ibm_runtime import EstimatorOptions
from scipy.optimize import minimize

from hamiltonian.base import HamiltonianType
from hamiltonian.free_wilson import BaseHamiltonian
from labels import VQEParameters
from solver.base import SimulatorType, BaseVQE, BaseCostFunction
from solver.circuits import XXPlusYYRZAnsatz1


class VQECostFunction(BaseCostFunction):
    pass


class VQE(BaseVQE):
    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group,
                 num_qubits: int,
                 num_layers: int, ):
        super().__init__(hamiltonian_factory, save_group,
                         XXPlusYYRZAnsatz1(num_qubits, num_layers))
        self.cost_function = VQECostFunction

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType,
            simulator_type: SimulatorType = SimulatorType.Statevector,
            simulator_options: dict[str, Any] = None,
            preset_pass_manager_options: dict[str, Any] = None,
            estimator_options: EstimatorOptions = None,
            optimizer_options: dict[str, Any] = None):
        local_group, h5_saver, estimator, pm, x0 = self.initialize_run(parameters_dict_list, hamiltonian_type,
                                                                       simulator_type, simulator_options,
                                                                       preset_pass_manager_options, estimator_options,
                                                                       optimizer_options)

        circuit = pm.run(self.ansatz())

        for parameters, non_singular_index in zip(h5_saver.parameters_list_dict, h5_saver.non_singular_indices_list):
            hamiltonian = self.hamiltonian_factory(**parameters)
            print(f"Calculating energies for {hamiltonian}", flush=True)
            h_operator = hamiltonian.hamiltonian_op(hamiltonian_type)
            h_operator = h_operator.apply_layout(layout=circuit.layout)

            minimize(fun=self.cost_function(circuit, h_operator, estimator, local_group, non_singular_index),
                     x0=x0, **optimizer_options)
