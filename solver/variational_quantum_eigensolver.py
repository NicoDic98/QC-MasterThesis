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
    def update_dataset_size(self, dataset: h5py.Dataset, iteration_index: int = -1):
        if self.iteration >= dataset.shape[iteration_index]:
            dataset.resize(self.iteration + 10, len(dataset.shape) + iteration_index)

    def __call__(self, params: np.ndarray) -> float:
        dataset = self.group[VQEParameters.CircuitParameters]
        self.update_dataset_size(dataset, -2)
        dataset[*self.current_non_singular_index, self.iteration, :] = params

        full_result = self.evaluate(params)
        pub_result = full_result[0]

        for key, value in pub_result.data.items():
            for i, operator_name_suffix in enumerate([VQEParameters.HamiltonianSuffix]):
                dataset = self.group[VQEParameters.DataPrefix + key + operator_name_suffix]
                self.update_dataset_size(dataset)
                if not (h5py.check_string_dtype(dataset.dtype) is None):
                    dataset[*self.current_non_singular_index, self.iteration] = str(value[i])  # only one pub
                else:
                    dataset[*self.current_non_singular_index, self.iteration] = value[i]  # only one pub

        for key, value in pub_result.metadata.items():  # pub specific metadata
            dataset = self.group[VQEParameters.MetaDataPrefix + key]
            self.update_dataset_size(dataset)
            if not (h5py.check_string_dtype(dataset.dtype) is None):
                dataset[*self.current_non_singular_index, self.iteration] = str(value)
            else:
                dataset[*self.current_non_singular_index, self.iteration] = value

        for key, value in full_result.metadata.items():  # general metadata
            dataset = self.group[VQEParameters.MetaDataPrefix + key]
            self.update_dataset_size(dataset)
            if not (h5py.check_string_dtype(dataset.dtype) is None):
                dataset[*self.current_non_singular_index, self.iteration] = str(value)
            else:
                dataset[*self.current_non_singular_index, self.iteration] = value

        dataset = self.group[VQEParameters.NIterations]
        dataset[*self.current_non_singular_index] = self.iteration

        energy = pub_result.data["evs"][0]
        self.iteration += 1
        return energy


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
            print(f"Calculating energies for {hamiltonian}")
            h_operator = hamiltonian.hamiltonian_op(hamiltonian_type)
            h_operator = h_operator.apply_layout(layout=circuit.layout)

            minimize(fun=self.cost_function(circuit, h_operator, estimator, local_group, non_singular_index),
                     x0=x0, **optimizer_options)
