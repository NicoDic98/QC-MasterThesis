from typing import Any, Callable

import h5py
import numpy as np
from matplotlib import pyplot as plt
from qiskit_ibm_runtime import EstimatorOptions
from scipy.optimize import minimize

from h5_interface import save_dict_as_attribute
from hamiltonian.base import HamiltonianType, BaseHamiltonian
from labels import VQEParameters
from misc import fill_defaults_in_dict, calc_im_part
from solver.base import SimulatorType, BaseVQE, BaseCostFunction
from solver.circuits import XXPlusYYRZAdaptAnsatz1, BaseADAPTVQEAnsatz


class AdaptVQECostFunction(BaseCostFunction):
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


class AdaptVQE(BaseVQE):
    ansatz: BaseADAPTVQEAnsatz

    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group,
                 num_qubits: int,
                 max_depth: int):
        super().__init__(hamiltonian_factory, save_group,
                         XXPlusYYRZAdaptAnsatz1(num_qubits))
        self.cost_function = AdaptVQECostFunction
        self.max_depth = max_depth

    def run(self, parameters_dict_list: dict[str, list], hamiltonian_type: HamiltonianType,
            simulator_type: SimulatorType = SimulatorType.Statevector,
            simulator_options: dict[str, Any] = None,
            preset_pass_manager_options: dict[str, Any] = None,
            estimator_options: EstimatorOptions = None,
            optimizer_options: dict[str, Any] = None,
            adapt_options: dict[str, Any] = None, ):
        local_group, h5_saver, estimator, pm, x0 = self.initialize_run(parameters_dict_list, hamiltonian_type,
                                                                       simulator_type, simulator_options,
                                                                       preset_pass_manager_options, estimator_options,
                                                                       optimizer_options)
        if adapt_options is None:
            adapt_options = {}
        adapt_options_default = {
            "prec_cutoff": 1e-3
        }
        fill_defaults_in_dict(adapt_options, adapt_options_default)
        save_dict_as_attribute(local_group, adapt_options, VQEParameters.AdaptOptions)

        h5_saver.create_dataset_with_dim_labels(VQEParameters.StartIterations,
                                                [0],
                                                [VQEParameters.AnsatzOperatorAxis],
                                                int,
                                                [None],
                                                -1)

        h5_saver.create_dataset_with_dim_labels(VQEParameters.AnsatzOperators,
                                                [0],
                                                [VQEParameters.AnsatzOperatorAxis],
                                                int,
                                                [None],
                                                -1)

        for parameters, non_singular_index in zip(h5_saver.parameters_list_dict, h5_saver.non_singular_indices_list):
            hamiltonian = self.hamiltonian_factory(**parameters)
            print(f"Calculating energies for {hamiltonian}")
            h_operator = hamiltonian.hamiltonian_op(hamiltonian_type)
            print("h:", calc_im_part(h_operator))
            commutator_list = [(h_operator @ op - op @ h_operator).simplify() for op in self.ansatz.operator_pool]
            # sec_commutator_list = [(op2 @ op1 - op1 @ op2).simplify()
            #                        for op1, op2 in zip(self.ansatz.operator_pool, commutator_list)]
            # print("[op]:\n",self.ansatz.operator_pool)
            # print("[,]:\n",commutator_list)

            self.ansatz.set_ansatz([])
            circuit = pm.run(self.ansatz())
            params = x0
            op_index_list = []
            cost_function_instance = self.cost_function(circuit, h_operator.apply_layout(circuit.layout),
                                                        estimator, local_group, non_singular_index)
            for i in range(self.max_depth):
                pub = (circuit, [op.apply_layout(circuit.layout) for op in commutator_list], [params])
                # noinspection PyTypeChecker
                job = estimator.run(pubs=[pub])
                full_result = job.result()
                pub_result = full_result[0]
                gradients = pub_result.data.evs
                abs_gradients = np.abs(gradients)

                # pub = (circuit, sec_commutator_list, [params])
                # # print(pub)
                # # noinspection PyTypeChecker
                # job = estimator.run(pubs=[pub])
                # full_result = job.result()
                # pub_result = full_result[0]
                # print(pub_result.data.evs)

                if abs_gradients.sum() < adapt_options["prec_cutoff"]:
                    print("Precision cutoff reached.")
                    break

                new_op_index = np.argmax(abs_gradients)
                if len(op_index_list):
                    if new_op_index == op_index_list[-1]:
                        print("Adding the same operator twice is not sensible, terminating.")
                        break
                op_index_list.append(new_op_index)
                params = np.append(params, 0)
                print(abs_gradients.sum(), op_index_list)

                self.ansatz.set_ansatz(op_index_list)
                circuit = pm.run(self.ansatz())
                self.ansatz().draw("mpl")
                plt.show()

                cost_function_instance.ansatz = circuit
                cost_function_instance.hamiltonian = h_operator.apply_layout(layout=circuit.layout)
                optimize_result= minimize(fun=cost_function_instance, x0=params, **optimizer_options)
                params = optimize_result.x
