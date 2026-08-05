from typing import Any, Callable

import h5py
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorOptions
from scipy.optimize import minimize

from h5_interface import save_dict_as_attribute
from hamiltonian.base import HamiltonianType, BaseHamiltonian
from labels import VQEParameters
from misc import fill_defaults_in_dict, calc_im_part
from solver.base import SimulatorType, BaseVQE, BaseCostFunction
from solver.circuits import BaseADAPTVQEAnsatz, HardwareAdaptAnsatz24


def remove_zero_operators(operator_list: list[SparsePauliOp], threshold: float = 1e-4) -> np.ndarray:
    op_sizes = np.array([op.size for op in operator_list])
    temp = np.abs(np.array([op.to_list()[0][1] for op in operator_list]))
    zero_op_indices = np.argwhere(np.logical_and(op_sizes <= 1, temp < threshold))[:, 0]
    # start deleting from the back
    for index in zero_op_indices[::-1]:
        del operator_list[index]
    return zero_op_indices


def update_operator_dataset_size(dataset: h5py.Dataset, num_operators: int, resize_index: int = -1):
    if num_operators > dataset.shape[resize_index]:
        dataset.resize(num_operators, len(dataset.shape) + resize_index)


class AdaptVQECostFunction(BaseCostFunction):
    def update_ansatz_operators_dataset(self, num_operators: int, op_index_list: list[int]):
        dataset = self.group[VQEParameters.AnsatzOperators]
        update_operator_dataset_size(dataset, num_operators)
        dataset[*self.current_non_singular_index, :num_operators] = op_index_list

    def update_ansatz_operator_derivatives_dataset(self, num_operators: int, derivatives_list: list[float]):
        dataset = self.group[VQEParameters.AnsatzOperatorDerivatives]
        update_operator_dataset_size(dataset, num_operators, -2)
        dataset[*self.current_non_singular_index, num_operators - 1, :] = derivatives_list

    def update_ansatz_operator_second_derivatives_dataset(self, num_operators: int, sec_derivatives_list: list[float]):
        dataset = self.group[VQEParameters.AnsatzOperatorSecondDerivatives]
        update_operator_dataset_size(dataset, num_operators, -2)
        dataset[*self.current_non_singular_index, num_operators - 1, :] = sec_derivatives_list

    def update_start_iterations_dataset(self, num_operators: int):
        dataset = self.group[VQEParameters.StartIterations]
        update_operator_dataset_size(dataset, num_operators)
        dataset[*self.current_non_singular_index, num_operators - 1] = self.iteration

    def update_circuit_parameters_dataset(self, num_operators: int):
        dataset = self.group[VQEParameters.CircuitParameters]
        update_operator_dataset_size(dataset, num_operators)


class AdaptVQE(BaseVQE):
    ansatz: BaseADAPTVQEAnsatz

    def __init__(self,
                 hamiltonian_factory: Callable[..., BaseHamiltonian],
                 save_group: h5py.Group,
                 ansatz: BaseADAPTVQEAnsatz):
        super().__init__(hamiltonian_factory, save_group, ansatz)
        self.cost_function = AdaptVQECostFunction

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
            "prec_cutoff": 1e-3,
            "max_depth": 20,
            "gradient_hamiltonian_type": hamiltonian_type,
            "precision": 0,
            "parameter_prec_cutoff": 1e-3,
            "allow_duplicate_gates": False,
            "SelSeed": 42,
        }
        do_random_select = ("SelSeed" in adapt_options)

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

        h5_saver.create_dataset_with_dim_labels(VQEParameters.AnsatzOperatorDerivatives,
                                                [0, len(self.ansatz.operator_pool)],
                                                [VQEParameters.AnsatzOperatorAxis,
                                                 VQEParameters.AnsatzPoolOperatorAxis],
                                                float,
                                                [None, len(self.ansatz.operator_pool)]
                                                )

        h5_saver.create_dataset_with_dim_labels(VQEParameters.AnsatzOperatorSecondDerivatives,
                                                [0, len(self.ansatz.operator_pool)],
                                                [VQEParameters.AnsatzOperatorAxis,
                                                 VQEParameters.AnsatzPoolOperatorAxis],
                                                float,
                                                [None, len(self.ansatz.operator_pool)]
                                                )

        # Set file into single writer multiple reader mode:
        # https://docs.h5py.org/en/latest/swmr.html
        local_group.file.swmr_mode = True

        sel_rng = np.random.default_rng(seed=adapt_options["SelSeed"])

        for parameters, non_singular_index in zip(h5_saver.parameters_list_dict, h5_saver.non_singular_indices_list):
            hamiltonian = self.hamiltonian_factory(**parameters)
            print(f"Calculating energies for {hamiltonian}")
            h_operator = hamiltonian.hamiltonian_op(hamiltonian_type)
            print("h_im:", calc_im_part(h_operator))
            temp = hamiltonian.hamiltonian_op(adapt_options["gradient_hamiltonian_type"])
            commutator_list = [(temp @ op - op @ temp).simplify() for op in self.ansatz.operator_pool]
            sec_commutator_list = [(op2 @ op1 - op1 @ op2).simplify()
                                   for op1, op2 in zip(self.ansatz.operator_pool, commutator_list)]

            zero_commutator_indices = remove_zero_operators(commutator_list)
            zero_sec_commutator_indices = remove_zero_operators(sec_commutator_list)

            # l0 = np.array([op.to_list()[0] for op in self.ansatz.operator_pool])
            # l1 = np.array([op.size for op in commutator_list])
            # l2 = np.array([op.size for op in sec_commutator_list])
            # l3 = np.array([op.to_list()[0] for op in commutator_list])
            # l4 = np.array([op.to_list()[0] for op in sec_commutator_list])
            # print(l0[l1<2])
            # print(l0[l2<2])
            # print(np.argwhere(l1<2))
            # print(np.argwhere(l2<2))
            #
            # print(l1[l1<2])
            # print(l3[l1<2])
            #
            # print(l2[l2<2])
            # print(l4[l2<2])
            # commutator_list = [op for op in commutator_list if op.size>=2]
            # sec_commutator_list = [op for op in sec_commutator_list if op.size>=2]

            # print("[op]:\n",self.ansatz.operator_pool)
            # print("[,]:\n",commutator_list)

            self.ansatz.set_ansatz([], update_last_parameter_on_qubit=True)
            circuit = pm.run(self.ansatz())
            params = x0
            op_index_list = []
            # print("Prec", estimator.default_precision)
            cost_function_instance = self.cost_function(circuit, h_operator.apply_layout(circuit.layout),
                                                        estimator, local_group, non_singular_index)
            # ref_f = 0
            print("Lets GO!", flush=True)
            for i in range(adapt_options["max_depth"]):
                local_group.file.flush()

                # f = np.zeros(len(self.ansatz.operator_pool))
                # new_params = []
                # for oi, op in enumerate(self.ansatz.operator_pool):
                #     # pis = self.ansatz.get_previous_parameter_indices(oi)
                #     if len(params) <=4:
                #         pis = list(range(len(params)))
                #     else:
                #         pis = list(range(len(params)-4,len(params)))
                #     pis.append(len(params))
                #     offsets = np.zeros(3)
                #     offsets[1] = 2 * np.pi / 3
                #     offsets[2] = -2 * np.pi / 3
                #     temp1 = np.array(np.meshgrid(*([offsets] * len(pis)), indexing="ij")).T
                #     temp_shape = temp1.shape[:-1]
                #     temp1 = temp1.reshape(-1, len(pis))
                #     temp2 = np.append(params, 0)
                #     temp_params = np.tile(temp2, [3 ** len(pis), 1])
                #     for j, pi in enumerate(pis):
                #         temp_params[:, pi] = temp1[:, j]
                #
                #     self.ansatz.set_ansatz([*op_index_list, oi], False)
                #     temp_circuit = pm.run(self.ansatz())
                #
                #     # print("=" * 40)
                #     # print(temp_params.shape)
                #     pub = (temp_circuit, h_operator.apply_layout(layout=temp_circuit.layout), temp_params)
                #     job = estimator.run(pubs=[pub], precision=adapt_options["precision"])
                #     full_result = job.result()
                #     pub_result = full_result[0]
                #     # print(pub_result.data.evs.reshape(temp_shape))
                #     temp_x0 = temp1[np.argmin(pub_result.data.evs)]
                #     # print(temp_x0)
                #     # print(pub_result.data.evs.shape)
                #     # print("=" * 40)
                #
                #
                #     temp_params = np.append(params, 0)
                #
                #     def opt_f(x: np.ndarray):
                #         for rpi, pi in enumerate(pis):
                #             temp_params[pi] = x[rpi]
                #         pub = (temp_circuit, [h_operator.apply_layout(layout=temp_circuit.layout)], [temp_params])
                #         job = estimator.run(pubs=[pub], precision=adapt_options["precision"])
                #         full_result = job.result()
                #         pub_result = full_result[0]
                #         return pub_result.data.evs.item()
                #
                #     temp_optimize_result = minimize(fun=opt_f, x0=temp_x0, method="slsqp",
                #                                     options={"maxiter": 20000})
                #     temp_x = temp_optimize_result.x
                #     f[oi] = temp_optimize_result.fun
                #     for rpi, pi in enumerate(pis):
                #         temp_params[pi] = temp_x[rpi]
                #     new_params.append(temp_params)
                #
                # self.ansatz.set_ansatz(op_index_list, False)
                # if i == 0:
                #     ref_f = np.max(f) + 42
                # f-=ref_f

                pub = (circuit, [op.apply_layout(circuit.layout) for op in commutator_list], params)
                # noinspection PyTypeChecker
                job = estimator.run(pubs=[pub], precision=adapt_options["precision"])
                full_result = job.result()
                pub_result = full_result[0]
                gradients = pub_result.data.evs

                pub = (circuit, [op.apply_layout(circuit.layout) for op in sec_commutator_list], params)
                # noinspection PyTypeChecker
                job = estimator.run(pubs=[pub], precision=adapt_options["precision"])
                full_result = job.result()
                pub_result = full_result[0]
                second_gradients = pub_result.data.evs

                # Reinsert 0 operators:
                gradients = np.insert(gradients, zero_commutator_indices, 0)
                second_gradients = np.insert(second_gradients, zero_sec_commutator_indices, 0)

                # atan2(0,0)=0
                b = np.atan2(gradients, -second_gradients)
                f = np.sqrt(gradients ** 2 + second_gradients ** 2) - second_gradients

                # f = []
                # for k in range(len(gradients)):
                #     if np.abs(np.sin(b[k])) > adapt_options["precision"]:
                #         f.append(-second_gradients[k] + (gradients[k] / np.sin(b[k])))
                #     else:
                #         f.append(-second_gradients[k] - (second_gradients[k] / np.cos(b[k])))
                #
                # f = np.array(f)

                abs_gradients = np.abs(gradients)
                # if do_random_select:
                #     new_op_index_opts = np.argwhere(abs_gradients >= abs_gradients.max())[:, 0]
                #     new_op_index = sel_rng.choice(new_op_index_opts)
                # else:
                #     new_op_index = np.argmax(abs_gradients)
                # new_op_index = np.argmin(f)

                new_op_index = np.argmax(f)

                # prefer smaller operators if they result in a significant reduction of energy
                if isinstance(self.ansatz, HardwareAdaptAnsatz24):
                    print("Special HardwareAdaptAnsatz24 selection activated")
                    prev_end = 0
                    for gate_size in range(1, self.ansatz.num_qubits + 1):
                        block_size = (3 ** gate_size)*(self.ansatz.num_qubits - (gate_size - 1))
                        print(f"Checking gate size {gate_size} with block size {block_size}.")
                        f_subset = f[prev_end:block_size]
                        if f_subset.sum() < 1e-3:
                            prev_end += block_size
                            continue
                        else:
                            new_op_index = prev_end + np.argmax(f_subset)
                            print(f"Choose gate size {gate_size}")
                            break

                _, qbits, gname = self.ansatz.get_operator_info(int(new_op_index), True)
                operator_name = self.ansatz.operator_pool[int(new_op_index)].to_list()[0][0]
                print(f"Current gradient magnitude: {abs_gradients.sum()}")
                print(f"New op index: {new_op_index}\t{operator_name}\t{gname}{qbits}", flush=True)
                # print(ref_f, "\n", f)
                print(f"gradient: {gradients[new_op_index]}")
                print(f"second_gradient: {second_gradients[new_op_index]}")
                print(f"b: {b[new_op_index]}")
                print(f"f: {f[new_op_index]}")
                # print(b, "\n", f)
                initial_parameter_value = np.pi + b[new_op_index]
                # initial_parameter_value = 0

                # Note that the derivative dataset will have one more entry as long as the depth limit is not reached
                cost_function_instance.update_ansatz_operator_derivatives_dataset(len(op_index_list) + 1, gradients)
                cost_function_instance.update_ansatz_operator_second_derivatives_dataset(len(op_index_list) + 1,
                                                                                         second_gradients)

                # if i == 0:
                #     for op_id, (grad, secgrad) in enumerate(zip(gradients, second_gradients)):
                #         print(f"{op_id}:\t{grad:.2e}\t{secgrad:.2e}")
                #     if abs_gradients.sum() < adapt_options["prec_cutoff"]:
                #         for op_id, secgrad in enumerate(second_gradients):
                #             if secgrad < -adapt_options["prec_cutoff"]:
                #                 new_op_index = op_id
                #                 initial_parameter_value = np.pi
                #                 print(f"Selecting maximum: {new_op_index} with second gradient of {secgrad}")
                #                 break
                # else:
                #
                #     if abs_gradients.sum() < adapt_options["prec_cutoff"]:
                #         print("Precision cutoff reached.")
                #         break

                if f.sum() < adapt_options["prec_cutoff"]:
                    print("Precision cutoff reached.")
                    break

                op_index_list.append(new_op_index)
                params = np.append(params, initial_parameter_value)
                # params = new_params[new_op_index]

                if self.ansatz.set_ansatz(op_index_list, not adapt_options["allow_duplicate_gates"], True):
                    print("Adding the same operator twice is not sensible, terminating.")
                    break
                circuit = pm.run(self.ansatz())

                cost_function_instance.update_ansatz_operators_dataset(len(op_index_list), op_index_list)
                cost_function_instance.update_start_iterations_dataset(len(op_index_list))
                cost_function_instance.update_circuit_parameters_dataset(len(op_index_list))

                cost_function_instance.ansatz = circuit
                # circuit.draw("text", filename="out/test.txt")
                cost_function_instance.hamiltonian = h_operator.apply_layout(layout=circuit.layout)
                optimize_result = minimize(fun=cost_function_instance, x0=params, **optimizer_options)
                if not optimize_result.success:
                    print(f"Optimization failed: {optimize_result.message}")
                params = optimize_result.x
                # ref_f = optimize_result.fun
                if np.abs(params[-1]) < adapt_options["parameter_prec_cutoff"]:
                    print("Parameter value cutoff reached.")
                    break
            # self.ansatz().draw("mpl")
            # plt.show()
