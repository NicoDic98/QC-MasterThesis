import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt, cm, lines
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from numpyencoder import NumpyEncoder
from qiskit import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from combine_data import combine_data
from h5_interface import H5Loader, load_attribute_as_dict
from hamiltonian.base import HamiltonianParameters
from labels import VQEParameters
from misc import plots_folder, data_folder
from solver.adapt_vqe import AdaptVQE
from solver.base import GlobalParameters
from solver.circuits import CircuitParameters, rebuild_ansatz, BaseVQEAnsatz, BaseADAPTVQEAnsatz
from solver.exact_diagonalization import EDParameters, ED
from solver.variational_quantum_eigensolver import VQE


class Animator:
    def __init__(self, data: np.ndarray, stepsize: int):
        fig, ax = plt.subplots(1, 1)
        self.fig = fig
        self.ax = ax
        self.data = data
        self.stepsize = stepsize

    def __call__(self, i):
        self.ax.clear()

        point = np.abs(self.data[i])

        self.ax.plot(np.arange(len(point)), point, color='green',
                     label=f"Iteration {i * self.stepsize}", marker='o')
        self.ax.set_xlabel("Layer")
        self.ax.set_ylabel("Fidelity")
        self.ax.set_xlim((-0.5, 8.5))
        self.ax.set_ylim((-0.05, 1.05))

        self.ax.legend(loc='upper left')

    def generate_frame_array(self):
        frames = np.arange(1)
        start = 1
        step = 1
        run = True
        while run:
            end = 50 * start
            if end > len(self.data):
                end = len(self.data)
                run = False
            frames = np.append(frames, np.arange(start, end, step))
            start = end
            step *= 10
        return np.append(frames, [len(self.data) - 1] * int(len(frames) / 20))

    def animate(self, filename: str):
        frames = self.generate_frame_array()
        fps = int(len(frames) / 10)
        ani = FuncAnimation(self.fig, self, frames=frames, interval=200, repeat=False)

        # Save the animation as an animated GIF
        ani.save(filename, dpi=300, writer=PillowWriter(fps=fps))


class ResultLoader:
    def __init__(self, group: h5py.Group):
        self.group = group
        self.solver = group.attrs[GlobalParameters.SolverName]

    def get_parameter_values_from_indices(self, parameters: dict[str, int]) -> dict[str, float]:
        return {key: self.group[key][value] for key, value in parameters.items()}

    def get_observables(self, observable_name: str, parameters: dict[str, int], dependency_names: list[str],
                        final_value: bool = True):
        """
        :param observable_name: Observable name
        :param parameters: A dictionary mapping parameter names to indices in the corresponding list of parameter values
        :param dependency_names: List of dependency names, which should not be fixed to one value
        :param final_value: If true, return only the value in the final iteration
        :return: Dataset values, Corresponding dependency values, Dependency dictionary {Name: Axis}
        """
        h5_loader = H5Loader(self.group, observable_name)
        observables, dep, dep_dict = h5_loader.retrieve_dependency(parameters, dependency_names, final_value)
        return observables, dep, dep_dict

    def info_dict(self, parameters: dict[str, int]):
        parameter_values = {}
        for key, value in parameters.items():
            parameter_values[key] = self.group[key][value]

        info_dict = load_attribute_as_dict(self.group, False)
        info_dict.update(parameter_values)
        if CircuitParameters.Circuit in self.group:
            info_dict[CircuitParameters.Circuit] = load_attribute_as_dict(self.group[CircuitParameters.Circuit])
        if VQEParameters.OptimizerOptions in self.group:
            info_dict[VQEParameters.OptimizerOptions] = load_attribute_as_dict(
                self.group[VQEParameters.OptimizerOptions])
        if VQEParameters.AdaptOptions in self.group:
            info_dict[VQEParameters.AdaptOptions] = load_attribute_as_dict(self.group[VQEParameters.AdaptOptions])
        return self.group.name, info_dict

    def get_energy_mass(self, parameters: dict[str, int]):
        if self.solver == ED.__name__:
            energy, dep, _ = self.get_observables(EDParameters.EigenValues, parameters,
                                                  [HamiltonianParameters.Mass])
            energy.sort(-1)
            energy = energy[:, :]
        elif self.solver == VQE.__name__:
            energy, dep, _ = self.get_observables(VQEParameters.Hamiltonian, parameters,
                                                  [HamiltonianParameters.Mass])
        elif self.solver == AdaptVQE.__name__:
            energy, dep, _ = self.get_observables(VQEParameters.Hamiltonian, parameters,
                                                  [HamiltonianParameters.Mass])
        else:
            raise NotImplementedError
        return energy, dep[0]

    def get_energy_evolution(self, parameters: dict[str, int]):
        if self.solver == ED.__name__:
            raise NotImplementedError
        elif self.solver == VQE.__name__:
            energy, _, _ = self.get_observables(VQEParameters.Hamiltonian,
                                                parameters,
                                                [], final_value=False)
            n_iterations, _, _ = self.get_observables(VQEParameters.NIterations, parameters, [])
            energy = energy[:n_iterations + 1]
        elif self.solver == AdaptVQE.__name__:
            energy, _, _ = self.get_observables(VQEParameters.Hamiltonian,
                                                parameters,
                                                [], final_value=False)
            n_iterations, _, _ = self.get_observables(VQEParameters.NIterations, parameters, [])
            energy = energy[:n_iterations + 1]
        else:
            raise NotImplementedError
        return energy

    def get_start_iterations(self, parameters: dict[str, int]):
        if self.solver == AdaptVQE.__name__:
            start_iterations, _, _ = self.get_observables(VQEParameters.StartIterations, parameters,
                                                          [])
            start_iterations = [it for it in start_iterations if it >= 0]
        else:
            raise NotImplementedError
        return start_iterations

    def get_operator_indices(self, parameters: dict[str, int]):
        if self.solver == AdaptVQE.__name__:
            operator_indices, _, _ = self.get_observables(VQEParameters.AnsatzOperators, parameters,
                                                          [])
            operator_indices = [idx for idx in operator_indices if idx >= 0]
        else:
            raise NotImplementedError
        return operator_indices

    def map_operator_indices_to_labels(self, parameters: dict[str, int], operator_indices: list[int]):
        if self.solver == AdaptVQE.__name__:
            ansatz = self.get_ansatz(parameters)
            ansatz: BaseADAPTVQEAnsatz
            labels = []
            for opid in operator_indices:
                if opid is None:
                    labels.append(None)
                else:
                    gi, qbit, gname = ansatz.get_operator_info(opid)
                    labels.append(f"{gname}" + "$^{" + f"{qbit}" + "}$")
        else:
            raise NotImplementedError
        return labels

    def get_operator_labels(self, parameters: dict[str, int]):
        if self.solver == AdaptVQE.__name__:
            opid = self.get_operator_indices(parameters)
            labels = self.map_operator_indices_to_labels(parameters, opid)
        else:
            raise NotImplementedError
        return labels

    def get_ansatz(self, parameters: dict[str, int]) -> BaseVQEAnsatz | BaseADAPTVQEAnsatz:
        if self.solver == VQE.__name__:
            ansatz = rebuild_ansatz(self.group)
            ansatz: BaseVQEAnsatz
        elif self.solver == AdaptVQE.__name__:
            ansatz = rebuild_ansatz(self.group)
            ansatz: BaseADAPTVQEAnsatz
            operator_indices = self.get_operator_indices(parameters)
            ansatz.set_ansatz(operator_indices)
        else:
            raise NotImplementedError
        return ansatz

    def get_circuit(self, parameters: dict[str, int], final=False):
        ansatz = self.get_ansatz(parameters)
        circuit = ansatz.full_ansatz
        if self.solver == VQE.__name__:
            if final:
                raise NotImplementedError
        elif self.solver == AdaptVQE.__name__:
            if final:
                circuit_parameters, _, circuit_parameters_dep_dict = self.get_observables(
                    VQEParameters.CircuitParameters,
                    parameters, [])
                circuit_parameters = circuit_parameters[circuit_parameters != 0]
                if circuit_parameters.shape[circuit_parameters_dep_dict[VQEParameters.CircuitParameterAxis]] != len(
                        circuit.parameters):
                    raise ValueError(f"Parameters do not match circuit parameters"
                                     f"{circuit_parameters.shape[circuit_parameters_dep_dict[VQEParameters.CircuitParameterAxis]]}"
                                     f"!={len(circuit.parameters)}")
                parameter_binds = {}
                for i, p in enumerate(circuit.parameters):
                    parameter_binds[p] = np.take(circuit_parameters, i,
                                                 circuit_parameters_dep_dict[VQEParameters.CircuitParameterAxis])
                circuit.assign_parameters(parameter_binds, inplace=True)
        else:
            raise NotImplementedError
        return circuit

    def get_derivatives(self, parameters: dict[str, int], second=False):
        if self.solver == AdaptVQE.__name__:
            if second:
                observable_name = VQEParameters.AnsatzOperatorSecondDerivatives
            else:
                observable_name = VQEParameters.AnsatzOperatorDerivatives
            der, _, der_dep_dict = self.get_observables(observable_name, parameters, [])

            operator_indices = self.get_operator_indices(parameters)
            der = der[:len(operator_indices) + 1]
        else:
            raise NotImplementedError
        return der, der_dep_dict, operator_indices

    def plot_derivatives(self, parameters: dict[str, int], second=False):
        if self.solver == AdaptVQE.__name__:
            fig, ax = plt.subplots(layout='constrained')

            ansatz = self.get_ansatz(parameters)
            ansatz: BaseADAPTVQEAnsatz

            n_qbits = ansatz.num_qubits
            param = np.arange(n_qbits)
            # Colormap setup
            cmap = plt.get_cmap("Set2")
            norm = plt.Normalize(vmin=param.min(), vmax=param.max())

            der, der_dep_dict, op = self.get_derivatives(parameters, second)

            linestyle_str = ['solid', 'dotted', 'dashed', 'dashdot']
            for i in range(100):
                linestyle_str.append("solid")
            already_labeled = []
            legend_elements = []

            # iterate over different operators in the pool
            for i in range(der.shape[der_dep_dict[VQEParameters.AnsatzPoolOperatorAxis]]):
                gi, qbit, gname = ansatz.get_operator_info(i)
                if gi in already_labeled:
                    pass
                else:
                    legend_elements.append(Line2D([0], [0],
                                                  color=cmap(norm(n_qbits // 2)), linestyle=linestyle_str[gi],
                                                  label=gname))
                    already_labeled.append(gi)
                ax.plot(der.take(i, der_dep_dict[VQEParameters.AnsatzPoolOperatorAxis]),
                        color=cmap(norm(qbit)), linestyle=linestyle_str[gi], alpha=0.8)
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                                # location="top", orientation="horizontal"
                                )
            cbar.set_label("Qubit")
            selected_data_pts = [der.take(oi, der_dep_dict[VQEParameters.AnsatzPoolOperatorAxis])[i] for i, oi in
                                 enumerate(op)]
            ax.scatter(list(range(len(selected_data_pts))), selected_data_pts, label="Selected",
                       marker="o", facecolors="none", edgecolors='r')
            ax.legend(handles=legend_elements)

            labels = self.map_operator_indices_to_labels(parameters, op)
            fig_legend_elements = []
            for i, (opid, label) in enumerate(zip(op, labels)):
                gi, qbit, gname = ansatz.get_operator_info(opid)
                # noinspection PyTypeChecker
                fig_legend_elements.append(Line2D([0], [0],
                                                  color=cmap(norm(qbit)), linestyle=linestyle_str[gi],
                                                  label=f"{i}: {label}"))
            source = list(range(der.shape[der_dep_dict[VQEParameters.AnsatzOperatorAxis]]))
            target = [str(i) for i in source]
            target[-1] = ""
            ax.set_xticks(source, target)
            ax.set_ylabel("Derivative")
            if second:
                ax.set_ylabel("Second derivative")
            ax.set_xlabel("Adapt Iteration")
            fig.legend(handles=fig_legend_elements, loc='outside right upper', title="Selected operators")
            # plt.tight_layout()
        else:
            raise NotImplementedError
        return fig

    def get_ground_state(self, parameters: dict[str, int]):
        if self.solver == ED.__name__:
            energy, _, energy_dep_dict = self.get_observables(EDParameters.EigenValues, parameters, [])
            eigen_vect, _, eigen_vect_dep_dict = self.get_observables(EDParameters.EigenVectors, parameters, [])
            return np.take(eigen_vect,
                           np.argmin(energy, energy_dep_dict[EDParameters.EigenValueAxis]),
                           eigen_vect_dep_dict[EDParameters.EigenValueAxis])
        else:
            raise NotImplementedError

    def get_excited_state(self, parameters: dict[str, int]):
        if self.solver == ED.__name__:
            energy, _, energy_dep_dict = self.get_observables(EDParameters.EigenValues, parameters, [])
            eigen_vect, _, eigen_vect_dep_dict = self.get_observables(EDParameters.EigenVectors, parameters, [])
            return np.take(eigen_vect,
                           np.argsort(energy, energy_dep_dict[EDParameters.EigenValueAxis])[1],
                           eigen_vect_dep_dict[EDParameters.EigenValueAxis])
        else:
            raise NotImplementedError

    def plot_ground_state(self, parameters: dict[str, int]):
        return plot_state(self.get_ground_state(parameters))

    def plot_excited_state(self, parameters: dict[str, int]):
        return plot_state(self.get_excited_state(parameters))

    def get_overlaps(self, parameters: dict[str, int], reference_state: np.ndarray, every_n_iterations: int = 10):
        if self.solver == ED.__name__:
            raise NotImplementedError
        elif self.solver == VQE.__name__:
            circuit_parameters, _, circuit_parameters_dep_dict = self.get_observables(VQEParameters.CircuitParameters,
                                                                                      parameters, [], False)

            # Only use relevant parameters
            n_iterations, _, _ = self.get_observables(VQEParameters.NIterations, parameters, [])
            circuit_parameters = circuit_parameters[:n_iterations:every_n_iterations + 1]

            ansatz = rebuild_ansatz(self.group)
            circuit, num_state_vectors = ansatz.build_full_ansatz_with_save_points()
            if circuit_parameters.shape[circuit_parameters_dep_dict[VQEParameters.CircuitParameterAxis]] != len(
                    circuit.parameters):
                raise ValueError("Parameters do not match circuit parameters")
            parameter_binds = {}
            for i, p in enumerate(circuit.parameters):
                parameter_binds[p] = np.take(circuit_parameters, i,
                                             circuit_parameters_dep_dict[VQEParameters.CircuitParameterAxis])

            simulator_options = {
                "method": "statevector",
            }
            backend = AerSimulator(**simulator_options)
            pm = generate_preset_pass_manager(backend=backend)
            circuit = pm.run(circuit)
            result = backend.run(circuit, shots=1, parameter_binds=[parameter_binds]).result()

            over_laps = np.zeros((circuit_parameters.shape[circuit_parameters_dep_dict[VQEParameters.IterationAxis]],
                                  num_state_vectors),
                                 dtype=np.complex128)
            # reference_state = result.data(0)[f"psi_{4}"].data
            for it in range(over_laps.shape[0]):
                for state_vector_index in range(over_laps.shape[1]):
                    sv = result.data(it)[CircuitParameters.StateVectorBaseName + f"{state_vector_index}"]
                    over_laps[it, state_vector_index] = np.vdot(sv, reference_state)
            return over_laps
        else:
            raise NotImplementedError

    def save_overlap_evolution(self, parameters: dict[str, int], reference_state: np.ndarray,
                               base_filename: str,
                               every_n_iterations: int = 10):
        sel_values = self.get_parameter_values_from_indices(parameters)
        info_str = ""
        for key, value in parameters.items():
            info_str += f"_{key}_{sel_values[key]:.2f}"
        print(info_str)
        overlaps = self.get_overlaps(parameters, reference_state, every_n_iterations)
        # TODO: Save overlaps to file
        Animator(overlaps, every_n_iterations).animate(
            f"{base_filename}fidelity_evolution{info_str}.gif")


def plot_state(state: np.ndarray):
    amplitudes = np.abs(state)
    max_amplitude = np.max(amplitudes)
    fig, ax = plt.subplots(figsize=(7.5, 10))
    cmap_0 = plt.get_cmap('autumn')
    cmap_1 = plt.get_cmap('winter')
    pie_labels = [np.binary_repr(i, 8) for i in range(len(state))]

    relevant_labels = [pie_label if amplitudes[index] > 0.1 * max_amplitude else "" for index, pie_label in
                       enumerate(pie_labels)]
    wedge_labels = []
    legend_labels = []
    wedge_number = 1
    wedge_limit = 20
    for pie_label in relevant_labels:
        if pie_label:
            wedge_labels.append(f"{wedge_number}")
            legend_labels.append(lines.Line2D([], [], ls="", markersize=10, color="black",
                                              label=pie_label, marker=f"${wedge_number}$"))
            wedge_number += 1
        else:
            wedge_labels.append("")
    if wedge_number > wedge_limit:
        wedge_labels = None

    norm_phase = plt.Normalize(vmin=0, vmax=2 * np.pi)

    nx, ny = (2, 2)
    radius = 3
    x = np.arange(nx) * 10 * radius
    y = np.arange(ny) * 10 * radius
    comp_shifts = [-1.5 * radius, 1.5 * radius]

    xv, yv = np.meshgrid(x, y)

    ax.scatter(x=xv, y=yv, s=0)
    for x_ in x:
        for y_ in y:
            circle = plt.Circle((x_, y_), 3 * radius, color='b', fill=False)
            ax.add_patch(circle)
    for i in range(2):
        for j in range(4):
            site = i * 4 + j
            colors = []
            for k, bin_rep in enumerate(pie_labels):
                if bin_rep[site] == "0":
                    colors.append(cmap_0(norm_phase(np.angle(state[k]))))
                else:
                    colors.append(cmap_1(norm_phase(np.angle(state[k]))))

            center = (float(x[j // 2] + comp_shifts[j % 2]), float(y[i]))
            print(center)
            ax.pie(amplitudes, colors=colors, radius=radius, labels=wedge_labels, center=center, labeldistance=1.1,
                   rotatelabels=True)
            wedge_labels = None
    _ = ax.xaxis.set_ticks(x)
    _ = ax.yaxis.set_ticks(y)
    ax.set_xlim((float(x[0] - 4 * radius), float(x[-1] + 4 * radius)))
    ax.set_ylim((float(y[0] - 4 * radius), float(y[-1] + 4 * radius)))

    ax.set_frame_on(True)
    ax.set_aspect(1.0)

    cbar_0 = fig.colorbar(cm.ScalarMappable(norm=norm_phase, cmap=cmap_0), ax=ax,
                          orientation='horizontal', pad=0.03)
    cbar_1 = fig.colorbar(cm.ScalarMappable(norm=norm_phase, cmap=cmap_1), ax=ax,
                          orientation='horizontal', pad=0.04)
    cbar_1.set_ticks([])
    cbar_0.ax.set_ylabel('0', rotation=0)
    cbar_1.ax.set_ylabel('1', rotation=0)
    if wedge_number <= wedge_limit:
        fig.legend(loc='outside right', handles=legend_labels, title="States")
    return fig, relevant_labels


def create_filename(group: h5py.Group, parameters: dict[str, int], plot_name: str):
    output_filename = plots_folder
    output_filename += str(group.name).split("/")[-1]
    Path(output_filename).mkdir(parents=True, exist_ok=True)

    output_filename += "/"
    output_filename += plot_name

    parameter_values = {}
    for key, value in parameters.items():
        output_filename = f"{output_filename}_{key}={group[key][value]:.2f}"
        parameter_values[key] = group[key][value]

    info_dict = load_attribute_as_dict(group, False)
    info_dict.update(parameter_values)
    info_dict[CircuitParameters.Circuit] = load_attribute_as_dict(group[CircuitParameters.Circuit])
    final_info_dict = {
        group.name: info_dict
    }
    with open(output_filename + ".json", "w") as finfo:
        json.dump(final_info_dict, finfo, sort_keys=True, indent=4, cls=NumpyEncoder)
    return output_filename + ".png"


def plot_energy_gap_ed(group: h5py.Group, parameters: dict[str, int]):
    output_filename = create_filename(group, parameters, "EnergyGapED")
    h5_loader = H5Loader(group, EDParameters.EigenValues)
    energies, dep, _ = h5_loader.retrieve_dependency(parameters, [HamiltonianParameters.Mass])

    masses = dep[0]
    energies.sort(-1)
    energy_gap = energies[:, 1] - energies[:, 0]

    fig, ax = plt.subplots()
    ax.plot(masses, energy_gap)
    ax.set(xlabel='Mass', ylabel='Energy gap')
    ax.set_title("Energy gap")
    plt.savefig(output_filename)


def plot_energies(group: h5py.Group, parameters: dict[str, int], n_plot: int):
    loader = ResultLoader(group)
    output_filename = create_filename(group, parameters, f"Energies{loader.solver}")
    energies, masses = loader.get_energy_mass(parameters)
    fig, ax = plt.subplots()
    ax.plot(masses, energies)
    ax.set(xlabel='Mass', ylabel='Energies')
    ax.set_title("Energies")
    plt.savefig(output_filename)


if __name__ == "__main__":
    combine_data()
    h5_file = f"{data_folder}{datetime.now().strftime('%Y-%m-%U')}.hdf5"
    with h5py.File(h5_file, "r") as f:
        for name in f:
            print(name)
            my_group = f[name]
            my_parameters = {
            }
            # plot_energy_gap_ed(my_group, my_parameters)
            plot_energies(my_group, my_parameters, n_plot=2)
