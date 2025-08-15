from collections.abc import Callable

import numpy as np
from scipy.special import comb
from qiskit.quantum_info import SparsePauliOp


def alpha_1(m_x: int) -> float:
    """
    :param m_x: m_x
    :return: (-1)^m_x
    """
    if m_x % 2 == 0:
        return 1.0
    else:
        return -1.0


def alpha_2(m_x: int) -> float:
    """
    :param m_x: m_x
    :return: (1-(-1)^m_x)/2
    """
    if m_x % 2 == 0:
        return 0.0
    else:
        return 1.0


def alpha_3(m_x: int) -> float:
    """
    :param m_x: m_x
    :return: (1+(-1)^m_x)/2
    """
    if m_x % 2 == 0:
        return 1.0
    else:
        return 0.0


def d_func(n_y: int, i: int, alpha: Callable[[int], float]) -> SparsePauliOp:
    pass


class FreeWilson2D:
    def __init__(self, n_x: int, n_y: int, mass: float, r: float = 1.):
        """

        :param n_x: Lattice extend in x direction
        :param n_y: Lattice extend in y direction
        :param mass: Bare mass
        :param r: Wilson parameter
        """
        self.N_x = n_x
        self.N_y = n_y
        self.mass = mass
        self.r = r
        self.M_x = 2 * n_x
        self.N_sites = self.M_x * self.N_y
        self.j = self.snake_j

    def snake_j(self, m_x: int, n_y: int) -> int:
        if n_y % 2 == 0:
            return self.M_x * n_y + m_x
        else:
            return self.M_x * (n_y + 1) - (m_x + 1)

    def field(self, m_x: int, n_y: int, adjoint: bool = False) -> SparsePauliOp:
        """
        Note:
        With this ordering the site j=0, corresponds to the last qubit in the circuit
        due to the default ordering in qiskit:
        (Taken from https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Pauli#pauli )
        In the string representation qubit-0 corresponds to the right-most Pauli character,
        and qubit-(n−1) to the left-most Pauli character.
        For example 'XYZ' represents X⊗Y⊗Z with 'Z' on qubit-0, 'Y' on qubit-1, and 'X' on qubit-2.
        :param m_x: X-coordinate of site
        :param n_y: Y-coordinate of site
        :param adjoint: Whether to return the adjoint field
        :return:
        """
        m_x = m_x % self.M_x
        n_y = n_y % self.N_y
        j = self.j(m_x, n_y)
        prefix = j * "Z"
        postfix = (self.N_sites - j - 1) * "I"
        x_string = prefix + "X" + postfix
        y_string = adjoint * "-" + "i" + prefix + "Y" + postfix
        return SparsePauliOp([x_string, y_string], [0.5,0.5])

    def full_hamiltonian(self) -> SparsePauliOp:
        a = SparsePauliOp(self.N_sites * "I", 0)
        b = SparsePauliOp(self.N_sites * "I", 0)
        c = SparsePauliOp(self.N_sites * "I", 0)
        for m_x in range(self.M_x):
            for n_y in range(self.N_y):
                a += alpha_1(m_x) * (self.field(m_x, n_y, True) @ self.field(m_x, n_y, False))
                b += alpha_1(m_x) * (self.field(m_x, n_y, True) @ self.field(m_x + 2, n_y, False)
                                     + self.field(m_x, n_y, True) @ self.field(m_x, n_y + 1, False))
                c += (alpha_2(m_x) * (-1j * self.field(m_x, n_y, True) @ self.field(m_x + 1, n_y, False)
                                      - self.field(m_x, n_y, True) @ self.field(m_x - 1, n_y + 1, False))
                      + alpha_3(m_x) * (-1j * self.field(m_x, n_y, True) @ self.field(m_x + 3, n_y, False)
                                        + self.field(m_x, n_y, True) @ self.field(m_x + 1, n_y + 1, False)))
        return (-0.5 * (c + c.adjoint() + self.r * (b + b.adjoint())) + (self.mass + 2 * self.r) * a).simplify()

    def size_of_zero_charge_sector(self) -> int:
        pen_operator = self.zero_charge_penalty_term().simplify()
        pen_matrix = pen_operator.to_matrix()
        pen_list = pen_matrix.diagonal()
        zeros = np.argwhere(np.isclose(pen_list, np.zeros_like(pen_list)))[:,0]
        # print([np.binary_repr(zero).count("1") for zero in zeros])
        if len(zeros)!= comb(self.N_sites, self.N_x*self.N_y):
            raise ValueError("Something went wrong.")
        return len(zeros)

    def zero_charge_penalty_term(self) -> SparsePauliOp:
        ret = SparsePauliOp(self.N_sites * "I", 0)
        # build number operator
        for m_x in range(self.M_x):
            for n_y in range(self.N_y):
                ret += self.field(m_x, n_y, True) @ self.field(m_x, n_y, False)
        # shift to have positive and negative charge
        ret -= SparsePauliOp(self.N_sites * "I", self.N_x*self.N_y)
        # square, s.t. everything except zero charge is positive
        return (ret@ret).simplify()

    def zero_charge_penalized_hamiltonian(self, penalty_factor: float = 100) -> SparsePauliOp:
        return (self.full_hamiltonian() + penalty_factor*self.zero_charge_penalty_term()).simplify()

    def zero_charge_projector(self, build_from_fields: bool = False) -> SparsePauliOp:
        if build_from_fields:
            pen_operator = self.zero_charge_penalty_term()
            penalty_eigenvalues = np.unique(pen_operator.to_matrix().diagonal())
            nonzero_eigenvalues = penalty_eigenvalues[np.nonzero(penalty_eigenvalues)]
            ret = SparsePauliOp(self.N_sites * "I", 1)
            """
            Based on:
            https://physics.stackexchange.com/questions/181105/how-do-you-find-the-projection-operator-onto-an-eigenspace-if-you-dont-know-the
            """
            for ev in np.rint(nonzero_eigenvalues.real):
                ret = ret@(-pen_operator/ev+SparsePauliOp(self.N_sites * "I", 1))
        else:
            ret = SparsePauliOp(self.N_sites * "I", 0)
            list_of_pauli_strings = ["I", "Z"]
            while len(list_of_pauli_strings) < 2**self.N_sites:
                old_list = list_of_pauli_strings.copy()
                list_of_pauli_strings = [local_str + "I" for local_str in old_list]
                list_of_pauli_strings.extend([local_str + "Z" for local_str in old_list])
            for i in range(2**self.N_sites):
                bin_i = np.binary_repr(i, self.N_sites)
                n_ones = bin_i.count("1")
                if n_ones == self.N_x*self.N_y:
                    signs = np.ones(2**self.N_sites)
                    for qubit in range(self.N_sites):
                        digit = bin_i[qubit]
                        if digit == "1":
                            signs*=np.array([+1 if pauli_string[qubit]=="I" else -1 for pauli_string in list_of_pauli_strings ])
                        else:
                            pass
                    ret += SparsePauliOp(list_of_pauli_strings, signs/(2**self.N_sites))
        return ret.simplify()

    def zero_charge_projected_hamiltonian(self, build_from_fields: bool = False) -> SparsePauliOp:
        return (self.zero_charge_projector(build_from_fields)@self.full_hamiltonian()@self.zero_charge_projector(build_from_fields)).simplify()
