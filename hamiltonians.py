from collections.abc import Callable

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
        m_x = m_x % self.M_x
        n_y = n_y % self.N_y
        j = self.j(m_x, n_y)
        prefix = j * "Z"
        postfix = (self.N_sites - j - 1) * "I"
        return SparsePauliOp([prefix + "X" + postfix, adjoint * "-" + "i" + prefix + "Y" + postfix])

    def hamiltonian(self) -> SparsePauliOp:
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
        a = a.simplify()
        b = b.simplify()
        c = c.simplify()
        return -0.5 * (c + c.adjoint() + self.r * (b + b.adjoint())) + (self.mass + 2 * self.r) * a
