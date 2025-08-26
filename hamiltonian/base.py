from enum import Enum, auto, StrEnum
from typing import Callable

from qiskit.quantum_info import SparsePauliOp


class HamiltonianType(Enum):
    Full = auto()
    ZeroChargePenalty = auto()
    ZeroChargeProjection = auto()


class HamiltonianParameters(StrEnum):
    XExtend = "n_x"
    YExtend = "n_y"
    ZExtend = "n_z"
    Mass = "mass"
    WilsonParameter = "r"


class BaseHamiltonian:
    field: Callable[..., SparsePauliOp]
    j: Callable[..., int]

    def __init__(self,
                 field: Callable[..., SparsePauliOp],
                 j: Callable[..., int], ):
        self.field = field
        self.j = j

    def __str__(self):
        return "BaseHamiltonian"

    def size_of_zero_charge_sector(self) -> int:
        return 42

    def full_hamiltonian(self) -> SparsePauliOp:
        return self.field()

    def zero_charge_penalized_hamiltonian(self) -> SparsePauliOp:
        return self.full_hamiltonian()

    def zero_charge_projected_hamiltonian(self) -> SparsePauliOp:
        return self.full_hamiltonian()

    def hamiltonian_op(self, hamiltonian_type: HamiltonianType) -> SparsePauliOp:
        if hamiltonian_type == HamiltonianType.Full:
            return self.full_hamiltonian()
        elif hamiltonian_type == HamiltonianType.ZeroChargePenalty:
            return self.zero_charge_penalized_hamiltonian()
        elif hamiltonian_type == HamiltonianType.ZeroChargeProjection:
            return self.zero_charge_projected_hamiltonian()

    def size_of_hamiltonian(self, hamiltonian_type: HamiltonianType) -> int:
        return self.hamiltonian_op(hamiltonian_type).to_matrix(sparse=True).shape[0]

    @classmethod
    def build_hamiltonian(cls, *args, **kwargs):
        return cls(*args, **kwargs)
