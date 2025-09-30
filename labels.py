from enum import StrEnum


class VQEParameters(StrEnum):
    DataPrefix = "Data/"
    MetaDataPrefix = "MetaData/"
    SimulatorOptions = "SimulatorOptions"
    PresetPassManagerOptions = "PresetPassManagerOptions"
    EstimatorOptions = "EstimatorOptions"
    OptimizerOptions = "OptimizerOptions"
    IterationAxis = "IterationAxis"
    CircuitParameterAxis = "CircuitParameterAxis"
    CircuitParameters = "CircuitParameters"
    HamiltonianSuffix = "/Hamiltonian"
    Hamiltonian = f"{DataPrefix}evs{HamiltonianSuffix}"
    NIterations = "NIterations"
    AnsatzOperatorAxis = "AnsatzOperatorAxis"
    AnsatzOperators = "AnsatzOperators"
    AdaptOptions = "AdaptOptions"
