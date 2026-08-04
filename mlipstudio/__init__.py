"""Programmatic API for MLIP Studio."""

from .exceptions import (
    MLIPStudioError,
    ModelConfigurationError,
    ModelDependencyError,
    ModelLoadError,
    TaskCalculationError,
    TaskValidationError,
    UnknownModelError,
)
from .models import (
    CalculatorFactory,
    ModelSpec,
    create_calculator,
    get_model_spec,
    list_models,
)
from .tasks import (
    AtomizationCohesiveEnergyResult,
    AtomizationCohesiveEnergyTask,
    BandGapDOSResult,
    BandGapDOSTask,
    DipoleMomentResult,
    DipoleMomentTask,
    HOMOLUMOGapResult,
    HOMOLUMOGapTask,
    OptimizationResult,
    OptimizationTask,
    SinglePointResult,
    SinglePointTask,
    SpinDeterminationResult,
    SpinDeterminationTask,
    SpinStateResult,
    Task,
)

__all__ = [
    "AtomizationCohesiveEnergyResult",
    "AtomizationCohesiveEnergyTask",
    "BandGapDOSResult",
    "BandGapDOSTask",
    "CalculatorFactory",
    "DipoleMomentResult",
    "DipoleMomentTask",
    "HOMOLUMOGapResult",
    "HOMOLUMOGapTask",
    "MLIPStudioError",
    "ModelConfigurationError",
    "ModelDependencyError",
    "ModelLoadError",
    "ModelSpec",
    "OptimizationResult",
    "OptimizationTask",
    "SinglePointResult",
    "SinglePointTask",
    "SpinDeterminationResult",
    "SpinDeterminationTask",
    "SpinStateResult",
    "Task",
    "TaskCalculationError",
    "TaskValidationError",
    "UnknownModelError",
    "create_calculator",
    "get_model_spec",
    "list_models",
]

__version__ = "0.1.0.dev0"
