"""Public exception hierarchy for the MLIP Studio Python API."""


class MLIPStudioError(Exception):
    """Base class for errors raised by the programmatic API."""


class UnknownModelError(MLIPStudioError, ValueError):
    """Raised when a model name is not present in the MLIP Studio catalog."""


class ModelConfigurationError(MLIPStudioError, ValueError):
    """Raised when model parameters are missing or invalid."""


class ModelDependencyError(MLIPStudioError, ImportError):
    """Raised when the optional package for a model family is unavailable."""


class ModelLoadError(MLIPStudioError, RuntimeError):
    """Raised when a supported model cannot be initialized."""


class TaskValidationError(MLIPStudioError, ValueError):
    """Raised when a task cannot run with the supplied structure or options."""


class TaskCalculationError(MLIPStudioError, RuntimeError):
    """Raised when an otherwise valid task fails during calculation."""
