"""Geometry and cell optimization tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, BFGSLineSearch, FIRE, GPMin, LBFGS, LBFGSLineSearch, MDMin

from ..exceptions import TaskCalculationError, TaskValidationError
from .base import Task, calculator_name, detached_copy, prepare_atoms


_OPTIMIZERS = {
    "BFGS": BFGS,
    "BFGSLineSearch": BFGSLineSearch,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "FIRE": FIRE,
    "GPMin": GPMin,
    "MDMin": MDMin,
}


def _maximum_force(optimizable: Any) -> float:
    forces = np.asarray(optimizable.get_forces(), dtype=float)
    if forces.size == 0:
        return 0.0
    return float(np.linalg.norm(forces.reshape((-1, 3)), axis=1).max(initial=0.0))


@dataclass(frozen=True)
class OptimizationResult:
    """Raw optimization output in ASE units."""

    atoms: Atoms
    trajectory: tuple[Atoms, ...]
    energy: float
    forces: np.ndarray
    converged: bool
    steps: int
    optimizer: str
    optimize_cell: bool
    optimizer_max_force: float
    atomic_max_force: float
    runtime_seconds: float
    calculator_name: str


@dataclass(frozen=True)
class OptimizationTask(Task):
    """Run a standard ASE geometry or full-cell optimization."""

    optimizer: str = "LBFGS"
    fmax: float = 0.01
    steps: int = 50
    optimize_cell: bool = False
    optimizer_parameters: Mapping[str, Any] = field(default_factory=dict)
    logfile: Any = None
    copy_atoms: bool = True
    record_trajectory: bool = True

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> OptimizationResult:
        if self.optimizer not in _OPTIMIZERS:
            raise TaskValidationError(
                f"Unknown optimizer '{self.optimizer}'. Available: "
                f"{', '.join(_OPTIMIZERS)}."
            )
        if self.fmax <= 0:
            raise TaskValidationError("fmax must be greater than zero.")
        if self.steps < 0:
            raise TaskValidationError("steps must be non-negative.")

        working_atoms, resolved_calculator = prepare_atoms(
            atoms,
            calculator,
            copy_atoms=self.copy_atoms,
        )
        if self.optimize_cell:
            if not np.all(working_atoms.pbc):
                raise TaskValidationError(
                    "Cell optimization requires periodic boundary conditions in all directions."
                )
            if working_atoms.cell.volume <= 0:
                raise TaskValidationError("Cell optimization requires a valid non-zero cell.")
            optimizable: Any = FrechetCellFilter(working_atoms)
        else:
            optimizable = working_atoms

        kwargs = dict(self.optimizer_parameters)
        kwargs.setdefault("logfile", self.logfile)
        frames: list[Atoms] = []
        if self.record_trajectory:
            frames.append(detached_copy(working_atoms))

        try:
            optimizer = _OPTIMIZERS[self.optimizer](optimizable, **kwargs)
            if self.record_trajectory:
                optimizer.attach(
                    lambda: frames.append(detached_copy(working_atoms)),
                    interval=1,
                )
            started = time.perf_counter()
            converged = bool(optimizer.run(fmax=float(self.fmax), steps=int(self.steps)))
            runtime = time.perf_counter() - started
            energy = float(working_atoms.get_potential_energy())
            forces = np.asarray(working_atoms.get_forces(), dtype=float).copy()
            optimizer_max_force = _maximum_force(optimizable)
            atomic_max_force = _maximum_force(working_atoms)
        except Exception as exc:
            raise TaskCalculationError(f"Optimization failed: {exc}") from exc

        return OptimizationResult(
            atoms=detached_copy(working_atoms),
            trajectory=tuple(frames),
            energy=energy,
            forces=forces,
            converged=converged,
            steps=optimizer.get_number_of_steps(),
            optimizer=self.optimizer,
            optimize_cell=self.optimize_cell,
            optimizer_max_force=optimizer_max_force,
            atomic_max_force=atomic_max_force,
            runtime_seconds=runtime,
            calculator_name=calculator_name(resolved_calculator),
        )
