"""Energy, force, and stress evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms

from ..exceptions import TaskCalculationError, TaskValidationError
from .base import Task, calculator_name, detached_copy, prepare_atoms


_SUPPORTED_PROPERTIES = frozenset({"energy", "forces", "stress"})


@dataclass(frozen=True)
class SinglePointResult:
    """Numerical single-point output in ASE units."""

    atoms: Atoms
    energy: float | None
    forces: np.ndarray | None
    stress: np.ndarray | None
    runtime_seconds: float
    calculator_name: str

    @property
    def max_force(self) -> float | None:
        if self.forces is None:
            return None
        return float(np.linalg.norm(self.forces, axis=1).max(initial=0.0))


@dataclass(frozen=True)
class SinglePointTask(Task):
    """Evaluate a selected set of ASE properties on one structure."""

    properties: tuple[str, ...] = ("energy",)
    copy_atoms: bool = True

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> SinglePointResult:
        requested = tuple(dict.fromkeys(self.properties))
        unknown = set(requested).difference(_SUPPORTED_PROPERTIES)
        if unknown:
            raise TaskValidationError(
                f"Unsupported single-point properties: {', '.join(sorted(unknown))}."
            )
        if not requested:
            raise TaskValidationError("At least one property must be requested.")

        working_atoms, resolved_calculator = prepare_atoms(
            atoms,
            calculator,
            copy_atoms=self.copy_atoms,
        )
        if "stress" in requested and not np.any(working_atoms.pbc):
            raise TaskValidationError("Stress requires a periodic structure.")

        started = time.perf_counter()
        try:
            energy = (
                float(working_atoms.get_potential_energy())
                if "energy" in requested
                else None
            )
            forces = (
                np.asarray(working_atoms.get_forces(), dtype=float).copy()
                if "forces" in requested
                else None
            )
            stress = (
                np.asarray(working_atoms.get_stress(), dtype=float).copy()
                if "stress" in requested
                else None
            )
        except Exception as exc:
            raise TaskCalculationError(f"Single-point calculation failed: {exc}") from exc

        return SinglePointResult(
            atoms=detached_copy(working_atoms),
            energy=energy,
            forces=forces,
            stress=stress,
            runtime_seconds=time.perf_counter() - started,
            calculator_name=calculator_name(resolved_calculator),
        )
