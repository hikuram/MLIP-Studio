"""Shared helpers for UI-independent calculation tasks."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from ..exceptions import TaskValidationError


class Task:
    """Marker base class for MLIP Studio task objects."""

    def calculate(self, atoms: Atoms, calculator: Any | None = None) -> Any:
        raise NotImplementedError


def validate_atoms(atoms: Atoms) -> None:
    """Validate the common single-structure task input contract."""

    if not isinstance(atoms, Atoms):
        raise TaskValidationError("atoms must be an ase.Atoms object.")
    if len(atoms) == 0:
        raise TaskValidationError("atoms must contain at least one atom.")


def prepare_atoms(
    atoms: Atoms,
    calculator: Any | None,
    *,
    copy_atoms: bool,
) -> tuple[Atoms, Any]:
    """Validate input and attach a calculator to an isolated working object."""

    validate_atoms(atoms)
    resolved_calculator = calculator if calculator is not None else atoms.calc
    if resolved_calculator is None:
        raise TaskValidationError(
            "No calculator was supplied and atoms.calc is not set."
        )
    working_atoms = atoms.copy() if copy_atoms else atoms
    working_atoms.calc = resolved_calculator
    return working_atoms, resolved_calculator


def detached_copy(atoms: Atoms) -> Atoms:
    """Copy a structure without retaining a heavyweight live calculator."""

    copied = atoms.copy()
    copied.calc = None
    return copied


def calculator_name(calculator: Any) -> str:
    """Return the MLIP Studio model label when available."""

    return str(
        getattr(calculator, "_mlipstudio_model_name", calculator.__class__.__name__)
    )
