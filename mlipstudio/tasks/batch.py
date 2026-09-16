"""Sequential, failure-aware batch task execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Mapping, TypeVar

from ase import Atoms

from ..exceptions import TaskCalculationError, TaskValidationError
from .electronic_structure import HOMOLUMOGapResult, HOMOLUMOGapTask
from .energetics import (
    AtomizationCohesiveEnergyResult,
    AtomizationCohesiveEnergyTask,
)
from .single_point import SinglePointResult, SinglePointTask
from .base import Task


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class BatchItemResult(Generic[ResultT]):
    """The result or error associated with one input structure."""

    index: int
    label: str
    result: ResultT | None
    error: str | None
    runtime_seconds: float

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class BatchResult(Generic[ResultT]):
    """Ordered outcomes from a batch calculation."""

    items: tuple[BatchItemResult[ResultT], ...]
    runtime_seconds: float

    @property
    def successful_items(self) -> tuple[BatchItemResult[ResultT], ...]:
        return tuple(item for item in self.items if item.succeeded)

    @property
    def failed_items(self) -> tuple[BatchItemResult[ResultT], ...]:
        return tuple(item for item in self.items if not item.succeeded)

    @property
    def results(self) -> tuple[ResultT, ...]:
        return tuple(
            item.result for item in self.successful_items if item.result is not None
        )


def _materialize_structures(structures: Iterable[Atoms]) -> tuple[Atoms, ...]:
    if isinstance(structures, (str, bytes, Atoms)):
        raise TaskValidationError(
            "Batch tasks require an iterable of ase.Atoms objects, not one structure."
        )
    try:
        materialized = tuple(structures)
    except TypeError as exc:
        raise TaskValidationError(
            "Batch tasks require an iterable of ase.Atoms objects."
        ) from exc
    if not materialized:
        raise TaskValidationError("Batch tasks require at least one structure.")
    return materialized


def _label(atoms: Any, index: int) -> str:
    if isinstance(atoms, Atoms):
        supplied = atoms.info.get("source_name", atoms.info.get("name"))
        if supplied:
            return str(supplied)
        return f"{atoms.get_chemical_formula()}_{index + 1}"
    return f"structure_{index + 1}"


def _run_batch(
    structures: Iterable[Atoms],
    calculator: Any,
    task: Any,
    *,
    continue_on_error: bool,
) -> BatchResult[Any]:
    materialized = _materialize_structures(structures)
    items: list[BatchItemResult[Any]] = []
    total_started = time.perf_counter()
    for index, atoms in enumerate(materialized):
        item_started = time.perf_counter()
        try:
            result = task.calculate(atoms, calculator)
            items.append(
                BatchItemResult(
                    index=index,
                    label=_label(atoms, index),
                    result=result,
                    error=None,
                    runtime_seconds=time.perf_counter() - item_started,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise TaskCalculationError(
                    f"Batch calculation failed for item {index} ({_label(atoms, index)}): "
                    f"{exc}"
                ) from exc
            items.append(
                BatchItemResult(
                    index=index,
                    label=_label(atoms, index),
                    result=None,
                    error=f"{exc.__class__.__name__}: {exc}",
                    runtime_seconds=time.perf_counter() - item_started,
                )
            )
    return BatchResult(
        items=tuple(items),
        runtime_seconds=time.perf_counter() - total_started,
    )


@dataclass(frozen=True)
class BatchSinglePointTask(Task):
    """Calculate energy, forces, and/or stress for multiple structures."""

    properties: tuple[str, ...] = ("energy", "forces", "stress")
    continue_on_error: bool = True
    copy_atoms: bool = True

    def calculate(
        self,
        structures: Iterable[Atoms],
        calculator: Any,
    ) -> BatchResult[SinglePointResult]:
        task = SinglePointTask(properties=self.properties, copy_atoms=self.copy_atoms)
        return _run_batch(
            structures,
            calculator,
            task,
            continue_on_error=self.continue_on_error,
        )


@dataclass(frozen=True)
class BatchHOMOLUMOGapTask(Task):
    """Predict HOMO-LUMO gaps for multiple QM9-compatible molecules."""

    allowed_elements: frozenset[str] = frozenset({"H", "C", "N", "O", "F"})
    continue_on_error: bool = True
    copy_atoms: bool = True

    def calculate(
        self,
        structures: Iterable[Atoms],
        calculator: Any,
    ) -> BatchResult[HOMOLUMOGapResult]:
        task = HOMOLUMOGapTask(
            allowed_elements=self.allowed_elements,
            copy_atoms=self.copy_atoms,
        )
        return _run_batch(
            structures,
            calculator,
            task,
            continue_on_error=self.continue_on_error,
        )


@dataclass(frozen=True)
class BatchAtomizationCohesiveEnergyTask(Task):
    """Calculate atomization or cohesive energies for multiple structures."""

    calculation_type: str = "auto"
    isolated_atom_energies: Mapping[int | str, float] | None = None
    reference_set: str | None = None
    isolated_atom_cell_A: float = 20.0
    system_metadata: Mapping[str, Any] = field(
        default_factory=lambda: {
            "charge": 0,
            "total_charge": 0,
            "spin": 1,
            "total_spin": 1,
            "external_field": [0.0, 0.0, 0.0],
        }
    )
    isolated_atom_metadata: Mapping[str, Any] = field(
        default_factory=lambda: {
            "charge": 0,
            "total_charge": 0,
            "spin": 0,
            "total_spin": 0,
            "external_field": [0.0, 0.0, 0.0],
        }
    )
    continue_on_error: bool = True
    copy_atoms: bool = True

    def calculate(
        self,
        structures: Iterable[Atoms],
        calculator: Any,
    ) -> BatchResult[AtomizationCohesiveEnergyResult]:
        task = AtomizationCohesiveEnergyTask(
            calculation_type=self.calculation_type,
            isolated_atom_energies=self.isolated_atom_energies,
            reference_set=self.reference_set,
            isolated_atom_cell_A=self.isolated_atom_cell_A,
            system_metadata=self.system_metadata,
            isolated_atom_metadata=self.isolated_atom_metadata,
            copy_atoms=self.copy_atoms,
        )
        return _run_batch(
            structures,
            calculator,
            task,
            continue_on_error=self.continue_on_error,
        )


BatchEnergyForceStressTask = BatchSinglePointTask
BatchAtomizationEnergyTask = BatchAtomizationCohesiveEnergyTask
