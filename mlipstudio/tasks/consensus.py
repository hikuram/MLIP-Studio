"""UI-independent model-consensus calculations."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from ase import Atoms

from ..exceptions import TaskCalculationError, TaskValidationError
from .base import Task, calculator_name, detached_copy, validate_atoms


@dataclass(frozen=True)
class ConsensusPrediction:
    """One model's prediction, including a readable failure when applicable."""

    label: str
    calculator_name: str
    energy_eV: float | None
    forces_eV_per_A: np.ndarray | None
    stress_eV_per_A3: np.ndarray | None
    runtime_seconds: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ModelConsensusResult:
    """Raw model predictions and disagreement statistics."""

    atoms: Atoms
    predictions: tuple[ConsensusPrediction, ...]
    successful_predictions: tuple[ConsensusPrediction, ...]
    mean_energy_eV: float
    mean_energy_eV_per_atom: float
    energy_standard_deviation_eV_per_atom: float
    energy_range_eV_per_atom: float
    mean_forces_eV_per_A: np.ndarray
    force_disagreement_eV_per_A: np.ndarray
    pairwise_force_rmse_eV_per_A: np.ndarray
    pairwise_labels: tuple[str, ...]
    mean_stress_eV_per_A3: np.ndarray | None
    within_tolerance: bool | None
    runtime_seconds: float

    @property
    def max_force_disagreement_eV_per_A(self) -> float:
        return float(self.force_disagreement_eV_per_A.max(initial=0.0))


@dataclass(frozen=True)
class ModelConsensusTask(Task):
    """Compare energies and forces from two or more calculators.

    ``calculators`` can be a ``label -> calculator`` mapping or a sequence.
    A value with a ``create()`` method (for example ``CalculatorFactory``) is
    constructed only for its own prediction and then released, limiting peak
    memory use. A zero-argument callable factory is supported as well.
    """

    include_stress: bool = True
    continue_on_error: bool = True
    energy_tolerance_eV_per_atom: float | None = None
    force_tolerance_eV_per_A: float | None = None
    system_metadata: Mapping[str, Any] = field(default_factory=dict)
    copy_atoms: bool = True

    @staticmethod
    def _entries(
        calculators: Mapping[str, Any] | Sequence[Any],
    ) -> tuple[tuple[str, Any], ...]:
        if isinstance(calculators, Mapping):
            entries = tuple((str(label), provider) for label, provider in calculators.items())
        elif isinstance(calculators, Sequence) and not isinstance(
            calculators, (str, bytes)
        ):
            generated: list[tuple[str, Any]] = []
            counts: dict[str, int] = {}
            for provider in calculators:
                base_label = str(
                    getattr(
                        provider,
                        "model_name",
                        getattr(provider, "_mlipstudio_model_name", provider.__class__.__name__),
                    )
                )
                counts[base_label] = counts.get(base_label, 0) + 1
                suffix = f" #{counts[base_label]}" if counts[base_label] > 1 else ""
                generated.append((base_label + suffix, provider))
            entries = tuple(generated)
        else:
            raise TaskValidationError(
                "calculators must be a label-to-calculator mapping or a sequence."
            )
        if len(entries) < 2:
            raise TaskValidationError("Model consensus requires at least two calculators.")
        if any(not label.strip() for label, _ in entries):
            raise TaskValidationError("Consensus calculator labels cannot be empty.")
        if len({label for label, _ in entries}) != len(entries):
            raise TaskValidationError("Consensus calculator labels must be unique.")
        return entries

    @staticmethod
    def _create_calculator(provider: Any) -> tuple[Any, bool]:
        create = getattr(provider, "create", None)
        if callable(create):
            return create(), True
        if callable(provider) and not callable(
            getattr(provider, "get_potential_energy", None)
        ):
            return provider(), True
        return provider, False

    def calculate(
        self,
        atoms: Atoms,
        calculators: Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> ModelConsensusResult:
        validate_atoms(atoms)
        if calculators is None:
            raise TaskValidationError("ModelConsensusTask requires calculators.")
        if self.energy_tolerance_eV_per_atom is not None and (
            self.energy_tolerance_eV_per_atom < 0
            or not np.isfinite(self.energy_tolerance_eV_per_atom)
        ):
            raise TaskValidationError(
                "energy_tolerance_eV_per_atom must be finite and non-negative."
            )
        if self.force_tolerance_eV_per_A is not None and (
            self.force_tolerance_eV_per_A < 0
            or not np.isfinite(self.force_tolerance_eV_per_A)
        ):
            raise TaskValidationError(
                "force_tolerance_eV_per_A must be finite and non-negative."
            )

        entries = self._entries(calculators)
        predictions: list[ConsensusPrediction] = []
        result_atoms = atoms.copy()
        result_atoms.info.update(dict(self.system_metadata))
        total_started = time.perf_counter()
        for label, provider in entries:
            calculator: Any = None
            working_atoms: Atoms | None = None
            was_created = False
            prediction_started = time.perf_counter()
            try:
                calculator, was_created = self._create_calculator(provider)
                if calculator is None:
                    raise TaskValidationError("the calculator provider returned None")
                working_atoms = atoms.copy() if self.copy_atoms else atoms
                working_atoms.info.update(dict(self.system_metadata))
                working_atoms.calc = calculator
                energy = float(working_atoms.get_potential_energy())
                forces = np.asarray(working_atoms.get_forces(), dtype=float).copy()
                if not np.isfinite(energy):
                    raise ValueError("energy is not finite")
                if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
                    raise ValueError(
                        f"forces must be finite with shape {(len(atoms), 3)}"
                    )
                stress: np.ndarray | None = None
                if self.include_stress and np.any(working_atoms.pbc):
                    try:
                        candidate = np.asarray(working_atoms.get_stress(), dtype=float).copy()
                        if np.all(np.isfinite(candidate)):
                            stress = candidate
                    except Exception:
                        stress = None
                predictions.append(
                    ConsensusPrediction(
                        label=label,
                        calculator_name=calculator_name(calculator),
                        energy_eV=energy,
                        forces_eV_per_A=forces,
                        stress_eV_per_A3=stress,
                        runtime_seconds=time.perf_counter() - prediction_started,
                    )
                )
            except Exception as exc:
                name = (
                    calculator_name(calculator)
                    if calculator is not None
                    else str(getattr(provider, "model_name", provider.__class__.__name__))
                )
                predictions.append(
                    ConsensusPrediction(
                        label=label,
                        calculator_name=name,
                        energy_eV=None,
                        forces_eV_per_A=None,
                        stress_eV_per_A3=None,
                        runtime_seconds=time.perf_counter() - prediction_started,
                        error=f"{exc.__class__.__name__}: {exc}",
                    )
                )
                if not self.continue_on_error:
                    raise TaskCalculationError(
                        f"Consensus calculation failed for '{label}': {exc}"
                    ) from exc
            finally:
                if was_created:
                    if working_atoms is not None:
                        working_atoms.calc = None
                    del calculator
                    gc.collect()

        successful = tuple(prediction for prediction in predictions if prediction.succeeded)
        if len(successful) < 2:
            failures = "; ".join(
                f"{prediction.label}: {prediction.error}"
                for prediction in predictions
                if prediction.error
            )
            suffix = f" Failures: {failures}" if failures else ""
            raise TaskCalculationError(
                "Model consensus requires at least two successful energy/force "
                f"predictions; got {len(successful)}.{suffix}"
            )

        energies = np.asarray(
            [prediction.energy_eV for prediction in successful], dtype=float
        )
        forces = np.stack(
            [prediction.forces_eV_per_A for prediction in successful], axis=0
        )
        mean_forces = np.mean(forces, axis=0)
        force_disagreement = np.sqrt(
            np.mean(np.sum((forces - mean_forces[None, :, :]) ** 2, axis=2), axis=0)
        )
        pairwise = np.zeros((len(successful), len(successful)), dtype=float)
        for row in range(len(successful)):
            for column in range(row + 1, len(successful)):
                rmse = float(np.sqrt(np.mean((forces[row] - forces[column]) ** 2)))
                pairwise[row, column] = pairwise[column, row] = rmse

        available_stresses = [
            prediction.stress_eV_per_A3
            for prediction in successful
            if prediction.stress_eV_per_A3 is not None
        ]
        mean_stress = (
            np.mean(np.stack(available_stresses, axis=0), axis=0)
            if len(available_stresses) == len(successful)
            else None
        )
        atom_count = len(atoms)
        energy_std = float(np.std(energies) / atom_count)
        energy_range = float(np.ptp(energies) / atom_count)
        tolerance_checks: list[bool] = []
        if self.energy_tolerance_eV_per_atom is not None:
            tolerance_checks.append(energy_range <= self.energy_tolerance_eV_per_atom)
        if self.force_tolerance_eV_per_A is not None:
            tolerance_checks.append(
                float(force_disagreement.max(initial=0.0))
                <= self.force_tolerance_eV_per_A
            )

        return ModelConsensusResult(
            atoms=detached_copy(result_atoms),
            predictions=tuple(predictions),
            successful_predictions=successful,
            mean_energy_eV=float(np.mean(energies)),
            mean_energy_eV_per_atom=float(np.mean(energies) / atom_count),
            energy_standard_deviation_eV_per_atom=energy_std,
            energy_range_eV_per_atom=energy_range,
            mean_forces_eV_per_A=mean_forces,
            force_disagreement_eV_per_A=force_disagreement,
            pairwise_force_rmse_eV_per_A=pairwise,
            pairwise_labels=tuple(prediction.label for prediction in successful),
            mean_stress_eV_per_A3=(
                None if mean_stress is None else np.asarray(mean_stress).copy()
            ),
            within_tolerance=(all(tolerance_checks) if tolerance_checks else None),
            runtime_seconds=time.perf_counter() - total_started,
        )
