"""Spin-multiplicity energy scans."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms

from ..exceptions import TaskCalculationError, TaskValidationError
from .base import Task, calculator_name, detached_copy, validate_atoms


@dataclass(frozen=True)
class SpinStateResult:
    """Energy and status for one tested spin multiplicity."""

    multiplicity: int
    total_spin: float
    unpaired_electrons: int
    energy_eV: float | None
    runtime_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class SpinDeterminationResult:
    """Complete spin scan and its lowest-energy successful state."""

    atoms: Atoms
    charge: int
    states: tuple[SpinStateResult, ...]
    optimal_state: SpinStateResult
    runtime_seconds: float
    calculator_name: str

    @property
    def successful_states(self) -> tuple[SpinStateResult, ...]:
        return tuple(state for state in self.states if state.error is None)


@dataclass(frozen=True)
class SpinDeterminationTask(Task):
    """Evaluate one geometry across candidate spin multiplicities.

    When multiplicities are omitted, physically parity-compatible values up to
    five are generated from the electron count. Explicit multiplicities can be
    used to reproduce a different scan.
    """

    charge: int = 0
    multiplicities: tuple[int, ...] | None = None
    maximum_multiplicity: int = 5
    enforce_electron_parity: bool = True
    continue_on_error: bool = True

    @staticmethod
    def _validate_known_calculator(calculator: Any) -> None:
        model_name = getattr(calculator, "_mlipstudio_model_name", None)
        if model_name is None:
            return
        parameters = getattr(calculator, "_mlipstudio_parameters", {})
        compatible = (
            (model_name.startswith("UMA ") and parameters.get("task_name") == "omol")
            or (model_name.startswith("MACE ") and ("OMOL" in model_name or "POLAR" in model_name))
            or (model_name.startswith("V3 OMOL "))
        )
        if not compatible:
            raise TaskValidationError(
                f"{model_name} is not registered as a charge/spin-compatible OMOL model."
            )

    def _multiplicities(self, atoms: Atoms) -> tuple[int, ...]:
        electron_count = int(np.sum(atoms.get_atomic_numbers())) - self.charge
        if electron_count <= 0:
            raise TaskValidationError("Charge leaves the system with no electrons.")
        if self.maximum_multiplicity < 1:
            raise TaskValidationError("maximum_multiplicity must be at least one.")

        if self.multiplicities is None:
            upper = min(electron_count + 1, self.maximum_multiplicity)
            parity = 1 if electron_count % 2 == 0 else 0
            candidates = tuple(
                value for value in range(1, upper + 1) if value % 2 == parity
            )
        else:
            candidates = tuple(dict.fromkeys(self.multiplicities))

        if not candidates or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in candidates
        ):
            raise TaskValidationError(
                "multiplicities must contain one or more positive integers."
            )
        if any(value > electron_count + 1 for value in candidates):
            raise TaskValidationError(
                "A requested multiplicity exceeds the electron-count limit."
            )
        if self.enforce_electron_parity:
            expected_parity = 1 if electron_count % 2 == 0 else 0
            invalid = [value for value in candidates if value % 2 != expected_parity]
            if invalid:
                raise TaskValidationError(
                    "Multiplicity parity is incompatible with the electron count: "
                    + ", ".join(map(str, invalid))
                )
        return candidates

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> SpinDeterminationResult:
        validate_atoms(atoms)
        if calculator is None:
            raise TaskValidationError("SpinDeterminationTask requires a calculator.")
        self._validate_known_calculator(calculator)
        multiplicities = self._multiplicities(atoms)
        base_atoms = atoms.copy()
        states: list[SpinStateResult] = []
        scan_started = time.perf_counter()

        for multiplicity in multiplicities:
            state_atoms = base_atoms.copy()
            state_atoms.info["charge"] = self.charge
            state_atoms.info["total_charge"] = self.charge
            state_atoms.info["spin"] = multiplicity
            state_atoms.info["total_spin"] = multiplicity
            state_atoms.calc = calculator
            state_started = time.perf_counter()
            try:
                reset = getattr(calculator, "reset", None)
                if callable(reset):
                    reset()
                energy = float(state_atoms.get_potential_energy())
                if not np.isfinite(energy):
                    raise ValueError("calculated energy is not finite")
                state = SpinStateResult(
                    multiplicity=multiplicity,
                    total_spin=(multiplicity - 1) / 2,
                    unpaired_electrons=multiplicity - 1,
                    energy_eV=energy,
                    runtime_seconds=time.perf_counter() - state_started,
                )
            except Exception as exc:
                if not self.continue_on_error:
                    raise TaskCalculationError(
                        f"Spin multiplicity {multiplicity} failed: {exc}"
                    ) from exc
                state = SpinStateResult(
                    multiplicity=multiplicity,
                    total_spin=(multiplicity - 1) / 2,
                    unpaired_electrons=multiplicity - 1,
                    energy_eV=None,
                    runtime_seconds=time.perf_counter() - state_started,
                    error=str(exc),
                )
            states.append(state)

        successful = [state for state in states if state.energy_eV is not None]
        if not successful:
            errors = "; ".join(
                f"{state.multiplicity}: {state.error}" for state in states
            )
            raise TaskCalculationError(f"All spin-state calculations failed. {errors}")
        optimal = min(successful, key=lambda state: float(state.energy_eV))
        return SpinDeterminationResult(
            atoms=detached_copy(base_atoms),
            charge=self.charge,
            states=tuple(states),
            optimal_state=optimal,
            runtime_seconds=time.perf_counter() - scan_started,
            calculator_name=calculator_name(calculator),
        )
