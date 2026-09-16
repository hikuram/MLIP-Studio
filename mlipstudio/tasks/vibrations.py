"""Finite-difference vibrational mode analysis."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.units import kB

from ..exceptions import TaskCalculationError, TaskValidationError
from .base import Task, calculator_name, detached_copy, prepare_atoms


@dataclass(frozen=True)
class VibrationalModeResult:
    """Normal-mode frequencies, energies, displacements, and harmonic thermodynamics."""

    atoms: Atoms
    frequencies_cm_minus1: np.ndarray
    energies_eV: np.ndarray
    modes: np.ndarray
    imaginary_mode_indices: tuple[int, ...]
    zero_point_energy_eV: float
    vibrational_entropy_eV_per_K: float
    temperature_K: float
    runtime_seconds: float
    calculator_name: str

    @property
    def has_imaginary_modes(self) -> bool:
        return bool(self.imaginary_mode_indices)


@dataclass(frozen=True)
class VibrationalModeTask(Task):
    """Calculate harmonic vibrational modes with ASE finite differences.

    Temporary displacement files are always removed.  Complex frequencies
    are retained so imaginary modes are not silently discarded.
    """

    delta_A: float = 0.01
    nfree: int = 2
    indices: tuple[int, ...] | None = None
    temperature_K: float = 298.15
    imaginary_tolerance: float = 1.0e-10
    copy_atoms: bool = True

    def _validated_indices(self, atom_count: int) -> tuple[int, ...] | None:
        if self.indices is None:
            return None
        indices = tuple(self.indices)
        if not indices:
            raise TaskValidationError("indices must contain at least one atom index.")
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= atom_count
            for index in indices
        ):
            raise TaskValidationError(
                f"Vibrational atom indices must be integers from 0 to {atom_count - 1}."
            )
        if len(set(indices)) != len(indices):
            raise TaskValidationError("Vibrational atom indices must be unique.")
        return indices

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> VibrationalModeResult:
        if self.delta_A <= 0:
            raise TaskValidationError("delta_A must be greater than zero.")
        if self.nfree not in {2, 4}:
            raise TaskValidationError("nfree must be 2 or 4.")
        if self.temperature_K <= 0:
            raise TaskValidationError("temperature_K must be greater than zero.")
        if self.imaginary_tolerance < 0:
            raise TaskValidationError("imaginary_tolerance must be non-negative.")

        working_atoms, resolved_calculator = prepare_atoms(
            atoms,
            calculator,
            copy_atoms=self.copy_atoms,
        )
        indices = self._validated_indices(len(working_atoms))
        started = time.perf_counter()

        try:
            from ase.vibrations import Vibrations

            with tempfile.TemporaryDirectory(prefix="mlipstudio-vibrations-") as temp_dir:
                vibrations = Vibrations(
                    working_atoms,
                    indices=indices,
                    name=str(Path(temp_dir) / "vib"),
                    delta=float(self.delta_A),
                    nfree=int(self.nfree),
                )
                try:
                    vibrations.run()
                    frequencies = np.asarray(
                        vibrations.get_frequencies(), dtype=complex
                    ).reshape(-1)
                    energies = np.asarray(
                        vibrations.get_energies(), dtype=complex
                    ).reshape(-1)
                    modes = np.asarray(
                        [vibrations.get_mode(index) for index in range(len(frequencies))]
                    )
                finally:
                    vibrations.clean()
        except Exception as exc:
            raise TaskCalculationError(
                f"Vibrational mode calculation failed: {exc}"
            ) from exc

        if frequencies.size == 0 or energies.shape != frequencies.shape:
            raise TaskCalculationError(
                "Vibrational analysis returned empty or inconsistent mode arrays."
            )
        if not (
            np.all(np.isfinite(frequencies.real))
            and np.all(np.isfinite(frequencies.imag))
            and np.all(np.isfinite(energies.real))
            and np.all(np.isfinite(energies.imag))
        ):
            raise TaskCalculationError("Vibrational analysis returned non-finite values.")

        imaginary = tuple(
            int(index)
            for index in np.flatnonzero(
                np.abs(frequencies.imag) > float(self.imaginary_tolerance)
            )
        )
        physical_energies = energies.real[
            (np.abs(energies.imag) <= float(self.imaginary_tolerance))
            & (energies.real > float(self.imaginary_tolerance))
        ]
        zero_point_energy = 0.5 * float(np.sum(physical_energies))

        if physical_energies.size:
            x = physical_energies / (kB * float(self.temperature_K))
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                entropy_terms = x / np.expm1(x) - np.log1p(-np.exp(-x))
            entropy = float(kB * np.sum(entropy_terms))
        else:
            entropy = 0.0

        return VibrationalModeResult(
            atoms=detached_copy(working_atoms),
            frequencies_cm_minus1=frequencies.copy(),
            energies_eV=energies.copy(),
            modes=modes.copy(),
            imaginary_mode_indices=imaginary,
            zero_point_energy_eV=zero_point_energy,
            vibrational_entropy_eV_per_K=entropy,
            temperature_K=float(self.temperature_K),
            runtime_seconds=time.perf_counter() - started,
            calculator_name=calculator_name(resolved_calculator),
        )
