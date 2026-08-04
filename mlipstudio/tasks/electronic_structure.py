"""Band-gap, density-of-states, and molecular HOMO-LUMO tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms

from ..exceptions import TaskCalculationError, TaskValidationError
from .base import Task, calculator_name, detached_copy, validate_atoms


def _as_numpy(value: Any, label: str) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    try:
        array = np.asarray(converted, dtype=float).squeeze()
    except Exception as exc:
        raise TaskCalculationError(f"Could not convert {label} to a numerical array.") from exc
    return array


def _as_scalar(value: Any, label: str) -> float:
    array = _as_numpy(value, label).reshape(-1)
    if array.size != 1:
        raise TaskCalculationError(f"{label} must contain exactly one value.")
    result = float(array[0])
    if not np.isfinite(result):
        raise TaskCalculationError(f"{label} is not finite.")
    return result


@dataclass(frozen=True)
class BandGapDOSResult:
    """Electronic DOS result with energies in eV."""

    atoms: Atoms
    band_gap_eV: float
    fermi_level_eV: float
    energies_eV: np.ndarray
    energies_relative_to_fermi_eV: np.ndarray
    density_of_states: np.ndarray
    runtime_seconds: float
    calculator_name: str


@dataclass(frozen=True)
class BandGapDOSTask(Task):
    """Calculate DOS, Fermi level, and band gap using PET-MAD-DOS."""

    copy_atoms: bool = True

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> BandGapDOSResult:
        validate_atoms(atoms)
        if calculator is None:
            raise TaskValidationError("BandGapDOSTask requires a PET-MAD-DOS calculator.")
        required_methods = ("calculate_dos", "calculate_bandgap", "calculate_efermi")
        missing = [name for name in required_methods if not callable(getattr(calculator, name, None))]
        if missing:
            raise TaskValidationError(
                "The calculator does not provide the PET-MAD-DOS method(s): "
                + ", ".join(missing)
            )

        working_atoms = atoms.copy() if self.copy_atoms else atoms
        started = time.perf_counter()
        try:
            raw_energies, raw_dos = calculator.calculate_dos(working_atoms)
            raw_band_gap = calculator.calculate_bandgap(working_atoms, dos=raw_dos)
            raw_fermi = calculator.calculate_efermi(working_atoms, dos=raw_dos)
            energies = _as_numpy(raw_energies, "DOS energies").reshape(-1)
            dos = _as_numpy(raw_dos, "density of states").reshape(-1)
            band_gap = _as_scalar(raw_band_gap, "band gap")
            fermi = _as_scalar(raw_fermi, "Fermi level")
        except TaskCalculationError:
            raise
        except Exception as exc:
            raise TaskCalculationError(f"Band-gap/DOS calculation failed: {exc}") from exc

        if energies.shape != dos.shape:
            raise TaskCalculationError(
                "DOS energy grid and density values must have the same length."
            )
        if energies.size == 0 or not np.all(np.isfinite(energies)) or not np.all(np.isfinite(dos)):
            raise TaskCalculationError("DOS output must be non-empty and finite.")

        return BandGapDOSResult(
            atoms=detached_copy(working_atoms),
            band_gap_eV=band_gap,
            fermi_level_eV=fermi,
            energies_eV=energies.copy(),
            energies_relative_to_fermi_eV=(energies - fermi),
            density_of_states=dos.copy(),
            runtime_seconds=time.perf_counter() - started,
            calculator_name=calculator_name(calculator),
        )


@dataclass(frozen=True)
class HOMOLUMOGapResult:
    """Molecular HOMO-LUMO prediction in eV."""

    atoms: Atoms
    gap_eV: float
    runtime_seconds: float
    calculator_name: str


@dataclass(frozen=True)
class HOMOLUMOGapTask(Task):
    """Predict a QM9-like molecule's HOMO-LUMO gap."""

    allowed_elements: frozenset[str] = frozenset({"H", "C", "N", "O", "F"})
    copy_atoms: bool = True

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> HOMOLUMOGapResult:
        validate_atoms(atoms)
        if np.any(atoms.pbc):
            raise TaskValidationError("HOMO-LUMO prediction requires a non-periodic molecule.")
        invalid = set(atoms.get_chemical_symbols()).difference(self.allowed_elements)
        if invalid:
            raise TaskValidationError(
                "QM9-Gap does not support element(s): " + ", ".join(sorted(invalid))
            )
        if calculator is None or not callable(
            getattr(calculator, "calculate_homo_lumo_gap", None)
        ):
            raise TaskValidationError(
                "HOMOLUMOGapTask requires the MLIP Studio QM9-Gap calculator."
            )

        working_atoms = atoms.copy() if self.copy_atoms else atoms
        started = time.perf_counter()
        try:
            gap = float(calculator.calculate_homo_lumo_gap(working_atoms))
        except Exception as exc:
            raise TaskCalculationError(f"HOMO-LUMO prediction failed: {exc}") from exc
        if not np.isfinite(gap):
            raise TaskCalculationError("Predicted HOMO-LUMO gap is not finite.")
        return HOMOLUMOGapResult(
            atoms=detached_copy(working_atoms),
            gap_eV=gap,
            runtime_seconds=time.perf_counter() - started,
            calculator_name=calculator_name(calculator),
        )
