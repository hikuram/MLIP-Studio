"""Equation-of-state sampling and fitting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from ase import Atoms

from ..exceptions import TaskCalculationError, TaskValidationError
from .base import Task, calculator_name, detached_copy, prepare_atoms


EV_PER_A3_TO_GPA = 160.21766208


def murnaghan(
    volume: np.ndarray,
    equilibrium_energy: float,
    equilibrium_volume: float,
    bulk_modulus: float,
    pressure_derivative: float,
) -> np.ndarray:
    ratio = volume / equilibrium_volume
    return equilibrium_energy + bulk_modulus * equilibrium_volume * (
        ratio ** (1.0 - pressure_derivative)
        / (pressure_derivative * (pressure_derivative - 1.0))
        + ratio / pressure_derivative
        - 1.0 / (pressure_derivative - 1.0)
    )


def birch_murnaghan(
    volume: np.ndarray,
    equilibrium_energy: float,
    equilibrium_volume: float,
    bulk_modulus: float,
    pressure_derivative: float,
) -> np.ndarray:
    eta = (equilibrium_volume / volume) ** (2.0 / 3.0)
    return equilibrium_energy + 9.0 / 16.0 * bulk_modulus * equilibrium_volume * (
        (eta - 1.0) ** 3 * pressure_derivative
        + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def vinet(
    volume: np.ndarray,
    equilibrium_energy: float,
    equilibrium_volume: float,
    bulk_modulus: float,
    pressure_derivative: float,
) -> np.ndarray:
    x = (volume / equilibrium_volume) ** (1.0 / 3.0)
    return (
        equilibrium_energy
        + 2.0
        * bulk_modulus
        * equilibrium_volume
        / (pressure_derivative - 1.0) ** 2
        * (
            2.0
            - (
                5.0
                + 3.0 * x * (pressure_derivative - 1.0)
                - 3.0 * pressure_derivative
            )
            * np.exp(
                -1.5 * (pressure_derivative - 1.0) * (x - 1.0)
            )
        )
    )


_EQUATIONS: dict[str, Callable[..., np.ndarray]] = {
    "Birch-Murnaghan": birch_murnaghan,
    "Murnaghan": murnaghan,
    "Vinet": vinet,
}


@dataclass(frozen=True)
class EquationOfStateResult:
    """Sampled E(V) data and fitted four-parameter EOS values."""

    atoms: Atoms
    equation: str
    volumes_A3: np.ndarray
    energies_eV: np.ndarray
    fitted_energies_eV: np.ndarray
    residuals_eV: np.ndarray
    equilibrium_energy_eV: float
    equilibrium_volume_A3: float
    bulk_modulus_eV_per_A3: float
    bulk_modulus_GPa: float
    pressure_derivative: float
    parameter_covariance: np.ndarray
    parameter_standard_errors: np.ndarray
    rmse_eV: float
    sampled_structures: tuple[Atoms, ...]
    equilibrium_structure: Atoms
    runtime_seconds: float
    calculator_name: str


@dataclass(frozen=True)
class EquationOfStateTask(Task):
    """Uniformly scale a 3D periodic cell and fit an EOS curve."""

    equation: str = "Birch-Murnaghan"
    num_points: int = 7
    volume_range_percent: float = 5.0
    max_fit_evaluations: int = 20_000
    copy_atoms: bool = True

    def calculate(
        self,
        atoms: Atoms,
        calculator: Any | None = None,
    ) -> EquationOfStateResult:
        if self.equation not in _EQUATIONS:
            raise TaskValidationError(
                f"Unknown equation '{self.equation}'. Available: "
                + ", ".join(_EQUATIONS)
                + "."
            )
        if not isinstance(self.num_points, int) or isinstance(self.num_points, bool):
            raise TaskValidationError("num_points must be an integer.")
        if self.num_points < 5:
            raise TaskValidationError("num_points must be at least 5 for a four-parameter fit.")
        if not 0 < self.volume_range_percent < 100:
            raise TaskValidationError("volume_range_percent must be between 0 and 100.")
        if self.max_fit_evaluations <= 0:
            raise TaskValidationError("max_fit_evaluations must be greater than zero.")

        working_atoms, resolved_calculator = prepare_atoms(
            atoms,
            calculator,
            copy_atoms=self.copy_atoms,
        )
        if not np.all(working_atoms.pbc):
            raise TaskValidationError(
                "Equation-of-state fitting requires periodic boundary conditions "
                "in all three directions."
            )
        original_volume = float(working_atoms.get_volume())
        if not np.isfinite(original_volume) or original_volume <= 0:
            raise TaskValidationError("Equation-of-state fitting requires a valid cell.")

        original_cell = working_atoms.cell.copy()
        scaled_positions = working_atoms.get_scaled_positions().copy()
        fraction = float(self.volume_range_percent) / 100.0
        volumes = np.linspace(
            original_volume * (1.0 - fraction),
            original_volume * (1.0 + fraction),
            int(self.num_points),
        )
        energies: list[float] = []
        sampled: list[Atoms] = []
        started = time.perf_counter()
        for volume in volumes:
            sample = working_atoms.copy()
            sample.set_cell(
                original_cell * (float(volume) / original_volume) ** (1.0 / 3.0),
                scale_atoms=False,
            )
            sample.set_scaled_positions(scaled_positions)
            sample.calc = resolved_calculator
            try:
                energy = float(sample.get_potential_energy())
            except Exception as exc:
                raise TaskCalculationError(
                    f"EOS energy calculation failed at {volume:.6g} Angstrom^3: {exc}"
                ) from exc
            if not np.isfinite(energy):
                raise TaskCalculationError(
                    f"EOS energy is not finite at {volume:.6g} Angstrom^3."
                )
            energies.append(energy)
            sampled.append(detached_copy(sample))

        energy_array = np.asarray(energies, dtype=float)
        minimum_index = int(np.argmin(energy_array))
        equilibrium_volume_guess = float(volumes[minimum_index])
        equilibrium_energy_guess = float(energy_array[minimum_index])
        try:
            quadratic = np.polyfit(volumes, energy_array, 2)
            bulk_modulus_guess = max(
                equilibrium_volume_guess * 2.0 * float(quadratic[0]),
                1.0e-3,
            )
        except Exception:
            bulk_modulus_guess = 1.0

        try:
            from scipy.optimize import curve_fit
        except (ImportError, ModuleNotFoundError) as exc:
            raise TaskCalculationError(
                "Equation-of-state fitting requires SciPy. Install the project "
                "dependencies using the MLIP Studio README instructions."
            ) from exc

        equation = _EQUATIONS[self.equation]
        lower_bounds = (-np.inf, volumes.min() * 0.8, 1.0e-12, 1.01)
        upper_bounds = (np.inf, volumes.max() * 1.2, np.inf, 20.0)
        try:
            fitted_parameters, covariance = curve_fit(
                equation,
                volumes,
                energy_array,
                p0=(
                    equilibrium_energy_guess,
                    equilibrium_volume_guess,
                    bulk_modulus_guess,
                    4.0,
                ),
                bounds=(lower_bounds, upper_bounds),
                maxfev=int(self.max_fit_evaluations),
            )
        except Exception as exc:
            raise TaskCalculationError(
                f"Failed to fit the {self.equation} equation of state: {exc}"
            ) from exc

        fitted_parameters = np.asarray(fitted_parameters, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if not np.all(np.isfinite(fitted_parameters)):
            raise TaskCalculationError("EOS fitting returned non-finite parameters.")
        fitted_energies = np.asarray(
            equation(volumes, *fitted_parameters), dtype=float
        )
        residuals = energy_array - fitted_energies
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        equilibrium_energy, equilibrium_volume, bulk_modulus, pressure_derivative = (
            float(value) for value in fitted_parameters
        )

        equilibrium_structure = working_atoms.copy()
        equilibrium_structure.set_cell(
            original_cell * (equilibrium_volume / original_volume) ** (1.0 / 3.0),
            scale_atoms=False,
        )
        equilibrium_structure.set_scaled_positions(scaled_positions)

        return EquationOfStateResult(
            atoms=detached_copy(working_atoms),
            equation=self.equation,
            volumes_A3=volumes.copy(),
            energies_eV=energy_array,
            fitted_energies_eV=fitted_energies,
            residuals_eV=residuals,
            equilibrium_energy_eV=equilibrium_energy,
            equilibrium_volume_A3=equilibrium_volume,
            bulk_modulus_eV_per_A3=bulk_modulus,
            bulk_modulus_GPa=bulk_modulus * EV_PER_A3_TO_GPA,
            pressure_derivative=pressure_derivative,
            parameter_covariance=covariance.copy(),
            parameter_standard_errors=standard_errors,
            rmse_eV=float(np.sqrt(np.mean(residuals**2))),
            sampled_structures=tuple(sampled),
            equilibrium_structure=detached_copy(equilibrium_structure),
            runtime_seconds=time.perf_counter() - started,
            calculator_name=calculator_name(resolved_calculator),
        )


EOSTask = EquationOfStateTask
EOSResult = EquationOfStateResult

