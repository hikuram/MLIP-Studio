"""Minimal vibrational mode analysis."""

from ase.build import molecule

import mlipstudio


atoms = molecule("H2O")
calculator = mlipstudio.create_calculator("MACE MPA Medium")
result = mlipstudio.VibrationalModeTask().calculate(atoms, calculator)

print("Frequencies (cm^-1):", result.frequencies_cm_minus1)
print("Zero-point energy (eV):", result.zero_point_energy_eV)
