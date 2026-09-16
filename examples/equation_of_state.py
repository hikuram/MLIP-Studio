"""Minimal equation-of-state fit."""

from ase.build import bulk

import mlipstudio


atoms = bulk("Cu", "fcc", a=3.62)
calculator = mlipstudio.create_calculator("MACE MPA Medium")
result = mlipstudio.EquationOfStateTask().calculate(atoms, calculator)

print("Equilibrium volume (A^3):", result.equilibrium_volume_A3)
print("Bulk modulus (GPa):", result.bulk_modulus_GPa)
