"""Minimal analytical MACE Hessian calculation."""

from ase.build import molecule

import mlipstudio


atoms = molecule("H2O")
calculator = mlipstudio.create_calculator("MACE MPA Medium")
result = mlipstudio.MACEHessianTask().calculate(atoms, calculator)

print("Hessian shape:", result.hessian_eV_per_A2.shape)
print("Symmetry error:", result.symmetry_error_eV_per_A2)
