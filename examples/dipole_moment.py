"""Minimal dipole-moment and partial-charge calculation."""

import torch
from ase.build import molecule

import mlipstudio


device = "cuda" if torch.cuda.is_available() else "cpu"
atoms = molecule("H2O")
calculator = mlipstudio.create_calculator("MACE POLAR 1 S", device=device)
task = mlipstudio.DipoleMomentTask(charge=0, spin_multiplicity=1)
result = task.calculate(atoms, calculator)

print("Dipole vector (e*Angstrom):", result.dipole_eA)
print(f"Dipole magnitude: {result.dipole_magnitude_eA:.6f} e*Angstrom")
print("Partial charges:", result.partial_charges_e)
