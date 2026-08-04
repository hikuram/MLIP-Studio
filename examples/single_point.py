"""Minimal single-point energy, force, and stress calculation."""

import torch
from ase.build import bulk

import mlipstudio


device = "cuda" if torch.cuda.is_available() else "cpu"
atoms = bulk("Cu", "fcc", a=3.62).repeat((2, 2, 2))
calculator = mlipstudio.create_calculator("MACE MPA Medium", device=device)
task = mlipstudio.SinglePointTask(properties=("energy", "forces", "stress"))
result = task.calculate(atoms, calculator)

print(f"Energy: {result.energy:.6f} eV")
print(f"Maximum force: {result.max_force:.6f} eV/Angstrom")
print("Stress (eV/Angstrom^3):", result.stress)
