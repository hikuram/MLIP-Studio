"""Minimal cohesive-energy calculation."""

import torch
from ase.build import bulk

import mlipstudio


device = "cuda" if torch.cuda.is_available() else "cpu"
atoms = bulk("Cu", "fcc", a=3.62).repeat((2, 2, 2))
calculator = mlipstudio.create_calculator("MACE MPA Medium", device=device)
task = mlipstudio.AtomizationCohesiveEnergyTask()
result = task.calculate(atoms, calculator)

print(f"Cohesive energy: {result.cohesive_energy_eV_per_atom:.6f} eV/atom")
print("Reference source:", result.reference_source)
