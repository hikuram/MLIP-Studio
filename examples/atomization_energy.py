"""Minimal molecular atomization-energy calculation."""

import torch
from ase.build import molecule

import mlipstudio


device = "cuda" if torch.cuda.is_available() else "cpu"
atoms = molecule("H2O")
calculator = mlipstudio.create_calculator("MACE POLAR 1 S", device=device)
task = mlipstudio.AtomizationCohesiveEnergyTask()
result = task.calculate(atoms, calculator)

print(f"Atomization energy: {result.atomization_energy_eV:.6f} eV/molecule")
print("Reference source:", result.reference_source)
