"""Minimal geometry optimization."""

import torch
from ase.build import bulk

import mlipstudio


device = "cuda" if torch.cuda.is_available() else "cpu"
atoms = bulk("Cu", "fcc", a=3.62).repeat((2, 2, 2))
atoms.positions[0] += [0.10, -0.06, 0.04]

calculator = mlipstudio.create_calculator("MACE MPA Medium", device=device)
task = mlipstudio.OptimizationTask(optimizer="LBFGS", fmax=0.05, steps=30)
result = task.calculate(atoms, calculator)

print("Converged:", result.converged)
print("Steps:", result.steps)
print(f"Final energy: {result.energy:.6f} eV")
print(f"Final maximum force: {result.atomic_max_force:.6f} eV/Angstrom")
