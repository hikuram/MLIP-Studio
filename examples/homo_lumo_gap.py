"""Minimal HOMO-LUMO gap prediction."""

import torch
from ase.build import molecule

import mlipstudio


device = "cuda" if torch.cuda.is_available() else "cpu"
atoms = molecule("CH4")
calculator = mlipstudio.create_calculator("QM9-Gap", device=device)
task = mlipstudio.HOMOLUMOGapTask()
result = task.calculate(atoms, calculator)

print(f"HOMO-LUMO gap: {result.gap_eV:.6f} eV")
