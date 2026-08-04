"""Minimal molecular spin-state scan."""

import torch
from ase.build import molecule

import mlipstudio


device = "cuda" if torch.cuda.is_available() else "cpu"
atoms = molecule("H2O")
calculator = mlipstudio.create_calculator(
    "UMA Small 1.2",
    device=device,
    task_name="omol",
)
task = mlipstudio.SpinDeterminationTask(charge=0, multiplicities=(1, 3, 5))
result = task.calculate(atoms, calculator)

for state in result.states:
    print(f"Multiplicity {state.multiplicity}: {state.energy_eV} eV")
print("Lowest-energy multiplicity:", result.optimal_state.multiplicity)
