"""Minimal batch energy, force, and stress calculation."""

from ase.build import bulk

import mlipstudio


structures = [bulk("Cu", "fcc", a=3.60), bulk("Cu", "fcc", a=3.65)]
calculator = mlipstudio.create_calculator("MACE MPA Medium")
batch = mlipstudio.BatchEnergyForceStressTask().calculate(structures, calculator)

for item in batch.successful_items:
    print(item.label, item.result.energy, item.result.max_force)
