"""Minimal batch atomization-energy calculation."""

from ase.build import molecule

import mlipstudio


molecules = [molecule("H2O"), molecule("CH4")]
calculator = mlipstudio.create_calculator("MACE MPA Medium")
batch = mlipstudio.BatchAtomizationEnergyTask().calculate(molecules, calculator)

for item in batch.successful_items:
    print(item.label, item.result.atomization_energy_eV)
