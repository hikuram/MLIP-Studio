"""Minimal batch HOMO-LUMO gap prediction."""

from ase.build import molecule

import mlipstudio


molecules = [molecule("CH4"), molecule("H2O")]
calculator = mlipstudio.create_calculator("QM9-Gap")
batch = mlipstudio.BatchHOMOLUMOGapTask().calculate(molecules, calculator)

for item in batch.successful_items:
    print(item.label, item.result.gap_eV)
