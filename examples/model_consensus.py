"""Minimal two-model consensus calculation."""

from ase.build import bulk

import mlipstudio


atoms = bulk("Cu", "fcc", a=3.62)
models = {
    "MPA": mlipstudio.CalculatorFactory("MACE MPA Medium"),
    "MP": mlipstudio.CalculatorFactory("MACE MP 0b2 Small"),
}
result = mlipstudio.ModelConsensusTask().calculate(atoms, models)

print("Energy range (eV/atom):", result.energy_range_eV_per_atom)
print("Maximum force disagreement (eV/A):", result.max_force_disagreement_eV_per_A)
