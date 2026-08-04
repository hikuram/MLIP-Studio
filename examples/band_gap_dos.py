"""Minimal band-gap and density-of-states calculation."""

import matplotlib.pyplot as plt
from ase.build import bulk

import mlipstudio


atoms = bulk("Si", "diamond", a=5.43)
calculator = mlipstudio.create_calculator("PET-MAD-DOS", device="cpu")
task = mlipstudio.BandGapDOSTask()
result = task.calculate(atoms, calculator)

print(f"Band gap: {result.band_gap_eV:.6f} eV")
print(f"Fermi level: {result.fermi_level_eV:.6f} eV")

plt.plot(result.energies_relative_to_fermi_eV, result.density_of_states)
plt.axvline(0.0, color="black", linestyle="--")
plt.xlabel("Energy - Fermi level (eV)")
plt.ylabel("Density of states")
plt.show()
