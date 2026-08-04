# Simple API examples

Each file is a small, linear example of one MLIP Studio task. There is no
command-line interface or configuration framework to learn: open a script,
change the atoms or model name, and run it.

Install MLIP Studio from the repository root:

```bash
python -m pip install . --no-deps
```

Then run an example:

```bash
python examples/single_point.py
```

Available examples:

- `single_point.py` — energy, forces, and stress
- `geometry_optimization.py` — atomic-position optimization
- `cohesive_energy.py` — periodic cohesive energy
- `atomization_energy.py` — molecular atomization energy
- `homo_lumo_gap.py` — molecular HOMO-LUMO gap
- `dipole_moment.py` — dipole moment and partial charges
- `spin_determination.py` — molecular spin-state scan
- `band_gap_dos.py` — material band gap and density of states

The examples use small ASE-built structures so the API call is easy to see.
Replace, for example,

```python
atoms = molecule("H2O")
```

with an ASE-readable input file when you are ready:

```python
from ase.io import read

atoms = read("my_structure.xyz")
```

The UMA spin example requires Hugging Face authentication and model access.
MACE and PET-MAD-DOS may download model files on their first run. See the main
README and Colab notebook for installation and access instructions.
