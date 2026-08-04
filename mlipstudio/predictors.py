"""Task-specific model providers that are not general ASE calculators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms

from .exceptions import ModelConfigurationError


class QM9GapCalculator:
    """Loaded in-house MPNN for molecular HOMO-LUMO gap prediction."""

    allowed_elements = frozenset({"H", "C", "N", "O", "F"})

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        cutoff: float = 10.0,
    ) -> None:
        if cutoff <= 0:
            raise ModelConfigurationError("cutoff must be greater than zero.")
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise ModelConfigurationError(f"QM9 gap checkpoint not found: {path}")

        import torch

        from predict import load_model

        self.device = torch.device(device)
        self.cutoff = float(cutoff)
        self.model_path = path.resolve()
        self.model, self.training_parameters = load_model(
            str(self.model_path),
            self.device,
            verbose=False,
        )

    def calculate_homo_lumo_gap(self, atoms: Atoms) -> float:
        """Predict the molecular HOMO-LUMO gap in eV."""

        import torch
        from torch_geometric.loader import DataLoader

        from data import atoms_to_graph

        graph = atoms_to_graph(atoms, cutoff=self.cutoff, target_key=None)
        loader = DataLoader([graph], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(self.device)
        with torch.no_grad():
            prediction = self.model(batch)
        values = prediction.detach().cpu().reshape(-1)
        if values.numel() != 1:
            raise RuntimeError(
                f"QM9 gap model returned {values.numel()} values for one molecule."
            )
        return float(values.item())
