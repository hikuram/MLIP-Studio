"""Packaged providers for task-specific property models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from .exceptions import ModelConfigurationError


_QM9_NUM_ELEMENTS = 86


def _atoms_to_qm9_graph(atoms: Atoms, cutoff: float):
    """Convert an ASE molecule to the graph representation used for training."""

    import torch
    from torch_geometric.data import Data

    atomic_numbers = atoms.get_atomic_numbers()
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > _QM9_NUM_ELEMENTS):
        raise ModelConfigurationError("QM9-Gap encountered an unsupported element.")

    node_features = np.zeros((len(atoms), _QM9_NUM_ELEMENTS), dtype=np.float32)
    node_features[np.arange(len(atoms)), atomic_numbers - 1] = 1.0

    positions = atoms.get_positions()
    distances = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
        axis=-1,
    )
    source, destination = np.where(
        (distances <= cutoff) & ~np.eye(len(atoms), dtype=bool)
    )
    if source.size == 0:
        raise ModelConfigurationError(
            "QM9-Gap could not create any molecular graph edges; increase cutoff."
        )

    edge_distances = torch.tensor(
        distances[source, destination],
        dtype=torch.float,
    )
    centers = torch.linspace(0.0, 6.0, 64)
    gamma = 1.0 / ((6.0 - 0.0) / 64) ** 2
    edge_attributes = torch.exp(
        -gamma * (edge_distances.unsqueeze(-1) - centers.unsqueeze(0)) ** 2
    )

    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(
            np.vstack((source, destination)),
            dtype=torch.long,
        ),
        edge_attr=edge_attributes,
        edge_dist=edge_distances.unsqueeze(-1),
        y=None,
        n_atoms=torch.tensor([len(atoms)], dtype=torch.long),
    )


def _load_qm9_model(model_path: Path, device):
    """Load the bundled MPNN without relying on repository-level modules."""

    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing, global_mean_pool

    class MPNNLayer(MessagePassing):
        def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
            super().__init__(aggr="add")
            self.msg_mlp = nn.Sequential(
                nn.Linear(node_dim + edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.upd_mlp = nn.Sequential(
                nn.Linear(node_dim + hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, node_dim),
            )
            self.layer_norm = nn.LayerNorm(node_dim)

        def forward(self, x, edge_index, edge_attr):
            aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            updated = self.upd_mlp(torch.cat([x, aggregated], dim=-1))
            return self.layer_norm(x + updated)

        def message(self, x_j, edge_attr):
            return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))

    class MPNN(nn.Module):
        def __init__(
            self,
            n_atom_features: int,
            n_edge_features: int,
            node_dim: int,
            hidden_dim: int,
            n_mp_layers: int,
            readout_hidden: int,
            target_mean: float,
            target_std: float,
        ):
            super().__init__()
            self.target_mean = target_mean
            self.target_std = target_std
            self.node_embed = nn.Sequential(
                nn.Linear(n_atom_features, node_dim),
                nn.SiLU(),
            )
            self.edge_embed = nn.Sequential(
                nn.Linear(n_edge_features, hidden_dim),
                nn.SiLU(),
            )
            self.mp_layers = nn.ModuleList(
                MPNNLayer(node_dim, hidden_dim, hidden_dim)
                for _ in range(n_mp_layers)
            )
            self.readout = nn.Sequential(
                nn.Linear(node_dim, readout_hidden),
                nn.SiLU(),
                nn.Linear(readout_hidden, readout_hidden),
                nn.SiLU(),
                nn.Linear(readout_hidden, 1),
            )

        def forward(self, data: Data):
            node_features = self.node_embed(data.x)
            edge_features = self.edge_embed(data.edge_attr)
            for message_passing in self.mp_layers:
                node_features = message_passing(
                    node_features,
                    data.edge_index,
                    edge_features,
                )
            graph_features = global_mean_pool(node_features, data.batch)
            normalized = self.readout(graph_features).squeeze(-1)
            return normalized * self.target_std + self.target_mean

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    parameters = checkpoint["args"]
    model = MPNN(
        n_atom_features=_QM9_NUM_ELEMENTS,
        n_edge_features=parameters.get("n_gaussians", 64),
        node_dim=parameters.get("node_dim", 128),
        hidden_dim=parameters.get("hidden_dim", 128),
        n_mp_layers=parameters.get("n_mp_layers", 2),
        readout_hidden=parameters.get("readout_hidden", 64),
        target_mean=checkpoint["target_mean"],
        target_std=checkpoint["target_std"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, parameters


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

        self.device = torch.device(device)
        self.cutoff = float(cutoff)
        self.model_path = path.resolve()
        self.model, self.training_parameters = _load_qm9_model(
            self.model_path,
            self.device,
        )

    def calculate_homo_lumo_gap(self, atoms: Atoms) -> float:
        """Predict the molecular HOMO-LUMO gap in eV."""

        import torch
        from torch_geometric.loader import DataLoader

        graph = _atoms_to_qm9_graph(atoms, cutoff=self.cutoff)
        batch = next(iter(DataLoader([graph], batch_size=1))).to(self.device)
        with torch.no_grad():
            prediction = self.model(batch)
        values = prediction.detach().cpu().reshape(-1)
        if values.numel() != 1:
            raise RuntimeError(
                f"QM9 gap model returned {values.numel()} values for one molecule."
            )
        return float(values.item())
