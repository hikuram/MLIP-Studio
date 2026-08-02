"""Model-consensus calculations and Streamlit rendering for MLIP Studio.

The dataset/model registry in this module is deliberately curated rather than
inferred from display-name substrings.  Calculator libraries are imported only
inside their family-specific factories so importing this module does not load
optional MLIP packages or model checkpoints.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write


MODEL_CONSENSUS_TASK = "Model Consensus"
MIN_CONSENSUS_MODELS = 2
MAX_CONSENSUS_MODELS = 6
SOURCE_FRAME_COUNT_KEY = "_mlip_studio_source_frame_count"

CONSENSUS_WARNING = (
    "Model agreement measures consensus among the selected models and can "
    "indicate extrapolation risk. Agreement does not guarantee agreement with "
    "DFT, because models may share correlated errors or training-domain gaps."
)


@dataclass(frozen=True)
class ConsensusModelSpec:
    """All static information needed to construct one curated calculator."""

    id: str
    label: str
    family: str
    dataset: str
    model_name: str
    force_mode: str
    task_name: str | None = None
    modal: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConsensusSelection:
    """Serializable values selected in the Model Consensus controls."""

    dataset: str
    model_specs: tuple[ConsensusModelSpec, ...]
    device: str
    charge: int | float | None
    spin: int | None
    use_tolerances: bool = False
    force_tolerance: float | None = None
    energy_tolerance: float | None = None
    run_perturbation_scan: bool = False
    perturbation_count: int = 7
    perturbation_amplitude: float = 0.05
    perturbation_seed: int = 42
    fingerprint: str | None = None

    @property
    def is_valid(self) -> bool:
        return (
            MIN_CONSENSUS_MODELS
            <= len(self.model_specs)
            <= MAX_CONSENSUS_MODELS
            and self.fingerprint is not None
        )


@dataclass
class ModelPrediction:
    """Numerical prediction and status for one selected model."""

    spec: ConsensusModelSpec
    energy: float | None
    forces: np.ndarray | None
    stress: np.ndarray | None
    runtime_seconds: float
    error: str | None
    stress_error: str | None = None


@dataclass
class PerturbationPrediction:
    """One model prediction at one point along the perturbation path."""

    spec: ConsensusModelSpec
    configuration_index: int
    path_coordinate: float
    energy: float | None
    forces: np.ndarray | None
    runtime_seconds: float
    error: str | None


@dataclass
class PerturbationResult:
    """Prepared energy and force comparisons along a 1D PES slice."""

    configuration_count: int
    max_rms_displacement: float
    random_seed: int
    path_coordinates: np.ndarray
    predictions: list[PerturbationPrediction]
    model_configuration_table: pd.DataFrame
    configuration_summary_table: pd.DataFrame


@dataclass
class ConsensusResult:
    """Prepared numerical consensus results with no live calculator objects."""

    dataset: str
    predictions: list[ModelPrediction]
    successful_predictions: list[ModelPrediction]
    model_table: pd.DataFrame
    atom_table: pd.DataFrame
    pairwise_force_rmse: pd.DataFrame
    mean_forces: np.ndarray
    force_disagreement: np.ndarray
    metrics: dict[str, float]
    perturbation: PerturbationResult | None = None


class ConsensusError(RuntimeError):
    """Base class for readable Model Consensus failures."""


class ConsensusValidationError(ConsensusError):
    """Raised when an input, selection, or numerical prediction is invalid."""


class ConsensusModelError(ConsensusError):
    """Raised when an optional model package or checkpoint cannot be loaded."""


class ConsensusCalculationError(ConsensusValidationError):
    """Raised when fewer than two models produce valid energy/force results."""

    def __init__(
        self,
        message: str,
        predictions: Sequence[ModelPrediction] = (),
        model_table: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(message)
        self.predictions = list(predictions)
        self.model_table = model_table


DATASET_DESCRIPTIONS: dict[str, str] = {
    "OMat24": (
        "**OMat24:** Meta FAIR Open Materials 2024 dataset "
        "(*arXiv:2410.12771*). This committee contains only repository models "
        "explicitly configured for OMat/`omat24` inference."
    ),
    "OMol25": (
        "**OMol25:** Meta FAIR Open Molecules dataset family. This committee "
        "contains only repository models explicitly identified or configured "
        "for OMOL inference; total charge and spin multiplicity are shared by "
        "every selected model."
    ),
}


# This is an explicit compatibility registry.  Labels, checkpoint identifiers,
# task names, modals, versions, and constructor settings mirror model_config.py
# and the calculator branches in Home.py.
CONSENSUS_MODEL_REGISTRY: tuple[ConsensusModelSpec, ...] = (
    # OMat24 — MACE
    ConsensusModelSpec(
        id="omat24:mace:medium",
        label="MACE OMAT Medium",
        family="MACE",
        dataset="OMat24",
        model_name=(
            "https://github.com/ACEsuit/mace-mp/releases/download/"
            "mace_omat_0/mace-omat-0-medium.model"
        ),
        force_mode="Conservative",
        metadata={"dispersion": False, "default_dtype": "float32"},
    ),
    ConsensusModelSpec(
        id="omat24:mace:small",
        label="MACE OMAT Small",
        family="MACE",
        dataset="OMat24",
        model_name=(
            "https://github.com/ACEsuit/mace-mp/releases/download/"
            "mace_omat_0/mace-omat-0-small.model"
        ),
        force_mode="Conservative",
        metadata={"dispersion": False, "default_dtype": "float32"},
    ),
    # OMat24 — FairChem UMA
    ConsensusModelSpec(
        id="omat24:fairchem:uma-s-1p2",
        label="UMA Small 1.2",
        family="FairChem",
        dataset="OMat24",
        model_name="uma-s-1p2",
        force_mode="Direct",
        task_name="omat",
    ),
    ConsensusModelSpec(
        id="omat24:fairchem:uma-s-1p1",
        label="UMA Small 1.1",
        family="FairChem",
        dataset="OMat24",
        model_name="uma-s-1p1",
        force_mode="Direct",
        task_name="omat",
    ),
    # OMat24 — ORB
    ConsensusModelSpec(
        id="omat24:orb:v3-conservative-inf",
        label="V3 OMAT Conservative (inf)",
        family="ORB",
        dataset="OMat24",
        model_name="orb_v3_conservative_inf_omat",
        force_mode="Conservative",
        metadata={"precision": "float32-high"},
    ),
    ConsensusModelSpec(
        id="omat24:orb:v3-conservative-20",
        label="V3 OMAT Conservative (20)",
        family="ORB",
        dataset="OMat24",
        model_name="orb_v3_conservative_20_omat",
        force_mode="Conservative",
        metadata={"precision": "float32-high"},
    ),
    ConsensusModelSpec(
        id="omat24:orb:v3-direct-inf",
        label="V3 OMAT Direct (inf)",
        family="ORB",
        dataset="OMat24",
        model_name="orb_v3_direct_inf_omat",
        force_mode="Direct",
        metadata={"precision": "float32-high"},
    ),
    ConsensusModelSpec(
        id="omat24:orb:v3-direct-20",
        label="V3 OMAT Direct (20)",
        family="ORB",
        dataset="OMat24",
        model_name="orb_v3_direct_20_omat",
        force_mode="Direct",
        metadata={"precision": "float32-high"},
    ),
    # OMat24 — PET/UPET.  Direct forces match the existing UI default.
    ConsensusModelSpec(
        id="omat24:pet:xs-v1.0.0",
        label="PET-OMAT-XS-V1.0.0",
        family="UPET/PET",
        dataset="OMat24",
        model_name="pet-omat-xs",
        force_mode="Direct",
        metadata={"version": "1.0.0", "non_conservative": True},
    ),
    ConsensusModelSpec(
        id="omat24:pet:s-v1.0.0",
        label="PET-OMAT-S-V1.0.0",
        family="UPET/PET",
        dataset="OMat24",
        model_name="pet-omat-s",
        force_mode="Direct",
        metadata={"version": "1.0.0", "non_conservative": True},
    ),
    ConsensusModelSpec(
        id="omat24:pet:m-v1.0.0",
        label="PET-OMAT-M-V1.0.0",
        family="UPET/PET",
        dataset="OMat24",
        model_name="pet-omat-m",
        force_mode="Direct",
        metadata={"version": "1.0.0", "non_conservative": True},
    ),
    ConsensusModelSpec(
        id="omat24:pet:l-v1.0.0",
        label="PET-OMAT-L-V1.0.0",
        family="UPET/PET",
        dataset="OMat24",
        model_name="pet-omat-l",
        force_mode="Direct",
        metadata={"version": "1.0.0", "non_conservative": True},
    ),
    # OMat24 — SevenNet
    ConsensusModelSpec(
        id="omat24:sevennet:7net-omat",
        label="7net-omat",
        family="SevenNet",
        dataset="OMat24",
        model_name="7net-omat",
        force_mode="Conservative",
    ),
    ConsensusModelSpec(
        id="omat24:sevennet:7net-omni",
        label="7net-omni",
        family="SevenNet",
        dataset="OMat24",
        model_name="7net-omni",
        force_mode="Conservative",
        modal="omat24",
    ),
    # OMol25 — MACE
    ConsensusModelSpec(
        id="omol25:mace:xl-4m",
        label="MACE OMOL-0 XL 4M",
        family="MACE",
        dataset="OMol25",
        model_name=(
            "https://github.com/ACEsuit/mace-foundations/releases/download/"
            "mace_omol_0/mace-omol-0-extra-large-4M.model"
        ),
        force_mode="Conservative",
        metadata={"dispersion": False, "default_dtype": "float32"},
    ),
    ConsensusModelSpec(
        id="omol25:mace:xl-1024",
        label="MACE OMOL-0 XL 1024",
        family="MACE",
        dataset="OMol25",
        model_name=(
            "https://github.com/ACEsuit/mace-foundations/releases/download/"
            "mace_omol_0/MACE-omol-0-extra-large-1024.model"
        ),
        force_mode="Conservative",
        metadata={"dispersion": False, "default_dtype": "float32"},
    ),
    # OMol25 — FairChem UMA/eSEN
    ConsensusModelSpec(
        id="omol25:fairchem:uma-s-1p2",
        label="UMA Small 1.2",
        family="FairChem",
        dataset="OMol25",
        model_name="uma-s-1p2",
        force_mode="Direct",
        task_name="omol",
    ),
    ConsensusModelSpec(
        id="omol25:fairchem:uma-s-1p1",
        label="UMA Small 1.1",
        family="FairChem",
        dataset="OMol25",
        model_name="uma-s-1p1",
        force_mode="Direct",
        task_name="omol",
    ),
    ConsensusModelSpec(
        id="omol25:fairchem:esen-md-direct",
        label="ESEN MD Direct All OMOL",
        family="FairChem",
        dataset="OMol25",
        model_name="esen-md-direct-all-omol",
        force_mode="Direct",
        task_name="omol",
    ),
    ConsensusModelSpec(
        id="omol25:fairchem:esen-sm-conserving",
        label="ESEN SM Conserving All OMOL",
        family="FairChem",
        dataset="OMol25",
        model_name="esen-sm-conserving-all-omol",
        force_mode="Conservative",
        task_name="omol",
    ),
    ConsensusModelSpec(
        id="omol25:fairchem:esen-sm-direct",
        label="ESEN SM Direct All OMOL",
        family="FairChem",
        dataset="OMol25",
        model_name="esen-sm-direct-all-omol",
        force_mode="Direct",
        task_name="omol",
    ),
    # OMol25 — ORB
    ConsensusModelSpec(
        id="omol25:orb:v3-conservative",
        label="V3 OMOL Conservative",
        family="ORB",
        dataset="OMol25",
        model_name="orb_v3_conservative_omol",
        force_mode="Conservative",
        metadata={"precision": "float32-high"},
    ),
    ConsensusModelSpec(
        id="omol25:orb:v3-direct",
        label="V3 OMOL Direct",
        family="ORB",
        dataset="OMol25",
        model_name="orb_v3_direct_omol",
        force_mode="Direct",
        metadata={"precision": "float32-high"},
    ),
)


def get_consensus_datasets() -> list[str]:
    """Return curated dataset groups in their stable display order."""

    return list(DATASET_DESCRIPTIONS)


def get_consensus_models(dataset: str) -> list[ConsensusModelSpec]:
    """Return only explicitly registered models for *dataset*."""

    return [spec for spec in CONSENSUS_MODEL_REGISTRY if spec.dataset == dataset]


def validate_single_structure(atoms: Any) -> Atoms:
    """Require exactly one non-empty ASE ``Atoms`` structure."""

    if isinstance(atoms, (list, tuple)):
        raise ConsensusValidationError(
            "Model Consensus supports exactly one structure; a batch or "
            f"trajectory containing {len(atoms)} frame(s) was provided."
        )
    if not isinstance(atoms, Atoms):
        raise ConsensusValidationError(
            "Model Consensus supports exactly one ASE Atoms object, not a "
            "batch or trajectory."
        )
    if len(atoms) == 0:
        raise ConsensusValidationError(
            "Model Consensus cannot run on a structure containing zero atoms."
        )
    source_frame_count = atoms.info.get(SOURCE_FRAME_COUNT_KEY, 1)
    try:
        frame_count = int(source_frame_count)
    except (TypeError, ValueError):
        frame_count = 1
    if frame_count != 1:
        raise ConsensusValidationError(
            "Model Consensus supports exactly one structure. The selected "
            f"source contains {frame_count} frames; please provide a "
            "single-frame structure file."
        )
    return atoms


def validate_consensus_selection(
    atoms: Any,
    model_specs: Sequence[ConsensusModelSpec],
    dataset: str | None = None,
) -> None:
    """Validate the single structure and curated committee selection."""

    validate_single_structure(atoms)
    if len(model_specs) < MIN_CONSENSUS_MODELS:
        raise ConsensusValidationError(
            "Select at least two compatible models for Model Consensus."
        )
    if len(model_specs) > MAX_CONSENSUS_MODELS:
        raise ConsensusValidationError(
            f"Select no more than {MAX_CONSENSUS_MODELS} models to limit "
            "accelerator memory use."
        )
    ids = [spec.id for spec in model_specs]
    if len(ids) != len(set(ids)):
        raise ConsensusValidationError(
            "The Model Consensus committee contains duplicate model IDs."
        )
    datasets = {spec.dataset for spec in model_specs}
    if len(datasets) != 1:
        raise ConsensusValidationError(
            "All Model Consensus models must belong to the same curated dataset."
        )
    if dataset is not None and datasets != {dataset}:
        raise ConsensusValidationError(
            f"Every selected model must belong to the {dataset} registry."
        )


def make_consensus_fingerprint(
    atoms: Atoms,
    dataset: str,
    model_ids: Sequence[str],
    charge: int | float | None,
    spin: int | None,
    device: str,
    *,
    run_perturbation_scan: bool = False,
    perturbation_count: int = 7,
    perturbation_amplitude: float = 0.05,
    perturbation_seed: int = 42,
) -> str:
    """Hash the structure and all calculator-defining consensus controls."""

    validate_single_structure(atoms)
    digest = hashlib.sha256()
    for array in (
        np.asarray(atoms.numbers, dtype=np.int64),
        np.asarray(atoms.positions, dtype=np.float64),
        np.asarray(atoms.cell.array, dtype=np.float64),
        np.asarray(atoms.pbc, dtype=np.bool_),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    config = {
        "dataset": dataset,
        "model_ids": list(model_ids),
        "charge": charge,
        "spin": spin,
        "device": device,
        "run_perturbation_scan": run_perturbation_scan,
        "perturbation_count": (
            perturbation_count if run_perturbation_scan else None
        ),
        "perturbation_amplitude": (
            perturbation_amplitude if run_perturbation_scan else None
        ),
        "perturbation_seed": (
            perturbation_seed if run_perturbation_scan else None
        ),
    }
    digest.update(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def render_consensus_sidebar(atoms: Atoms | None) -> ConsensusSelection:
    """Render ordered Model Consensus controls and return serializable values."""

    import streamlit as st

    st.sidebar.markdown("## Model Consensus")
    dataset = st.sidebar.selectbox(
        "Training Dataset",
        get_consensus_datasets(),
        key="consensus_dataset",
    )
    st.sidebar.info(DATASET_DESCRIPTIONS[dataset])

    specs = get_consensus_models(dataset)
    spec_by_id = {spec.id: spec for spec in specs}
    previous_dataset = st.session_state.get("_consensus_previous_dataset")
    if previous_dataset is not None and previous_dataset != dataset:
        st.session_state.pop("consensus_model_ids", None)
    st.session_state["_consensus_previous_dataset"] = dataset

    default_ids = [spec.id for spec in specs[:3]] if previous_dataset is None else []
    selected_ids = st.sidebar.multiselect(
        "Compatible Models",
        options=[spec.id for spec in specs],
        default=default_ids,
        format_func=lambda model_id: spec_by_id[model_id].label,
        max_selections=MAX_CONSENSUS_MODELS,
        help=(
            "Select at least two models. Three or more models are recommended "
            "for a more informative committee comparison."
        ),
        key="consensus_model_ids",
    )
    selected_specs = tuple(spec_by_id[model_id] for model_id in selected_ids)

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except (ImportError, AttributeError):
        cuda_available = False

    device_label = st.sidebar.radio(
        "Computation Device:",
        ["CPU", "CUDA (GPU)"],
        index=1 if cuda_available else 0,
        key="consensus_device",
    )
    device = "cuda" if device_label == "CUDA (GPU)" and cuda_available else "cpu"
    if device_label == "CUDA (GPU)" and not cuda_available:
        st.sidebar.warning("CUDA is not available; Model Consensus will use CPU.")
    elif device == "cpu" and not cuda_available:
        st.sidebar.info("No GPU detected. Using CPU.")

    charge: int | float | None = None
    spin: int | None = None
    if dataset == "OMol25":
        default_charge = int(atoms.info.get("charge", 0)) if atoms is not None else 0
        default_spin = int(atoms.info.get("spin", 1)) if atoms is not None else 1
        charge = st.sidebar.number_input(
            "Total Charge",
            min_value=-10,
            max_value=10,
            value=default_charge,
            step=1,
            key="consensus_charge",
        )
        spin = int(
            st.sidebar.number_input(
                "Spin Multiplicity (2S + 1)",
                min_value=1,
                max_value=20,
                value=max(1, default_spin),
                step=1,
                key="consensus_spin",
            )
        )

    with st.sidebar.expander("Advanced consensus settings"):
        run_perturbation_scan = st.checkbox(
            "Generate perturbed configurations and compare the PES",
            value=False,
            key="consensus_run_perturbation_scan",
            help=(
                "Evaluate every selected model along the same reproducible "
                "collective atomic-displacement path."
            ),
        )
        perturbation_count = 7
        perturbation_amplitude = 0.05
        perturbation_seed = 42
        if run_perturbation_scan:
            perturbation_count = int(
                st.number_input(
                    "Number of perturbed configurations",
                    min_value=5,
                    max_value=10,
                    value=7,
                    step=1,
                    key="consensus_perturbation_count",
                )
            )
            perturbation_amplitude = float(
                st.number_input(
                    "Maximum RMS displacement (Å)",
                    min_value=0.005,
                    max_value=0.25,
                    value=0.05,
                    step=0.005,
                    format="%.3f",
                    key="consensus_perturbation_amplitude",
                )
            )
            perturbation_seed = int(
                st.number_input(
                    "Perturbation random seed",
                    min_value=0,
                    max_value=1_000_000,
                    value=42,
                    step=1,
                    key="consensus_perturbation_seed",
                )
            )
            st.caption(
                "The original structure is added separately as the "
                "zero-displacement reference. Every generated configuration "
                "follows one fixed random collective direction up to the "
                "selected RMS displacement."
            )

        st.markdown("---")
        use_tolerances = st.checkbox(
            "Use my consensus tolerances",
            value=False,
            key="consensus_use_tolerances",
            help=(
                "These are user-defined screening thresholds, not calibrated "
                "error bounds."
            ),
        )
        force_tolerance = None
        energy_tolerance = None
        if use_tolerances:
            force_tolerance = float(
                st.number_input(
                    "Maximum atomic force disagreement (eV/Å)",
                    min_value=0.0,
                    value=0.10,
                    step=0.01,
                    format="%.4f",
                    key="consensus_force_tolerance",
                )
            )
            energy_tolerance = float(
                st.number_input(
                    "Energy range tolerance (eV/atom)",
                    min_value=0.0,
                    value=0.05,
                    step=0.01,
                    format="%.4f",
                    key="consensus_energy_tolerance",
                )
            )
            st.caption(
                "Applied to maximum atomic force disagreement and the "
                "model-wise energy-per-atom range."
            )

    modes = {spec.force_mode for spec in selected_specs}
    if {"Direct", "Conservative"}.issubset(modes):
        st.sidebar.warning(
            "The selected committee mixes direct-force and conservative-force "
            "models; interpret force disagreement with that distinction in mind."
        )
    if len(selected_specs) < MIN_CONSENSUS_MODELS:
        st.sidebar.warning("Select at least two compatible models to run.")

    fingerprint = None
    if atoms is not None:
        try:
            fingerprint = make_consensus_fingerprint(
                atoms,
                dataset,
                selected_ids,
                charge,
                spin,
                device,
                run_perturbation_scan=run_perturbation_scan,
                perturbation_count=perturbation_count,
                perturbation_amplitude=perturbation_amplitude,
                perturbation_seed=perturbation_seed,
            )
        except ConsensusValidationError as exc:
            st.sidebar.error(str(exc))
            fingerprint = None

    return ConsensusSelection(
        dataset=dataset,
        model_specs=selected_specs,
        device=device,
        charge=charge,
        spin=spin,
        use_tolerances=use_tolerances,
        force_tolerance=force_tolerance,
        energy_tolerance=energy_tolerance,
        run_perturbation_scan=run_perturbation_scan,
        perturbation_count=perturbation_count,
        perturbation_amplitude=perturbation_amplitude,
        perturbation_seed=perturbation_seed,
        fingerprint=fingerprint,
    )


def _build_mace_calculator(spec: ConsensusModelSpec, device: str) -> Any:
    from mace.calculators import mace_mp

    return mace_mp(
        model=spec.model_name,
        dispersion=bool(spec.metadata.get("dispersion", False)),
        device=device,
        default_dtype=str(spec.metadata.get("default_dtype", "float32")),
    )


def _build_fairchem_calculator(spec: ConsensusModelSpec, device: str) -> Any:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        spec.model_name,
        inference_settings="default",
        device=device,
    )
    return FAIRChemCalculator(predictor, task_name=spec.task_name or "omol")


def _build_orb_calculator(spec: ConsensusModelSpec, device: str) -> Any:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    try:
        factory = getattr(pretrained, spec.model_name)
    except AttributeError as exc:
        raise ConsensusModelError(
            f"ORB factory '{spec.model_name}' is unavailable."
        ) from exc
    orbff = factory(
        device=device,
        precision=str(spec.metadata.get("precision", "float32-high")),
    )
    return ORBCalculator(orbff, device=device)


def _build_upet_calculator(spec: ConsensusModelSpec, device: str) -> Any:
    from upet.calculator import UPETCalculator

    return UPETCalculator(
        model=spec.model_name,
        version=str(spec.metadata["version"]),
        non_conservative=bool(spec.metadata.get("non_conservative", True)),
        device=device,
    )


def _build_sevennet_calculator(spec: ConsensusModelSpec, device: str) -> Any:
    from sevenn.calculator import SevenNetCalculator

    kwargs: dict[str, Any] = {"model": spec.model_name, "device": device}
    if spec.modal is not None:
        kwargs["modal"] = spec.modal
    return SevenNetCalculator(**kwargs)


def build_consensus_calculator(
    spec: ConsensusModelSpec,
    device: str,
) -> Any:
    """Construct one calculator lazily using the existing MLIP Studio APIs."""

    builders = {
        "MACE": _build_mace_calculator,
        "FairChem": _build_fairchem_calculator,
        "ORB": _build_orb_calculator,
        "UPET/PET": _build_upet_calculator,
        "SevenNet": _build_sevennet_calculator,
    }
    try:
        builder = builders[spec.family]
    except KeyError as exc:
        raise ConsensusModelError(
            f"Unsupported Model Consensus family '{spec.family}' for "
            f"{spec.label}."
        ) from exc
    try:
        return builder(spec, device)
    except ConsensusModelError:
        raise
    except (ImportError, ModuleNotFoundError) as exc:
        package = getattr(exc, "name", None) or spec.family
        raise ConsensusModelError(
            f"{spec.label} requires the optional package '{package}', which "
            "could not be imported."
        ) from exc
    except Exception as exc:
        raise ConsensusModelError(
            f"Failed to load {spec.label}: {exc}"
        ) from exc


def _normalise_stress(stress: Any) -> np.ndarray:
    array = np.asarray(stress, dtype=float)
    if array.shape == (6,):
        normalised = array.copy()
    elif array.shape == (3, 3):
        normalised = np.array(
            [
                array[0, 0],
                array[1, 1],
                array[2, 2],
                array[1, 2],
                array[0, 2],
                array[0, 1],
            ],
            dtype=float,
        )
    else:
        raise ValueError(f"unsupported stress shape {array.shape}")
    if not np.all(np.isfinite(normalised)):
        raise ValueError("stress contains non-finite values")
    return normalised


def _validated_prediction(
    prediction: ModelPrediction,
    number_of_atoms: int,
) -> ModelPrediction:
    if prediction.error is not None:
        return prediction
    try:
        if prediction.energy is None:
            raise ValueError("energy was not returned")
        energy = float(prediction.energy)
        if not np.isfinite(energy):
            raise ValueError("energy is non-finite")
        if prediction.forces is None:
            raise ValueError("forces were not returned")
        forces = np.asarray(prediction.forces, dtype=float)
        expected_shape = (number_of_atoms, 3)
        if forces.shape != expected_shape:
            raise ValueError(
                f"force shape is {forces.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(forces)):
            raise ValueError("forces contain non-finite values")
        stress = prediction.stress
        if stress is not None:
            try:
                stress = _normalise_stress(stress)
            except ValueError:
                stress = None
        return replace(prediction, energy=energy, forces=forces.copy(), stress=stress)
    except (TypeError, ValueError) as exc:
        return replace(
            prediction,
            energy=None,
            forces=None,
            stress=None,
            error=f"Invalid prediction: {exc}",
        )


def build_model_status_table(
    predictions: Sequence[ModelPrediction],
    number_of_atoms: int,
) -> pd.DataFrame:
    """Create one status row per selected model, including failures."""

    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        success = (
            prediction.error is None
            and prediction.energy is not None
            and prediction.forces is not None
        )
        energy = float(prediction.energy) if success else np.nan
        force_max = (
            float(np.max(np.linalg.norm(prediction.forces, axis=1)))
            if success and len(prediction.forces)
            else np.nan
        )
        rows.append(
            {
                "Model": prediction.spec.label,
                "Model ID": prediction.spec.id,
                "Family": prediction.spec.family,
                "Dataset": prediction.spec.dataset,
                "Force Mode": prediction.spec.force_mode,
                "Total Energy (eV)": energy,
                "Energy per Atom (eV/atom)": (
                    energy / number_of_atoms if success else np.nan
                ),
                "Maximum Force (eV/Å)": force_max,
                "Runtime (s)": float(prediction.runtime_seconds),
                "Status": "Success" if success else "Failed",
                "Error": prediction.error or "",
                "Stress Status": (
                    "Available"
                    if prediction.stress is not None
                    else (
                        f"Unavailable: {prediction.stress_error}"
                        if prediction.stress_error
                        else "Not available"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def prepare_consensus_result(
    atoms: Atoms,
    predictions: Sequence[ModelPrediction],
    dataset: str | None = None,
) -> ConsensusResult:
    """Calculate consensus metrics from completed model predictions.

    ``energy_per_atom`` is only the model's total system energy divided by the
    atom count; it is not an atomwise energy decomposition.  Force
    disagreement measures committee consensus and is not a calibrated error.
    """

    atoms = validate_single_structure(atoms)
    normalised = [
        _validated_prediction(prediction, len(atoms))
        for prediction in predictions
    ]
    model_table = build_model_status_table(normalised, len(atoms))
    successful = [prediction for prediction in normalised if prediction.error is None]
    if len(successful) < MIN_CONSENSUS_MODELS:
        details = [
            f"{prediction.spec.label}: {prediction.error}"
            for prediction in normalised
            if prediction.error
        ]
        suffix = f" Failures: {'; '.join(details)}" if details else ""
        raise ConsensusCalculationError(
            "Model Consensus requires at least two successful model "
            f"predictions; received {len(successful)}.{suffix}",
            predictions=normalised,
            model_table=model_table,
        )

    datasets = {prediction.spec.dataset for prediction in normalised}
    result_dataset = dataset or successful[0].spec.dataset
    if result_dataset not in datasets or len(datasets) != 1:
        raise ConsensusValidationError(
            "Predictions do not all belong to the same curated dataset."
        )

    energies = np.asarray(
        [prediction.energy for prediction in successful],
        dtype=float,
    )
    energy_per_atom = energies / len(atoms)
    forces = np.stack(
        [np.asarray(prediction.forces, dtype=float) for prediction in successful],
        axis=0,
    )
    mean_forces = forces.mean(axis=0)
    delta = forces - mean_forces[None, :, :]
    force_disagreement = np.sqrt(
        np.mean(np.sum(delta**2, axis=2), axis=0)
    )

    difference = forces[:, None, :, :] - forces[None, :, :, :]
    pairwise_values = np.sqrt(np.mean(difference**2, axis=(2, 3)))
    np.fill_diagonal(pairwise_values, 0.0)
    labels = [prediction.spec.label for prediction in successful]
    pairwise = pd.DataFrame(pairwise_values, index=labels, columns=labels)

    metrics = {
        "energy_std_per_atom": float(np.std(energy_per_atom)),
        "energy_range_per_atom": float(np.ptp(energy_per_atom)),
        "force_disagreement_max": float(np.max(force_disagreement)),
        "force_disagreement_mean": float(np.mean(force_disagreement)),
        "force_disagreement_rms": float(
            np.sqrt(np.mean(force_disagreement**2))
        ),
        "force_disagreement_p95": float(
            np.percentile(force_disagreement, 95)
        ),
    }

    mean_force_magnitude = np.linalg.norm(mean_forces, axis=1)
    atom_table = pd.DataFrame(
        {
            "Atom Index": np.arange(len(atoms), dtype=int),
            "Element": atoms.get_chemical_symbols(),
            "Mean Fx (eV/Å)": mean_forces[:, 0],
            "Mean Fy (eV/Å)": mean_forces[:, 1],
            "Mean Fz (eV/Å)": mean_forces[:, 2],
            "Mean Force Magnitude (eV/Å)": mean_force_magnitude,
            "Force Disagreement (eV/Å)": force_disagreement,
        }
    ).sort_values(
        "Force Disagreement (eV/Å)",
        ascending=False,
        kind="stable",
        ignore_index=True,
    )

    return ConsensusResult(
        dataset=result_dataset,
        predictions=normalised,
        successful_predictions=successful,
        model_table=model_table,
        atom_table=atom_table,
        pairwise_force_rmse=pairwise,
        mean_forces=mean_forces,
        force_disagreement=force_disagreement,
        metrics=metrics,
    )


def generate_perturbed_configurations(
    atoms: Atoms,
    configuration_count: int,
    max_rms_displacement: float,
    random_seed: int,
) -> tuple[np.ndarray, list[Atoms]]:
    """Generate a reproducible 1D collective-displacement PES path.

    The first point is the unperturbed structure. Remaining points follow one
    fixed random collective direction whose center-of-mass translation is
    removed for multi-atom systems. The path coordinate is the RMS Cartesian
    displacement per atom in ångström.
    """

    atoms = validate_single_structure(atoms)
    if not 5 <= int(configuration_count) <= 10:
        raise ConsensusValidationError(
            "The perturbation PES scan requires between 5 and 10 perturbed "
            "configurations."
        )
    amplitude = float(max_rms_displacement)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ConsensusValidationError(
            "Maximum RMS displacement must be a positive finite value."
        )

    rng = np.random.default_rng(int(random_seed))
    direction = rng.normal(size=(len(atoms), 3))
    if len(atoms) > 1:
        direction -= direction.mean(axis=0, keepdims=True)
    direction_rms = float(
        np.sqrt(np.mean(np.sum(direction**2, axis=1)))
    )
    if not np.isfinite(direction_rms) or direction_rms <= np.finfo(float).eps:
        raise ConsensusValidationError(
            "Could not generate a non-zero atomic perturbation direction."
        )
    direction /= direction_rms

    path_coordinates = np.linspace(
        0.0,
        amplitude,
        int(configuration_count) + 1,
        dtype=float,
    )
    original_positions = np.asarray(atoms.positions, dtype=float)
    configurations: list[Atoms] = []
    for coordinate in path_coordinates:
        configuration = atoms.copy()
        configuration.calc = None
        configuration.set_positions(
            original_positions + coordinate * direction,
            apply_constraint=False,
        )
        configurations.append(configuration)
    return path_coordinates, configurations


def _validated_perturbation_prediction(
    prediction: PerturbationPrediction,
    number_of_atoms: int,
) -> PerturbationPrediction:
    checked = _validated_prediction(
        ModelPrediction(
            spec=prediction.spec,
            energy=prediction.energy,
            forces=prediction.forces,
            stress=None,
            runtime_seconds=prediction.runtime_seconds,
            error=prediction.error,
        ),
        number_of_atoms,
    )
    return replace(
        prediction,
        energy=checked.energy,
        forces=checked.forces,
        error=checked.error,
    )


def prepare_perturbation_result(
    atoms: Atoms,
    predictions: Sequence[PerturbationPrediction],
    path_coordinates: Sequence[float],
    *,
    max_rms_displacement: float,
    random_seed: int,
) -> PerturbationResult:
    """Prepare model-wise energy and force comparisons along a PES slice."""

    atoms = validate_single_structure(atoms)
    coordinates = np.asarray(path_coordinates, dtype=float)
    if (
        coordinates.ndim != 1
        or not 6 <= len(coordinates) <= 11
        or not np.all(np.isfinite(coordinates))
    ):
        raise ConsensusValidationError(
            "Perturbation path coordinates must contain the original plus "
            "5 to 10 finite perturbed configurations."
        )
    if not np.isclose(coordinates[0], 0.0):
        raise ConsensusValidationError(
            "The first perturbation configuration must be the original "
            "zero-displacement structure."
        )

    normalised = [
        _validated_perturbation_prediction(prediction, len(atoms))
        for prediction in predictions
    ]
    for prediction in normalised:
        index = int(prediction.configuration_index)
        if index < 0 or index >= len(coordinates):
            raise ConsensusValidationError(
                "A perturbation prediction has an invalid configuration index."
            )
        if not np.isclose(prediction.path_coordinate, coordinates[index]):
            raise ConsensusValidationError(
                "A perturbation prediction does not match its path coordinate."
            )

    force_rmse_to_consensus: dict[tuple[int, str], float] = {}
    summary_rows: list[dict[str, Any]] = []
    for configuration_index, coordinate in enumerate(coordinates):
        configuration_predictions = [
            prediction
            for prediction in normalised
            if prediction.configuration_index == configuration_index
        ]
        successful = [
            prediction
            for prediction in configuration_predictions
            if prediction.error is None
        ]

        energy_std = np.nan
        energy_range = np.nan
        force_max = np.nan
        force_rms = np.nan
        force_p95 = np.nan
        if len(successful) >= MIN_CONSENSUS_MODELS:
            energies = np.asarray(
                [prediction.energy for prediction in successful],
                dtype=float,
            )
            energy_per_atom = energies / len(atoms)
            force_stack = np.stack(
                [
                    np.asarray(prediction.forces, dtype=float)
                    for prediction in successful
                ],
                axis=0,
            )
            mean_forces = force_stack.mean(axis=0)
            force_delta = force_stack - mean_forces[None, :, :]
            atomic_disagreement = np.sqrt(
                np.mean(np.sum(force_delta**2, axis=2), axis=0)
            )
            energy_std = float(np.std(energy_per_atom))
            energy_range = float(np.ptp(energy_per_atom))
            force_max = float(np.max(atomic_disagreement))
            force_rms = float(
                np.sqrt(np.mean(atomic_disagreement**2))
            )
            force_p95 = float(np.percentile(atomic_disagreement, 95))
            for prediction, model_forces in zip(successful, force_stack):
                force_rmse_to_consensus[
                    (configuration_index, prediction.spec.id)
                ] = float(np.sqrt(np.mean((model_forces - mean_forces) ** 2)))

        summary_rows.append(
            {
                "Configuration": (
                    "Original"
                    if configuration_index == 0
                    else f"Perturbed {configuration_index}"
                ),
                "Configuration Index": configuration_index,
                "RMS Displacement (Å)": float(coordinate),
                "Successful Models": len(successful),
                "Failed Models": len(configuration_predictions) - len(successful),
                "Energy Std / Atom (eV/atom)": energy_std,
                "Energy Range / Atom (eV/atom)": energy_range,
                "Maximum Force Disagreement (eV/Å)": force_max,
                "RMS Force Disagreement (eV/Å)": force_rms,
                "P95 Force Disagreement (eV/Å)": force_p95,
            }
        )

    reference_energy = {
        prediction.spec.id: float(prediction.energy)
        for prediction in normalised
        if (
            prediction.configuration_index == 0
            and prediction.error is None
            and prediction.energy is not None
        )
    }
    detail_rows: list[dict[str, Any]] = []
    for prediction in normalised:
        success = (
            prediction.error is None
            and prediction.energy is not None
            and prediction.forces is not None
        )
        energy = float(prediction.energy) if success else np.nan
        reference = reference_energy.get(prediction.spec.id)
        relative_energy = (
            energy - reference
            if success and reference is not None
            else np.nan
        )
        force_magnitudes = (
            np.linalg.norm(np.asarray(prediction.forces), axis=1)
            if success
            else np.asarray([], dtype=float)
        )
        detail_rows.append(
            {
                "Configuration": (
                    "Original"
                    if prediction.configuration_index == 0
                    else f"Perturbed {prediction.configuration_index}"
                ),
                "Configuration Index": prediction.configuration_index,
                "RMS Displacement (Å)": float(prediction.path_coordinate),
                "Model": prediction.spec.label,
                "Model ID": prediction.spec.id,
                "Family": prediction.spec.family,
                "Total Energy (eV)": energy,
                "Relative Energy (eV)": relative_energy,
                "Relative Energy / Atom (eV/atom)": (
                    relative_energy / len(atoms)
                    if np.isfinite(relative_energy)
                    else np.nan
                ),
                "Maximum Force Magnitude (eV/Å)": (
                    float(np.max(force_magnitudes))
                    if force_magnitudes.size
                    else np.nan
                ),
                "Mean Force Magnitude (eV/Å)": (
                    float(np.mean(force_magnitudes))
                    if force_magnitudes.size
                    else np.nan
                ),
                "Force RMSE vs Consensus (eV/Å)": (
                    force_rmse_to_consensus.get(
                        (
                            prediction.configuration_index,
                            prediction.spec.id,
                        ),
                        np.nan,
                    )
                ),
                "Runtime (s)": float(prediction.runtime_seconds),
                "Status": "Success" if success else "Failed",
                "Error": prediction.error or "",
            }
        )

    return PerturbationResult(
        configuration_count=len(coordinates),
        max_rms_displacement=float(max_rms_displacement),
        random_seed=int(random_seed),
        path_coordinates=coordinates,
        predictions=normalised,
        model_configuration_table=pd.DataFrame(detail_rows),
        configuration_summary_table=pd.DataFrame(summary_rows),
    )


def _collect_released_model_resources() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError):
        pass


def _calculate_with_loaded_model(
    atoms: Atoms,
    spec: ConsensusModelSpec,
    calculator: Any,
    *,
    charge: int | float | None,
    spin: int | None,
    include_stress: bool,
) -> tuple[float, np.ndarray, np.ndarray | None, str | None]:
    calc_atoms = atoms.copy()
    calc_atoms.calc = None
    if charge is not None:
        calc_atoms.info["charge"] = charge
        calc_atoms.info["total_charge"] = charge
    if spin is not None:
        calc_atoms.info["spin"] = spin
        calc_atoms.info["total_spin"] = spin
    stress: np.ndarray | None = None
    stress_error: str | None = None
    try:
        calc_atoms.calc = calculator
        energy = float(calc_atoms.get_potential_energy())
        forces = np.asarray(calc_atoms.get_forces(), dtype=float).copy()
        if not np.array_equal(
            np.asarray(calc_atoms.numbers),
            np.asarray(atoms.numbers),
        ):
            raise ConsensusValidationError(
                "calculator changed the input atom ordering"
            )
        checked = _validated_prediction(
            ModelPrediction(
                spec=spec,
                energy=energy,
                forces=forces,
                stress=None,
                runtime_seconds=0.0,
                error=None,
            ),
            len(atoms),
        )
        if checked.error is not None:
            raise ConsensusValidationError(checked.error)

        if include_stress and np.any(calc_atoms.pbc):
            try:
                stress = _normalise_stress(calc_atoms.get_stress())
            except Exception as exc:
                stress_error = str(exc)
        return energy, forces, stress, stress_error
    finally:
        calc_atoms.calc = None
        del calc_atoms


def run_model_consensus(
    atoms: Atoms,
    model_specs: Sequence[ConsensusModelSpec],
    device: str,
    charge: int | float | None = None,
    spin: int | None = None,
    *,
    perturbation_count: int | None = None,
    perturbation_amplitude: float = 0.05,
    perturbation_seed: int = 42,
) -> ConsensusResult:
    """Run selected models sequentially, optionally along a 1D PES slice.

    When perturbation sampling is enabled, each calculator is constructed only
    once and reused across fresh copies of all path configurations. Calculators
    are still released one model at a time, so the committee is never resident
    in memory simultaneously.
    """

    validate_consensus_selection(atoms, model_specs)
    predictions: list[ModelPrediction] = []
    perturbation_predictions: list[PerturbationPrediction] = []
    path_coordinates: np.ndarray | None = None
    configurations: list[Atoms] = []
    if perturbation_count is not None:
        path_coordinates, configurations = generate_perturbed_configurations(
            atoms,
            perturbation_count,
            perturbation_amplitude,
            perturbation_seed,
        )

    for spec in model_specs:
        calculator: Any = None
        model_started = time.perf_counter()
        try:
            calculator = build_consensus_calculator(spec, device)
        except Exception as exc:
            error = f"{spec.label}: {exc}"
            runtime = time.perf_counter() - model_started
            predictions.append(
                ModelPrediction(
                    spec=spec,
                    energy=None,
                    forces=None,
                    stress=None,
                    runtime_seconds=runtime,
                    error=error,
                )
            )
            if path_coordinates is not None:
                for configuration_index, coordinate in enumerate(path_coordinates):
                    perturbation_predictions.append(
                        PerturbationPrediction(
                            spec=spec,
                            configuration_index=configuration_index,
                            path_coordinate=float(coordinate),
                            energy=None,
                            forces=None,
                            runtime_seconds=(
                                runtime if configuration_index == 0 else 0.0
                            ),
                            error=error,
                        )
                    )
            _collect_released_model_resources()
            continue

        base_started = model_started
        try:
            energy, forces, stress, stress_error = _calculate_with_loaded_model(
                atoms,
                spec,
                calculator,
                charge=charge,
                spin=spin,
                include_stress=True,
            )
            base_prediction = ModelPrediction(
                spec=spec,
                energy=energy,
                forces=forces,
                stress=stress,
                runtime_seconds=time.perf_counter() - base_started,
                error=None,
                stress_error=stress_error,
            )
        except Exception as exc:
            base_prediction = ModelPrediction(
                spec=spec,
                energy=None,
                forces=None,
                stress=None,
                runtime_seconds=time.perf_counter() - base_started,
                error=f"{spec.label}: {exc}",
            )
        predictions.append(base_prediction)

        if path_coordinates is not None:
            perturbation_predictions.append(
                PerturbationPrediction(
                    spec=spec,
                    configuration_index=0,
                    path_coordinate=float(path_coordinates[0]),
                    energy=base_prediction.energy,
                    forces=(
                        None
                        if base_prediction.forces is None
                        else np.asarray(base_prediction.forces).copy()
                    ),
                    runtime_seconds=base_prediction.runtime_seconds,
                    error=base_prediction.error,
                )
            )
            for configuration_index in range(1, len(configurations)):
                coordinate = float(path_coordinates[configuration_index])
                configuration_started = time.perf_counter()
                try:
                    energy, forces, _, _ = _calculate_with_loaded_model(
                        configurations[configuration_index],
                        spec,
                        calculator,
                        charge=charge,
                        spin=spin,
                        include_stress=False,
                    )
                    perturbation_prediction = PerturbationPrediction(
                        spec=spec,
                        configuration_index=configuration_index,
                        path_coordinate=coordinate,
                        energy=energy,
                        forces=forces,
                        runtime_seconds=(
                            time.perf_counter() - configuration_started
                        ),
                        error=None,
                    )
                except Exception as exc:
                    perturbation_prediction = PerturbationPrediction(
                        spec=spec,
                        configuration_index=configuration_index,
                        path_coordinate=coordinate,
                        energy=None,
                        forces=None,
                        runtime_seconds=(
                            time.perf_counter() - configuration_started
                        ),
                        error=f"{spec.label}: {exc}",
                    )
                perturbation_predictions.append(perturbation_prediction)

        if calculator is not None:
            del calculator
        _collect_released_model_resources()

    result = prepare_consensus_result(
        atoms,
        predictions,
        dataset=model_specs[0].dataset,
    )
    if path_coordinates is not None:
        result.perturbation = prepare_perturbation_result(
            atoms,
            perturbation_predictions,
            path_coordinates,
            max_rms_displacement=perturbation_amplitude,
            random_seed=perturbation_seed,
        )
    return result


def make_consensus_extxyz(
    atoms: Atoms,
    result: ConsensusResult,
) -> bytes:
    """Serialize consensus force arrays and scalar metadata as one extXYZ frame."""

    atoms = validate_single_structure(atoms)
    if result.mean_forces.shape != (len(atoms), 3):
        raise ConsensusValidationError(
            "Consensus mean-force array has an invalid shape."
        )
    if result.force_disagreement.shape != (len(atoms),):
        raise ConsensusValidationError(
            "Force-disagreement array has an invalid shape."
        )

    output = atoms.copy()
    output.calc = None
    output.arrays["consensus_mean_forces"] = np.asarray(
        result.mean_forces,
        dtype=float,
    ).copy()
    output.arrays["force_disagreement"] = np.asarray(
        result.force_disagreement,
        dtype=float,
    ).copy()
    output.info["consensus_dataset"] = str(result.dataset)
    output.info["consensus_models"] = "; ".join(
        prediction.spec.label for prediction in result.predictions
    )
    output.info["consensus_successful_models"] = "; ".join(
        prediction.spec.label for prediction in result.successful_predictions
    )
    for key in (
        "energy_std_per_atom",
        "energy_range_per_atom",
        "force_disagreement_max",
        "force_disagreement_mean",
        "force_disagreement_rms",
        "force_disagreement_p95",
    ):
        output.info[key] = float(result.metrics[key])

    buffer = io.StringIO()
    write(buffer, output, format="extxyz")
    return buffer.getvalue().encode("utf-8")


def render_perturbation_results(result: PerturbationResult) -> None:
    """Render the optional 1D PES and force-comparison trajectory."""

    import plotly.graph_objects as go
    import streamlit as st

    st.markdown("---")
    st.markdown("## Perturbed-Structure PES Comparison")
    st.caption(
        f"{result.configuration_count - 1} perturbed configurations plus the "
        "original reference, along one reproducible collective displacement "
        f"direction (seed {result.random_seed}), from 0 to "
        f"{result.max_rms_displacement:.3f} Å RMS displacement. This is a "
        "one-dimensional PES slice, not a complete potential-energy surface."
    )
    st.info(
        "Each model's relative energy is referenced to its own prediction for "
        "the original structure. This removes constant energy-reference offsets "
        "when comparing PES shapes."
    )

    detail = result.model_configuration_table
    successful = detail[detail["Status"] == "Success"].copy()
    energy_plot = go.Figure()
    for model_label, rows in successful.groupby("Model", sort=False):
        rows = rows.sort_values("Configuration Index")
        valid = rows["Relative Energy / Atom (eV/atom)"].notna()
        if valid.any():
            energy_plot.add_trace(
                go.Scatter(
                    x=rows.loc[valid, "RMS Displacement (Å)"],
                    y=rows.loc[valid, "Relative Energy / Atom (eV/atom)"],
                    mode="lines+markers",
                    name=model_label,
                    hovertemplate=(
                        f"{model_label}<br>"
                        "RMS displacement: %{x:.4f} Å<br>"
                        "Relative energy: %{y:.6f} eV/atom<extra></extra>"
                    ),
                )
            )
    energy_plot.update_layout(
        title="1D PES Slice: Relative Energy by Model",
        xaxis_title="RMS Atomic Displacement (Å)",
        yaxis_title="Relative Energy (eV/atom)",
        template="plotly_white",
        hovermode="x unified",
        height=520,
    )
    st.plotly_chart(energy_plot, use_container_width=True)

    force_plot = go.Figure()
    for model_label, rows in successful.groupby("Model", sort=False):
        rows = rows.sort_values("Configuration Index")
        valid = rows["Force RMSE vs Consensus (eV/Å)"].notna()
        if valid.any():
            force_plot.add_trace(
                go.Scatter(
                    x=rows.loc[valid, "RMS Displacement (Å)"],
                    y=rows.loc[valid, "Force RMSE vs Consensus (eV/Å)"],
                    mode="lines+markers",
                    name=model_label,
                    hovertemplate=(
                        f"{model_label}<br>"
                        "RMS displacement: %{x:.4f} Å<br>"
                        "Force RMSE vs consensus: "
                        "%{y:.6f} eV/Å<extra></extra>"
                    ),
                )
            )
    force_plot.update_layout(
        title="Model Force RMSE Relative to the Committee Mean",
        xaxis_title="RMS Atomic Displacement (Å)",
        yaxis_title="Force RMSE vs Committee Mean (eV/Å)",
        template="plotly_white",
        hovermode="x unified",
        height=520,
    )
    st.plotly_chart(force_plot, use_container_width=True)

    configuration_summary = result.configuration_summary_table
    disagreement_plot = go.Figure()
    disagreement_plot.add_trace(
        go.Scatter(
            x=configuration_summary["RMS Displacement (Å)"],
            y=configuration_summary["RMS Force Disagreement (eV/Å)"],
            mode="lines+markers",
            name="RMS disagreement",
        )
    )
    disagreement_plot.add_trace(
        go.Scatter(
            x=configuration_summary["RMS Displacement (Å)"],
            y=configuration_summary[
                "Maximum Force Disagreement (eV/Å)"
            ],
            mode="lines+markers",
            name="Maximum atomic disagreement",
        )
    )
    disagreement_plot.update_layout(
        title="Committee Force Disagreement Along the PES Slice",
        xaxis_title="RMS Atomic Displacement (Å)",
        yaxis_title="Force Disagreement (eV/Å)",
        template="plotly_white",
        hovermode="x unified",
        height=480,
    )
    st.plotly_chart(disagreement_plot, use_container_width=True)

    with st.expander("Compare model force magnitudes"):
        magnitude_plot = go.Figure()
        for model_label, rows in successful.groupby("Model", sort=False):
            rows = rows.sort_values("Configuration Index")
            magnitude_plot.add_trace(
                go.Scatter(
                    x=rows["RMS Displacement (Å)"],
                    y=rows["Maximum Force Magnitude (eV/Å)"],
                    mode="lines+markers",
                    name=model_label,
                )
            )
        magnitude_plot.update_layout(
            title="Maximum Atomic Force Magnitude by Model",
            xaxis_title="RMS Atomic Displacement (Å)",
            yaxis_title="Maximum Force Magnitude (eV/Å)",
            template="plotly_white",
            hovermode="x unified",
            height=480,
        )
        st.plotly_chart(magnitude_plot, use_container_width=True)

    st.markdown("### PES Configuration Summary")
    st.dataframe(
        configuration_summary,
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Per-model trajectory values and errors"):
        st.dataframe(detail, use_container_width=True, hide_index=True)

    incomplete = configuration_summary[
        configuration_summary["Successful Models"] < MIN_CONSENSUS_MODELS
    ]
    if not incomplete.empty:
        st.warning(
            f"{len(incomplete)} configuration(s) had fewer than two successful "
            "models, so committee disagreement is unavailable at those points."
        )

    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download PES Model Values CSV",
        data=detail.to_csv(index=False).encode("utf-8"),
        file_name="model_consensus_pes_models.csv",
        mime="text/csv",
        key="download_consensus_pes_models",
    )
    download_columns[1].download_button(
        "Download PES Summary CSV",
        data=configuration_summary.to_csv(index=False).encode("utf-8"),
        file_name="model_consensus_pes_summary.csv",
        mime="text/csv",
        key="download_consensus_pes_summary",
    )


def render_consensus_results(
    result: ConsensusResult,
    atoms: Atoms,
    *,
    force_tolerance: float | None = None,
    energy_tolerance: float | None = None,
) -> None:
    """Render consensus cards, tables, plots, warnings, and downloads."""

    import plotly.graph_objects as go
    import streamlit as st

    successful_count = len(result.successful_predictions)
    failed_count = len(result.predictions) - successful_count
    metrics = result.metrics

    st.success("Model Consensus calculation completed.")
    st.markdown("### Consensus Summary")
    summary_cards = (
        ("Successful Models", str(successful_count)),
        ("Failed Models", str(failed_count)),
        (
            "Energy Std / Atom",
            f"{metrics['energy_std_per_atom']:.4f} eV/atom",
        ),
        (
            "Energy Range / Atom",
            f"{metrics['energy_range_per_atom']:.4f} eV/atom",
        ),
        (
            "Maximum Force Disagreement",
            f"{metrics['force_disagreement_max']:.4f} eV/Å",
        ),
        (
            "95th-Percentile Force Disagreement",
            f"{metrics['force_disagreement_p95']:.4f} eV/Å",
        ),
        (
            "RMS Force Disagreement",
            f"{metrics['force_disagreement_rms']:.4f} eV/Å",
        ),
    )
    cards_html = "".join(
        (
            '<div class="consensus-summary-card">'
            f'<div class="consensus-summary-label">{label}</div>'
            f'<div class="consensus-summary-value">{value}</div>'
            "</div>"
        )
        for label, value in summary_cards
    )
    st.markdown(
        f"""
        <style>
        .consensus-summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
            gap: 0.75rem;
            margin: 0.25rem 0 1rem;
        }}
        .consensus-summary-card {{
            min-width: 0;
            padding: 0.8rem 0.9rem;
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 0.55rem;
            background: rgba(128, 128, 128, 0.08);
        }}
        .consensus-summary-label {{
            min-height: 2.5em;
            font-size: 0.86rem;
            line-height: 1.25;
            font-weight: 600;
            overflow-wrap: anywhere;
        }}
        .consensus-summary-value {{
            margin-top: 0.35rem;
            font-size: clamp(1.35rem, 2.2vw, 1.8rem);
            line-height: 1.15;
            font-variant-numeric: tabular-nums;
            overflow-wrap: anywhere;
        }}
        </style>
        <div class="consensus-summary-grid">{cards_html}</div>
        """,
        unsafe_allow_html=True,
    )

    if force_tolerance is not None and energy_tolerance is not None:
        within = (
            metrics["force_disagreement_max"] <= force_tolerance
            and metrics["energy_range_per_atom"] <= energy_tolerance
        )
        message = (
            "Within selected consensus tolerance"
            if within
            else "Outside selected consensus tolerance"
        )
        st.info(
            f"**{message}.** These are user-defined thresholds, not calibrated "
            "error bounds."
        )

    st.warning(CONSENSUS_WARNING)
    st.caption(
        "Energy per atom is the total system energy divided by atom count; "
        "it is not an atomwise energy decomposition. Energy disagreement is a "
        "secondary indicator because model implementations may use different "
        "reference offsets."
    )

    st.markdown("### Per-Model Status")
    st.dataframe(result.model_table, use_container_width=True, hide_index=True)

    failed = [
        prediction for prediction in result.predictions if prediction.error
    ]
    if failed:
        with st.expander("Individual model errors"):
            for prediction in failed:
                st.error(f"**{prediction.spec.label}:** {prediction.error}")

    st.markdown("### Per-Atom Consensus")
    st.caption("Sorted by force disagreement, highest first.")
    st.dataframe(result.atom_table, use_container_width=True, hide_index=True)

    force_plot = go.Figure(
        go.Bar(
            x=np.arange(len(atoms)),
            y=result.force_disagreement,
            customdata=atoms.get_chemical_symbols(),
            hovertemplate=(
                "Atom %{x} (%{customdata})<br>"
                "Disagreement: %{y:.6f} eV/Å<extra></extra>"
            ),
        )
    )
    force_plot.update_layout(
        title="Force Disagreement by Atom",
        xaxis_title="Atom Index",
        yaxis_title="Force Disagreement (eV/Å)",
        template="plotly_white",
        height=430,
    )
    st.plotly_chart(force_plot, use_container_width=True)

    heatmap = go.Figure(
        go.Heatmap(
            z=result.pairwise_force_rmse.values,
            x=result.pairwise_force_rmse.columns,
            y=result.pairwise_force_rmse.index,
            colorscale="Viridis",
            colorbar={"title": "eV/Å"},
            hovertemplate=(
                "%{y} vs %{x}<br>Force RMSE: %{z:.6f} eV/Å<extra></extra>"
            ),
        )
    )
    heatmap.update_layout(
        title="Pairwise Force RMSE",
        template="plotly_white",
        height=520,
    )
    st.plotly_chart(heatmap, use_container_width=True)

    energy_rows = result.model_table[
        result.model_table["Status"] == "Success"
    ]
    energy_plot = go.Figure(
        go.Bar(
            x=energy_rows["Model"],
            y=energy_rows["Energy per Atom (eV/atom)"],
            hovertemplate="%{x}<br>%{y:.6f} eV/atom<extra></extra>",
        )
    )
    energy_plot.update_layout(
        title="Total Energy Normalized by Atom Count",
        xaxis_title="Model",
        yaxis_title="Energy (eV/atom)",
        template="plotly_white",
        height=430,
    )
    st.plotly_chart(energy_plot, use_container_width=True)

    if result.perturbation is not None:
        render_perturbation_results(result.perturbation)

    st.markdown("### Downloads")
    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download Model Summary CSV",
        data=result.model_table.to_csv(index=False).encode("utf-8"),
        file_name="model_consensus_summary.csv",
        mime="text/csv",
        key="download_consensus_model_csv",
    )
    download_columns[1].download_button(
        "Download Atom Consensus CSV",
        data=result.atom_table.to_csv(index=False).encode("utf-8"),
        file_name="model_consensus_atoms.csv",
        mime="text/csv",
        key="download_consensus_atom_csv",
    )
    download_columns[2].download_button(
        "Download Consensus extXYZ",
        data=make_consensus_extxyz(atoms, result),
        file_name="model_consensus.extxyz",
        mime="chemical/x-extxyz",
        key="download_consensus_extxyz",
    )
