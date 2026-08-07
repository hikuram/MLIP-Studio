"""Generate candidate crystal structures with Chemeleon-DNG."""

from __future__ import annotations

import io
import json
import random
import re
import sys
import tempfile
import time
import traceback
import uuid
import zipfile
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from ase import Atoms
from ase.io import read, write
from pymatgen.core import Composition
# from ui_background import get_local_video_uri


PAGE_TITLE = "Generate Material"
GENERATOR_NAME = "Chemeleon-DNG"
TASK_NAME = "CSP"
SESSION_RESULTS_KEY = "generate_material_results"
SESSION_ERROR_KEY = "generate_material_error"
CHEMELEON_DNG_REPOSITORY_URL = "https://github.com/hspark1212/chemeleon-dng"
CHEMELEON_PAPER_URL = "https://doi.org/10.1038/s41467-025-59636-y"
DEFAULT_CSP_MODEL = "Alex-MP-20 (Chemeleon default)"
CSP_MODELS = {
    DEFAULT_CSP_MODEL: {
        "checkpoint": "chemeleon_csp_alex_mp_20_v0.0.2.ckpt",
        "model_path": None,
        "description": "Trained with the combined Alexandria and MP-20 data.",
    },
    "MP-20": {
        "checkpoint": "chemeleon_csp_mp_20_v0.0.2.ckpt",
        "model_path": "ckpts/chemeleon_csp_mp_20_v0.0.2.ckpt",
        "description": "Trained with MP-20 data.",
    },
}


class FormulaValidationError(ValueError):
    """Raised when a formula cannot be used for Chemeleon CSP."""


class DeviceUnavailableError(RuntimeError):
    """Raised when CUDA was requested but cannot be used."""


class NoStructuresGeneratedError(RuntimeError):
    """Raised when Chemeleon produces no discoverable structures."""


def validate_formula(formula: str) -> str:
    """Strip and validate a formula, returning the user-provided representation."""
    cleaned_formula = formula.strip()
    if not cleaned_formula:
        raise FormulaValidationError("Enter a chemical composition before generating structures.")

    try:
        composition = Composition(cleaned_formula)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormulaValidationError(
            f"'{cleaned_formula}' is not a valid chemical formula. Try NaCl, LiMnO2, or TiO2."
        ) from exc

    amounts = composition.get_el_amt_dict().values()
    if not amounts or any(amount <= 0 for amount in amounts):
        raise FormulaValidationError("The chemical formula must contain positive element amounts.")

    # Chemeleon-DNG 0.1.x expands each amount with range(int(amount)), so
    # fractional occupancies would otherwise be silently truncated.
    if any(not float(amount).is_integer() for amount in amounts):
        raise FormulaValidationError(
            "Chemeleon CSP requires whole-number stoichiometric amounts; "
            "fractional occupancies are not supported."
        )

    if any(not isinstance(getattr(element, "Z", None), int) for element in composition.elements):
        raise FormulaValidationError("The chemical formula contains an unknown element symbol.")

    return cleaned_formula


def cuda_is_available() -> bool:
    """Return CUDA availability without making page rendering depend on PyTorch."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, OSError, RuntimeError):
        return False
    except Exception:
        # Some vendor-specific torch builds can fail during CUDA inspection.
        return False


def resolve_device(selection: str) -> str:
    """Resolve the UI device selection to a Chemeleon device string."""
    normalized = selection.strip().lower()
    if normalized == "auto":
        return "cuda" if cuda_is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not cuda_is_available():
            raise DeviceUnavailableError(
                "CUDA was selected, but a usable CUDA device is not available. "
                "Choose Auto or CPU."
            )
        return "cuda"
    raise ValueError(f"Unknown compute-device selection: {selection!r}")


def find_generated_cifs(output_dir: Path) -> list[Path]:
    """Find every generated CIF recursively and sort paths deterministically."""
    cif_files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".cif"
    ]
    return sorted(
        cif_files,
        key=lambda path: (
            path.relative_to(output_dir).as_posix().casefold(),
            path.relative_to(output_dir).as_posix(),
        ),
    )


def read_generated_structure(path: Path) -> Atoms:
    """Read the first structure from a generated CIF using ASE."""
    structure = read(path, index=0, format="cif")
    if not isinstance(structure, Atoms):
        raise TypeError("ASE did not return an atomic structure for this CIF.")
    return structure


def sanitize_formula_for_filename(formula: str) -> str:
    """Return a short formula string that is safe to use in download names."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", formula).strip("._-")
    return (sanitized or "material")[:80]


def _returned_items(sample_result: Any) -> list[Any]:
    """Normalize common Chemeleon return shapes without iterating structures."""
    if sample_result is None:
        return []
    if isinstance(sample_result, (Atoms, str, Path)) or hasattr(
        sample_result, "to_ase_atoms"
    ):
        return [sample_result]
    if isinstance(sample_result, dict):
        return list(sample_result.values())
    if isinstance(sample_result, Iterable):
        return list(sample_result)
    return [sample_result]


def _as_ase_atoms(structure: Any) -> Atoms:
    if isinstance(structure, Atoms):
        return structure
    converter = getattr(structure, "to_ase_atoms", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Atoms):
            return converted
    raise TypeError(f"Unsupported returned structure type: {type(structure).__name__}")


def materialize_returned_structures(sample_result: Any, output_dir: Path) -> list[str]:
    """Write returned structures as CIFs when the sampler did not write files."""
    errors: list[str] = []
    for index, structure in enumerate(_returned_items(sample_result), start=1):
        destination = output_dir / f"returned_structure_{index:03d}.cif"
        try:
            if isinstance(structure, (str, Path)):
                source = Path(structure)
                if source.is_file() and source.suffix.lower() == ".cif":
                    destination.write_bytes(source.read_bytes())
                    continue
            atoms = _as_ase_atoms(structure)
            write(destination, atoms, format="cif")
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            errors.append(f"Returned structure {index}: {type(exc).__name__}: {exc}")
    return errors


def _generated_formula(atoms: Atoms) -> str:
    try:
        return Composition(atoms.get_chemical_formula()).reduced_formula
    except (TypeError, ValueError):
        return atoms.get_chemical_formula()


def _lattice_metadata(atoms: Atoms) -> tuple[list[float] | None, float | None]:
    cell = np.asarray(atoms.cell.array, dtype=float)
    if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) <= 1.0e-12:
        return None, None
    return [float(value) for value in atoms.cell.cellpar()], float(atoms.get_volume())


def _candidate_record(
    path: Path,
    index: int,
    safe_formula: str,
) -> tuple[dict[str, Any], Atoms | None]:
    record: dict[str, Any] = {
        "index": index,
        "name": f"Candidate {index}",
        "download_name": f"{safe_formula}_chemeleon_candidate_{index:03d}.cif",
        "cif_bytes": path.read_bytes(),
        "source_name": path.name,
        "parse_error": None,
    }
    try:
        atoms = read_generated_structure(path)
        lattice_parameters, volume = _lattice_metadata(atoms)
        record.update(
            {
                "formula": _generated_formula(atoms),
                "num_atoms": len(atoms),
                "lattice_parameters": lattice_parameters,
                "volume": volume,
                "symbols": atoms.get_chemical_symbols(),
                "positions": np.asarray(atoms.positions, dtype=float).tolist(),
                "cell": np.asarray(atoms.cell.array, dtype=float).tolist(),
                "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
            }
        )
        return record, atoms
    except Exception as exc:
        # A malformed candidate must not hide other valid Chemeleon outputs.
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
        record["parse_debug"] = sanitize_debug_details(traceback.format_exc())
        return record, None


def create_cif_zip(candidates: list[dict[str, Any]], run_metadata: dict[str, Any]) -> bytes:
    """Create an in-memory CIF archive with a machine-readable provenance manifest."""
    manifest = {
        **run_metadata,
        "candidates": [
            {
                "candidate_index": candidate["index"],
                "filename": candidate["download_name"],
                "generated_formula": candidate.get("formula"),
                "number_of_atoms": candidate.get("num_atoms"),
                "parsed_successfully": candidate.get("parse_error") is None,
            }
            for candidate in candidates
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(manifest, indent=2))
        for candidate in candidates:
            archive.writestr(candidate["download_name"], candidate["cif_bytes"])
    return buffer.getvalue()


def create_combined_extxyz(
    indexed_structures: list[tuple[int, Atoms]],
    run_metadata: dict[str, Any],
) -> bytes:
    """Create a combined extXYZ containing provenance on every parsed frame."""
    structures: list[Atoms] = []
    for candidate_index, original_atoms in indexed_structures:
        atoms = original_atoms.copy()
        atoms.info.update(
            {
                "generator": run_metadata["generator"],
                "task": run_metadata["task"],
                "requested_composition": run_metadata["requested_composition"],
                "requested_structures": run_metadata["requested_structures"],
                "device": run_metadata["device"],
                "model": run_metadata["model"],
                "checkpoint": run_metadata["checkpoint"],
                "generation_timestamp": run_metadata["generation_timestamp"],
                "generation_time_seconds": run_metadata["generation_time_seconds"],
                "candidate_index": candidate_index,
            }
        )
        structures.append(atoms)

    if not structures:
        raise ValueError("No successfully parsed structures are available for extXYZ export.")
    buffer = io.StringIO()
    write(buffer, structures, format="extxyz")
    return buffer.getvalue().encode("utf-8")


def load_chemeleon_sample() -> Callable[..., Any]:
    """Import Chemeleon lazily so the page can explain missing installations."""
    from chemeleon_dng.sample import sample

    return sample


def resolve_model_path(
    model_selection: str,
    prepare_checkpoint: bool = True,
) -> tuple[str, str | None]:
    """Return the selected checkpoint name and optional sample() model path."""
    try:
        configuration = CSP_MODELS[model_selection]
    except KeyError as exc:
        raise ValueError(f"Unknown Chemeleon CSP model: {model_selection!r}") from exc

    checkpoint = str(configuration["checkpoint"])
    model_path = configuration["model_path"]
    if model_path is not None and prepare_checkpoint:
        from chemeleon_dng.download_util import get_checkpoint_path

        model_path = get_checkpoint_path("csp", {"csp": str(model_path)})
    return checkpoint, str(model_path) if model_path is not None else None


def generate_candidates(
    formula: str,
    num_samples: int,
    device: str,
    sample_function: Callable[..., Any] | None = None,
    model_selection: str = DEFAULT_CSP_MODEL,
) -> dict[str, Any]:
    """Run Chemeleon and convert its temporary output into persistent byte data."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    run_id = uuid.uuid4().hex
    safe_formula = sanitize_formula_for_filename(formula)
    checkpoint, model_path = resolve_model_path(
        model_selection,
        prepare_checkpoint=sample_function is None,
    )
    run_metadata: dict[str, Any] = {
        "generator": GENERATOR_NAME,
        "task": TASK_NAME,
        "requested_composition": formula,
        "requested_structures": num_samples,
        "device": device,
        "model": model_selection,
        "checkpoint": checkpoint,
        "generation_timestamp": timestamp,
    }

    sampler = sample_function or load_chemeleon_sample()
    with tempfile.TemporaryDirectory(prefix="mlip_studio_chemeleon_") as temporary_dir:
        output_dir = Path(temporary_dir)
        generation_start = time.perf_counter()
        sample_kwargs = {
            "task": "csp",
            "formulas": [formula],
            "num_samples": num_samples,
            "output_dir": str(output_dir),
            "device": device,
        }
        if model_path is not None:
            sample_kwargs["model_path"] = model_path
        sample_result = sampler(**sample_kwargs)
        run_metadata["generation_time_seconds"] = time.perf_counter() - generation_start

        cif_files = find_generated_cifs(output_dir)
        return_conversion_errors: list[str] = []
        if not cif_files and sample_result is not None:
            return_conversion_errors = materialize_returned_structures(
                sample_result, output_dir
            )
            cif_files = find_generated_cifs(output_dir)

        if not cif_files:
            details = ""
            if return_conversion_errors:
                details = " " + " ".join(return_conversion_errors)
            raise NoStructuresGeneratedError(
                "Chemeleon completed without producing any CIF structures." + details
            )

        candidates: list[dict[str, Any]] = []
        indexed_structures: list[tuple[int, Atoms]] = []
        for index, cif_path in enumerate(cif_files, start=1):
            record, atoms = _candidate_record(cif_path, index, safe_formula)
            candidates.append(record)
            if atoms is not None:
                indexed_structures.append((index, atoms))

        result: dict[str, Any] = {
            "run_id": run_id,
            "metadata": run_metadata,
            "candidates": candidates,
            "parsed_count": len(indexed_structures),
            "zip_name": f"{safe_formula}_chemeleon_structures.zip",
            "extxyz_name": f"{safe_formula}_chemeleon_structures.extxyz",
            "zip_bytes": None,
            "zip_error": None,
            "extxyz_bytes": None,
            "extxyz_error": None,
        }

        try:
            result["zip_bytes"] = create_cif_zip(candidates, run_metadata)
        except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
            result["zip_error"] = f"{type(exc).__name__}: {exc}"

        if indexed_structures:
            try:
                result["extxyz_bytes"] = create_combined_extxyz(
                    indexed_structures, run_metadata
                )
            except (OSError, TypeError, ValueError) as exc:
                result["extxyz_error"] = f"{type(exc).__name__}: {exc}"

        return result


def sanitize_debug_details(details: str) -> str:
    """Redact common absolute server paths from optional debug output."""
    replacements = {
        str(Path(__file__).resolve().parents[1]): "<application>",
        str(Path(sys.prefix).resolve()): "<python-environment>",
        str(Path(tempfile.gettempdir()).resolve()): "<temporary-directory>",
    }
    sanitized = details
    for path, replacement in sorted(replacements.items(), key=lambda item: -len(item[0])):
        sanitized = re.sub(re.escape(path), replacement, sanitized, flags=re.IGNORECASE)
    # Redact remaining absolute Windows paths that may come from third-party errors.
    sanitized = re.sub(
        r"(?i)\b[A-Z]:\\(?:[^\r\n\"']+\\)+",
        lambda _match: "<server-path>\\",
        sanitized,
    )
    return sanitized


def describe_generation_error(exc: Exception) -> str:
    """Translate common package/checkpoint/device failures into concise UI errors."""
    if isinstance(exc, DeviceUnavailableError):
        return str(exc)
    if isinstance(exc, NoStructuresGeneratedError):
        return "No generated CIF files were found. Chemeleon did not return a usable structure."
    if isinstance(exc, ModuleNotFoundError):
        if exc.name and exc.name.startswith("chemeleon_dng"):
            return (
                "Chemeleon-DNG is not installed in the application environment. "
                "Declare and install the 'chemeleon-dng' package before using this page."
            )
        missing_name = exc.name or "an unknown dependency"
        return f"Chemeleon-DNG could not start because the dependency '{missing_name}' is missing."
    if isinstance(exc, ImportError):
        return "Chemeleon-DNG or one of its required dependencies could not be imported."
    if isinstance(exc, FileNotFoundError):
        return (
            "The Chemeleon CSP checkpoint could not be found. Check the model files "
            "and the server's checkpoint configuration."
        )

    module_name = type(exc).__module__
    message = str(exc)
    if module_name.startswith("requests") or "checkpoint" in message.casefold() and any(
        word in message.casefold() for word in ("download", "connection", "http", "timeout")
    ):
        return (
            "The Chemeleon checkpoint download failed. Check network access and retry, "
            "or provide the checkpoint in the server environment."
        )
    if "cuda" in message.casefold():
        return (
            "Chemeleon could not use CUDA in this environment. Choose CPU, or verify the "
            "PyTorch and CUDA installation."
        )
    return "Chemeleon generation failed. See debugging details below for the underlying error."


def chemeleon_environment_warning() -> str | None:
    """Report installed-package Python metadata incompatibility, if detectable."""
    try:
        distribution = importlib_metadata.distribution("chemeleon-dng")
    except importlib_metadata.PackageNotFoundError:
        return None

    required_python = distribution.metadata.get("Requires-Python")
    if not required_python:
        return None
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        current_version = Version(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        if not SpecifierSet(required_python).contains(current_version):
            return (
                f"Chemeleon-DNG {distribution.version} declares Python {required_python}, "
                f"but this server is running Python {current_version}. Generation may fail "
                "until the application environment is aligned."
            )
    except (ImportError, ValueError):
        return None
    return None


def get_structure_viz(
    atoms: Atoms,
    style: str = "ball-stick",
    spin: bool = True,
    show_unit_cell: bool = True,
    width: int = 500,
    height: int = 430,
) -> Any:
    """Build the same py3Dmol structure view used by MLIP Studio's Home page."""
    import py3Dmol

    xyz_lines = [str(len(atoms)), "Chemeleon candidate"]
    xyz_lines.extend(
        f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}"
        for atom in atoms
    )
    view = py3Dmol.view(width=width, height=height)
    view.addModel("\n".join(xyz_lines) + "\n", "xyz")

    if style == "ball-stick":
        view.setStyle({"stick": {"radius": 0.2}, "sphere": {"scale": 0.3}})
    elif style == "stick":
        view.setStyle({"stick": {}})
    else:
        view.setStyle({"sphere": {"scale": 0.4}})

    if show_unit_cell and bool(np.asarray(atoms.pbc).any()):
        cell = np.asarray(atoms.cell.array, dtype=float)
        if cell.shape == (3, 3) and bool(cell.any()):
            origin = np.zeros(3)
            a, b, c = cell
            edges = [
                (origin, a),
                (origin, b),
                (a, a + b),
                (b, a + b),
                (c, c + a),
                (c, c + b),
                (c + a, c + a + b),
                (c + b, c + a + b),
                (origin, c),
                (a, a + c),
                (b, b + c),
                (a + b, a + b + c),
            ]
            for start, end in edges:
                view.addCylinder(
                    {
                        "start": dict(zip(("x", "y", "z"), map(float, start))),
                        "end": dict(zip(("x", "y", "z"), map(float, end))),
                        "radius": 0.05,
                        "color": "black",
                        "alpha": 0.7,
                    }
                )
    view.zoomTo()
    view.setBackgroundColor("white")
    view.spin(spin)
    return view


def _atoms_from_candidate(candidate: dict[str, Any]) -> Atoms:
    return Atoms(
        symbols=candidate["symbols"],
        positions=candidate["positions"],
        cell=candidate["cell"],
        pbc=candidate["pbc"],
    )


def _apply_home_page_chrome() -> None:
    """Apply the small background treatment and sidebar conventions from Home."""
    st.markdown(
        """
        <style>
            #myVideo {
                position: fixed;
                right: 0;
                bottom: 0;
                min-width: 100%;
                min-height: 100%;
                opacity: 0.08;
                pointer-events: none;
            }
            .content {
                position: fixed;
                bottom: 0;
                background: rgba(1, 1, 1, 1.0);
                color: #f1f1f1;
                width: 100%;
                padding: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    
    
    


def _render_last_error() -> None:
    error = st.session_state.get(SESSION_ERROR_KEY)
    if not error:
        return
    st.error(error["message"])
    if error.get("debug"):
        with st.expander("Debugging details"):
            st.code(error["debug"], language="text")


def _render_overview(candidates: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        lattice = candidate.get("lattice_parameters")
        rows.append(
            {
                "Candidate": candidate["index"],
                "Formula": candidate.get("formula", "Could not parse"),
                "Atoms": candidate.get("num_atoms"),
                "a (Å)": round(lattice[0], 4) if lattice else None,
                "b (Å)": round(lattice[1], 4) if lattice else None,
                "c (Å)": round(lattice[2], 4) if lattice else None,
                "Volume (Å³)": round(candidate["volume"], 4)
                if candidate.get("volume") is not None
                else None,
                "Status": "Parsed" if candidate.get("parse_error") is None else "Invalid CIF",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_candidate(candidate: dict[str, Any], result: dict[str, Any]) -> None:
    st.markdown(f"### {candidate['name']}")
    viewer_column, details_column = st.columns([3, 2])

    with viewer_column:
        if candidate.get("parse_error") is not None:
            st.error("This CIF could not be parsed, so a 3D preview is unavailable.")
            st.caption(candidate["parse_error"])
            if candidate.get("parse_debug"):
                with st.expander("Candidate parsing details"):
                    st.code(candidate["parse_debug"], language="text")
        else:
            viewer_controls = st.columns(2)
            with viewer_controls[0]:
                style = st.selectbox(
                    "Visualization style",
                    ["ball-stick", "stick", "ball"],
                    key=(
                        f"generate_material_style_{result['run_id']}_"
                        f"{candidate['index']}"
                    ),
                )
            with viewer_controls[1]:
                spin = st.checkbox(
                    "Spin",
                    value=True,
                    key=(
                        f"generate_material_spin_{result['run_id']}_"
                        f"{candidate['index']}"
                    ),
                )
            try:
                view = get_structure_viz(
                    _atoms_from_candidate(candidate), style=style, spin=spin
                )
                st.components.v1.html(view._make_html(), height=450)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                st.error("The structure viewer could not render this candidate.")
                with st.expander("Viewer debugging details"):
                    st.code(
                        sanitize_debug_details(f"{type(exc).__name__}: {exc}"),
                        language="text",
                    )

    with details_column:
        st.markdown("#### Structure information")
        if candidate.get("parse_error") is None:
            st.write(f"**Generated formula:** {candidate['formula']}")
            st.write(f"**Number of atoms:** {candidate['num_atoms']}")
            lattice = candidate.get("lattice_parameters")
            if lattice:
                st.write(
                    "**Lattice:** "
                    f"a = {lattice[0]:.4f} Å, b = {lattice[1]:.4f} Å, "
                    f"c = {lattice[2]:.4f} Å"
                )
                st.write(
                    "**Angles:** "
                    f"α = {lattice[3]:.3f}°, β = {lattice[4]:.3f}°, "
                    f"γ = {lattice[5]:.3f}°"
                )
                st.write(f"**Cell volume:** {candidate['volume']:.4f} Å³")
            else:
                st.write("**Lattice:** Not available")

        metadata = result["metadata"]
        with st.expander("Provenance", expanded=True):
            st.write(f"**Generator:** {metadata['generator']}")
            st.write(f"**Task:** {metadata['task']}")
            st.write(f"**Requested composition:** {metadata['requested_composition']}")
            st.write(f"**Requested structures:** {metadata['requested_structures']}")
            st.write(f"**Device:** {metadata['device']}")
            st.write(f"**Model:** {metadata['model']}")
            st.write(f"**Checkpoint:** {metadata['checkpoint']}")
            st.write(f"**Generation timestamp:** {metadata['generation_timestamp']}")
            st.write(
                "**Chemeleon generation time:** "
                f"{metadata['generation_time_seconds']:.2f} s"
            )
            st.write(f"**Candidate index:** {candidate['index']}")

        st.download_button(
            "Download this CIF",
            data=candidate["cif_bytes"],
            file_name=candidate["download_name"],
            mime="chemical/x-cif",
            key=f"generate_material_cif_{result['run_id']}_{candidate['index']}",
            use_container_width=True,
        )


def _render_results(result: dict[str, Any]) -> None:
    candidates = result["candidates"]
    parsed_count = result["parsed_count"]
    st.markdown("## Generated structures")
    st.success(
        f"Chemeleon generated {len(candidates)} CIF file(s); "
        f"{parsed_count} parsed successfully."
    )
    st.metric(
        "Chemeleon generation time",
        f"{result['metadata']['generation_time_seconds']:.2f} s",
        help=(
            "Elapsed time inside the Chemeleon sample() call. CIF discovery, parsing, "
            "visualization, and download preparation are excluded."
        ),
    )
    st.caption(
        "Timing covers only the Chemeleon generation call and excludes CIF parsing and "
        "result preparation."
    )
    if parsed_count == 0:
        st.error(
            "None of the generated CIF files could be parsed. The original CIF downloads "
            "remain available for inspection."
        )

    download_columns = st.columns(2)
    with download_columns[0]:
        if result.get("zip_bytes") is not None:
            st.download_button(
                "Download all CIFs (ZIP)",
                data=result["zip_bytes"],
                file_name=result["zip_name"],
                mime="application/zip",
                key=f"generate_material_zip_{result['run_id']}",
                use_container_width=True,
            )
        else:
            st.error("The ZIP archive could not be created.")
            if result.get("zip_error"):
                with st.expander("ZIP debugging details"):
                    st.code(result["zip_error"], language="text")
    with download_columns[1]:
        if result.get("extxyz_bytes") is not None:
            st.download_button(
                "Download parsed structures (extXYZ)",
                data=result["extxyz_bytes"],
                file_name=result["extxyz_name"],
                mime="chemical/x-xyz",
                key=f"generate_material_extxyz_{result['run_id']}",
                use_container_width=True,
            )
        elif result.get("extxyz_error"):
            st.error("The combined extXYZ file could not be created.")
            with st.expander("extXYZ debugging details"):
                st.code(result["extxyz_error"], language="text")
        else:
            st.info("No parsed structures are available for a combined extXYZ download.")

    st.markdown("### Candidate overview")
    _render_overview(candidates)
    selected_index = st.selectbox(
        "Candidate to inspect",
        options=list(range(len(candidates))),
        format_func=lambda index: candidates[index]["name"],
        key=f"generate_material_selected_{result['run_id']}",
    )
    _render_candidate(candidates[selected_index], result)


def _render_footer() -> None:
    st.markdown("---")
    st.markdown(
        "Universal MLIP Studio App | Created with Streamlit, ASE, MACE, FairChem, "
        "SevenNet, ORB, MatterSim, UPET, Py3DMol, Pymatgen and ❤️"
    )
    st.markdown(
        "Developed by [Dr. Manas Sharma](https://manas.bragitoff.com/) in the groups "
        "of [Prof. Ananth Govind Rajan Group](https://www.agrgroup.org/) and "
        "[Prof. Sudeep Punnathanam](https://chemeng.iisc.ac.in/sudeep/) at "
        "[IISc Bangalore](https://iisc.ac.in/)"
    )
    st.markdown(
        "📄 **Preprint:** [MLIP Studio: A unified platform for benchmarking universal "
        "machine learning interatomic potentials](https://arxiv.org/html/2607.07606v1)",
        unsafe_allow_html=True,
    )


def _render_chemeleon_citation() -> None:
    with st.expander("Chemeleon-DNG code and citation"):
        st.markdown(
            f"**Code:** [Chemeleon-DNG on GitHub]({CHEMELEON_DNG_REPOSITORY_URL})"
        )
        st.markdown(
            "**Please cite:** H. Park, A. Onwuli, and A. Walsh, “Exploration of "
            "crystal chemical space using text-guided generative artificial "
            f"intelligence,” *Nature Communications* **16**, 4379 (2025). "
            f"[DOI: 10.1038/s41467-025-59636-y]({CHEMELEON_PAPER_URL})"
        )
        st.code(
            """@article{park2025exploration,
  author = {Park, Hyunsoo and Onwuli, Anthony and Walsh, Aron},
  title = {Exploration of crystal chemical space using text-guided
           generative artificial intelligence},
  journal = {Nature Communications},
  volume = {16},
  pages = {4379},
  year = {2025},
  doi = {10.1038/s41467-025-59636-y}
}""",
            language="bibtex",
        )


def main() -> None:
    st.set_page_config(
        page_title="Generate Material | MLIP Studio",
        page_icon="🧪",
        layout="wide",
    )
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if SESSION_RESULTS_KEY not in st.session_state:
        st.session_state[SESSION_RESULTS_KEY] = None
    if SESSION_ERROR_KEY not in st.session_state:
        st.session_state[SESSION_ERROR_KEY] = None

    _apply_home_page_chrome()

    st.markdown(f"## {PAGE_TITLE}")
    st.write(
        "#### Generate candidate crystal structures from a chemical composition using "
        "Chemeleon-DNG"
    )
    st.markdown(
        "Chemeleon performs formula-conditioned crystal structure prediction (CSP). "
        "Generated candidates should subsequently be relaxed and evaluated with an "
        "appropriate machine learning interatomic potential (MLIP) or DFT method."
    )
    st.info(
        "Generated structures are unrelaxed candidates and may be unstable, duplicated, "
        "or physically invalid. Relax and evaluate them using an appropriate MLIP or "
        "electronic-structure method before drawing scientific conclusions."
    )
    st.info(
        "Download the generated candidates as individual CIF files or as a combined "
        "extXYZ trajectory, then upload them on the Home page to evaluate energies "
        "and forces or optimize their geometries with any available MLIP."
    )
    try:
        st.page_link(
            "pages/Home.py",
            label="Open Home to evaluate or optimize these structures",
            icon="🏠",
        )
    except KeyError:
        # A standalone page smoke test has no multipage registry to resolve.
        st.caption("Open Home from the MLIP Studio navigation to continue.")

    # environment_warning = chemeleon_environment_warning()
    # if environment_warning:
    #     st.warning(environment_warning)

    with st.form("generate_material_form"):
        input_columns = st.columns([2, 1])
        with input_columns[0]:
            formula_input = st.text_input(
                "Chemical composition",
                placeholder="NaCl, LiMnO2, or TiO2",
                help="Enter one formula with whole-number stoichiometry.",
            )
        with input_columns[1]:
            num_samples = st.number_input(
                "Number of structures",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
            )
        with st.expander("Advanced settings"):
            model_selection = st.selectbox(
                "Chemeleon CSP model",
                options=list(CSP_MODELS),
                index=0,
                help=(
                    "Choose the pretrained formula-conditioned CSP checkpoint. "
                    "Chemeleon downloads its checkpoint bundle on first use."
                ),
            )
            st.caption(CSP_MODELS[model_selection]["description"])
            device_selection = st.selectbox(
                "Compute device",
                options=["Auto", "CPU", "CUDA"],
                index=0,
                help="Auto uses CUDA when available and otherwise uses CPU.",
            )
            st.caption(
                "The first run of a model may require the server to obtain the "
                "Chemeleon checkpoint bundle."
            )
        submitted = st.form_submit_button(
            "Generate Structures", type="primary", use_container_width=True
        )

    if submitted:
        st.session_state[SESSION_ERROR_KEY] = None
        try:
            # if not st.session_state.get("logged_in", False):
            #     raise PermissionError(
            #         "🔒 Please log in from the main page before generating structures."
            #     )
            formula = validate_formula(formula_input)
            resolved_device = resolve_device(device_selection)
            with st.spinner(
                f"Generating {int(num_samples)} {formula} candidate(s) on "
                f"{resolved_device.upper()}..."
            ):
                st.session_state[SESSION_RESULTS_KEY] = generate_candidates(
                    formula=formula,
                    num_samples=int(num_samples),
                    device=resolved_device,
                    model_selection=model_selection,
                )
        except PermissionError as exc:
            st.session_state[SESSION_ERROR_KEY] = {"message": str(exc), "debug": None}
        except FormulaValidationError as exc:
            st.session_state[SESSION_ERROR_KEY] = {"message": str(exc), "debug": None}
        except Exception as exc:
            st.session_state[SESSION_ERROR_KEY] = {
                "message": describe_generation_error(exc),
                "debug": sanitize_debug_details(traceback.format_exc()),
            }

    _render_last_error()
    result = st.session_state.get(SESSION_RESULTS_KEY)
    if result:
        _render_results(result)

    _render_chemeleon_citation()
    _render_footer()


if __name__ == "__main__":
    main()
