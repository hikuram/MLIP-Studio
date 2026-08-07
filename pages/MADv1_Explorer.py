#!/usr/bin/env python3
"""
MAD Explorer - Extended XYZ Trajectory Dataset Explorer
A powerful Streamlit app for querying, visualizing, and exporting structures
from mad-train.xyz, mad-test.xyz, and mad-val.xyz trajectory files.
"""

import streamlit as st
import pandas as pd
import py3Dmol
import streamlit.components.v1 as components
from ase import Atoms
from ase.io import read, write
from ase.formula import Formula
import io
import tempfile
import os
import time
import numpy as np
import pickle
import hashlib
import json
from pathlib import Path
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="MADv1 (PBEsol) Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stButton>button { width: 100%; }
    .stats-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
DATASET_FILES = {
    "train": "./pages/MAD_Dataset/mad-train.xyz",
    "test": "./pages/MAD_Dataset/mad-test.xyz",
    "val": "./pages/MAD_Dataset/mad-val.xyz",
}
CACHE_DIR = Path("./pages/mad_explorer_cache")
DIMENSIONALITY_LABELS = {0: "0D (Molecule/Cluster)", 1: "1D (Wire/Chain)", 2: "2D (Slab/Surface)", 3: "3D (Bulk)"}
MAD_PAPER_URL = "https://doi.org/10.1038/s41597-025-06109-y"
MAD_ARCHIVE_URL = "https://archive.materialscloud.org/record/2025.98"
MAD_DOWNLOAD_URLS = {
    "Training split": "https://archive.materialscloud.org/records/xdsbt-a3r17/files/mad-train.xyz?download=1",
    "Validation split": "https://archive.materialscloud.org/records/xdsbt-a3r17/files/mad-val.xyz?download=1",
    "Test split": "https://archive.materialscloud.org/records/xdsbt-a3r17/files/mad-test.xyz?download=1",
}

# --- Session state defaults ---
for key, default in [
    ('current_dataset', None),
    ('query_results', []),
    ('selected_indices', []),
    ('dataset_loaded', False),
    ('all_elements', []),
    ('stats', None),
    ('frames', []),
    ('viz_pos', 0),
    ('query_elapsed_seconds', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ======================================================================
# Utility helpers
# ======================================================================

def _file_hash(path: str) -> str:
    """Return a short hash based on file path + size + mtime for cache invalidation."""
    p = Path(path)
    if not p.exists():
        return ""
    stat = p.stat()
    key = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _cache_path(dataset_name: str, suffix: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    h = _file_hash(DATASET_FILES[dataset_name])
    return CACHE_DIR / f"{dataset_name}_{h}_{suffix}"


def classify_dimensionality(atoms: Atoms) -> int:
    """
    Classify structure dimensionality based on the cell geometry.
    - 3D: all three cell vector lengths are substantial and pbc is True in all directions
    - 2D: two periodic directions (or very thin vacuum in one direction)
    - 1D: one periodic direction
    - 0D: no periodicity / isolated molecule
    
    Heuristic: a cell vector direction is considered "periodic" if pbc is True
    along that axis AND the cell length in that direction > 0.5 Å.
    If pbc info is missing we fall back to cell-length heuristic with a vacuum
    threshold of 12 Å.
    """
    pbc = atoms.pbc
    cell_lengths = atoms.cell.lengths()

    # If pbc information is available and meaningful, use it
    if hasattr(atoms, 'pbc') and np.any(pbc):
        return int(np.sum(pbc))

    # Fallback: use vacuum gap heuristic
    VACUUM_THRESHOLD = 12.0  # Angstrom
    periodic_dirs = 0
    for i in range(3):
        if cell_lengths[i] < VACUUM_THRESHOLD and cell_lengths[i] > 0.5:
            periodic_dirs += 1
        elif cell_lengths[i] >= VACUUM_THRESHOLD:
            # Check if atoms actually span this direction
            positions = atoms.positions[:, i]
            span = positions.max() - positions.min() if len(positions) > 0 else 0
            gap = cell_lengths[i] - span
            if gap < VACUUM_THRESHOLD:
                periodic_dirs += 1
    return periodic_dirs

def extract_energy(atoms):
    """Extract energy from atoms, checking info, calc.results, and calculator methods."""
    # 1. Check calculator results first (most reliable for extxyz)
    if getattr(atoms, "calc", None) is not None and hasattr(atoms.calc, "results"):
        if isinstance(atoms.calc.results, dict) and 'energy' in atoms.calc.results:
            return float(atoms.calc.results['energy'])
    
    # 2. Try calculator method
    if getattr(atoms, "calc", None) is not None:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            pass
    
    # 3. Fall back to info dict
    info = getattr(atoms, 'info', {}) or {}
    for key in ['energy', 'Energy', 'REF_energy', 'total_energy', 'free_energy']:
        if key in info:
            try:
                return float(info[key])
            except (TypeError, ValueError):
                pass
    return None


def extract_forces(atoms):
    """Extract forces from atoms, checking arrays, calc.results, and calculator methods."""
    # 1. Check calculator results first
    if getattr(atoms, "calc", None) is not None and hasattr(atoms.calc, "results"):
        if isinstance(atoms.calc.results, dict) and 'forces' in atoms.calc.results:
            return np.array(atoms.calc.results['forces'])
    
    # 2. Try calculator method
    if getattr(atoms, "calc", None) is not None:
        try:
            return np.array(atoms.get_forces())
        except Exception:
            pass
    
    # 3. Fall back to arrays dict
    arrays = getattr(atoms, 'arrays', {}) or {}
    for key in ['forces', 'force', 'REF_forces', 'REF_force']:
        if key in arrays:
            return np.array(arrays[key])
    
    # 4. Check info dict (rare but possible)
    info = getattr(atoms, 'info', {}) or {}
    for key in ['forces', 'force', 'REF_forces']:
        if key in info:
            try:
                return np.array(info[key])
            except Exception:
                pass
    return None

# ======================================================================
# Data loading & caching
# ======================================================================

@st.cache_data(show_spinner=False)
def load_trajectory(dataset_name: str):
    """Load an extxyz trajectory file, return list of Atoms."""
    filepath = DATASET_FILES[dataset_name]
    if not Path(filepath).exists():
        return None
    frames = read(filepath, index=":", format="extxyz")
    return frames


def compute_statistics(dataset_name: str, frames):
    """
    Compute comprehensive statistics and cache to disk.
    Returns a dict with all stats.
    """
    cache_file = _cache_path(dataset_name, "stats.pkl")
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    all_elements = set()
    formula_counts = Counter()
    natoms_list = []
    dim_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    energies = []
    max_forces = []
    mean_forces = []
    element_counts = Counter()

    # Collect per-atom scalar & array info keys
    scalar_props = set()
    array_props = set()

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(frames)

    for idx, atoms in enumerate(frames):
        symbols = atoms.get_chemical_symbols()
        all_elements.update(symbols)
        element_counts.update(symbols)
        formula = atoms.get_chemical_formula(mode="hill")
        formula_counts[formula] += 1
        natoms_list.append(len(atoms))

        dim = classify_dimensionality(atoms)
        dim_counts[dim] += 1

        info = atoms.info if hasattr(atoms, 'info') else {}
        arrays = atoms.arrays if hasattr(atoms, 'arrays') else {}

        # Energy — check calc.results, calculator methods, then info dict
        energy = extract_energy(atoms)
        if energy is not None:
            energies.append(energy)

        # Forces — check calc.results, calculator methods, then arrays dict
        forces = extract_forces(atoms)

        if forces is not None and len(forces) > 0:
            force_norms = np.linalg.norm(forces, axis=1)
            max_forces.append(float(np.max(force_norms)))
            mean_forces.append(float(np.mean(force_norms)))

        # Collect property keys from info
        for k, v in info.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                scalar_props.add(k)
        # Collect from arrays
        for k, v in arrays.items():
            if k not in ('numbers', 'positions'):
                array_props.add(k)
        # Collect from calculator results
        if getattr(atoms, "calc", None) is not None and hasattr(atoms.calc, "results"):
            if isinstance(atoms.calc.results, dict):
                for k, v in atoms.calc.results.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        scalar_props.add(f"calc:{k}")
                    elif isinstance(v, np.ndarray):
                        array_props.add(f"calc:{k}")

        if (idx + 1) % max(1, total // 100) == 0 or idx == total - 1:
            progress_bar.progress((idx + 1) / total)
            status_text.text(f"Analyzing {dataset_name}: {idx + 1}/{total}")

    progress_bar.empty()
    status_text.empty()

    stats = {
        'total_structures': total,
        'elements': sorted(all_elements),
        'element_counts': dict(element_counts),
        'scalar_properties': sorted(scalar_props),
        'array_properties': sorted(array_props),
        'unique_formulas': len(formula_counts),
        'formula_counts': dict(formula_counts.most_common(200)),  # top 200
        'natoms_list': natoms_list,
        'natoms_range': (int(min(natoms_list)), int(max(natoms_list))) if natoms_list else (0, 0),
        'dim_counts': dim_counts,
        'energies': energies,
        'max_forces': max_forces,
        'mean_forces': mean_forces,
    }

    with open(cache_file, "wb") as f:
        pickle.dump(stats, f)

    return stats


def get_or_build_dim_subsets(dataset_name: str, frames):
    """
    Split frames by dimensionality and cache as extxyz on disk.
    Returns dict {dim_int: list_of_frame_indices}.
    """
    index_cache = _cache_path(dataset_name, "dim_indices.pkl")
    if index_cache.exists():
        with open(index_cache, "rb") as f:
            return pickle.load(f)

    dim_indices = {0: [], 1: [], 2: [], 3: []}
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(frames)

    for idx, atoms in enumerate(frames):
        dim = classify_dimensionality(atoms)
        dim_indices[dim].append(idx)
        if (idx + 1) % max(1, total // 100) == 0 or idx == total - 1:
            progress_bar.progress((idx + 1) / total)
            status_text.text(f"Building dimensionality index for {dataset_name}: {idx + 1}/{total}")

    progress_bar.empty()
    status_text.empty()

    # Also write out subset xyz files for convenience
    for dim, indices in dim_indices.items():
        if indices:
            subset_file = _cache_path(dataset_name, f"dim{dim}.xyz")
            if not subset_file.exists():
                subset_atoms = [frames[i] for i in indices]
                write(str(subset_file), subset_atoms, format="extxyz")

    with open(index_cache, "wb") as f:
        pickle.dump(dim_indices, f)

    return dim_indices


# ======================================================================
# Query engine
# ======================================================================

def query_frames(frames, dim_indices, query_params):
    """Query frames based on user parameters."""
    results = []

    # Determine which frame indices to search
    if query_params['dimensionality'] is not None:
        candidate_indices = dim_indices.get(query_params['dimensionality'], [])
    else:
        candidate_indices = list(range(len(frames)))

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(candidate_indices)

    for count, idx in enumerate(candidate_indices):
        atoms = frames[idx]
        symbols = set(atoms.get_chemical_symbols())
        formula = atoms.get_chemical_formula(mode="hill")
        natoms = len(atoms)

        # N atoms filter
        if query_params['natoms_min'] is not None and natoms < query_params['natoms_min']:
            continue
        if query_params['natoms_max'] is not None and natoms > query_params['natoms_max']:
            continue

        # Element inclusion
        if query_params['elements']:
            if query_params['exact_elements_only']:
                if symbols != set(query_params['elements']):
                    continue
            else:
                if not all(e in symbols for e in query_params['elements']):
                    continue

        # Element exclusion
        if query_params['exclude_elements']:
            if any(e in symbols for e in query_params['exclude_elements']):
                continue

        # Exact formula
        if query_params['exact_formula']:
            try:
                target = Atoms(query_params['exact_formula']).get_chemical_formula(mode="hill")
                if formula != target:
                    continue
            except Exception:
                continue

        # Property filters
        info = atoms.info if hasattr(atoms, 'info') else {}
        skip = False
        for prop_name, (min_val, max_val) in query_params.get('property_filters', {}).items():
            val = info.get(prop_name, None)
            if val is not None:
                try:
                    val = float(val)
                    if min_val is not None and val < min_val:
                        skip = True
                        break
                    if max_val is not None and val > max_val:
                        skip = True
                        break
                except (TypeError, ValueError):
                    pass
        if skip:
            continue

        # Collect result info
        arrays = atoms.arrays if hasattr(atoms, 'arrays') else {}
        props = {}
        for k, v in info.items():
            if isinstance(v, (int, float, np.integer, np.floating, str)):
                props[k] = v

        results.append({
            'frame_index': idx,
            'atoms': atoms,
            'formula': formula,
            'natoms': natoms,
            'elements': ', '.join(sorted(symbols)),
            'dimensionality': classify_dimensionality(atoms),
            'properties': props,
        })

        if len(results) >= query_params['max_results']:
            break

        if (count + 1) % max(1, total // 100) == 0 or count == total - 1:
            progress_bar.progress((count + 1) / total)
            status_text.text(f"Searching: {count + 1}/{total} candidates checked, {len(results)} found")

    progress_bar.empty()
    status_text.empty()

    return results


# ======================================================================
# Visualization
# ======================================================================

def visualize_structure(atoms, index, structure_id, html_file_name='viz.html'):
    """Visualize atomic structure using py3Dmol."""
    col1, col2 = st.columns([3, 1])

    with col2:
        spin = st.checkbox('Spin', value=True, key=f'spin_{structure_id}')
        style = st.selectbox(
            'Style',
            ['Ball & Stick', 'Stick', 'Sphere', 'Line'],
            key=f'style_{structure_id}'
        )

    with col1:
        view = py3Dmol.view(width=700, height=500)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
            temp_filename = f.name
            write(temp_filename, atoms, format='cif')

        with open(temp_filename, 'r') as cif_file:
            cif_content = cif_file.read()

        try:
            os.unlink(temp_filename)
        except PermissionError:
            pass

        view.addModel(cif_content, 'cif')

        if style == 'Ball & Stick':
            view.setStyle({
                'sphere': {'colorscheme': 'Jmol', 'scale': 0.3},
                'stick': {'colorscheme': 'Jmol', 'radius': 0.2}
            })
        elif style == 'Stick':
            view.setStyle({'stick': {'colorscheme': 'Jmol', 'radius': 0.2}})
        elif style == 'Sphere':
            view.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.5}})
        elif style == 'Line':
            view.setStyle({'line': {'colorscheme': 'Jmol'}})

        view.addUnitCell()
        view.zoomTo()
        view.spin(spin)
        view.setClickable({'clickable': 'true'})
        view.enableContextMenu({'contextMenuEnabled': 'true'})
        view.render()

        t = view.js()
        with open(html_file_name, 'w') as f:
            f.write(t.startjs)
            f.write(t.endjs)

        with open(html_file_name, 'r', encoding='utf-8') as f:
            source_code = f.read()
        components.html(source_code, height=600, width=900)


def export_to_extxyz(results, selected_indices):
    """Export selected structures to extended XYZ format."""
    output = io.StringIO()
    for idx in selected_indices:
        atoms = results[idx]['atoms']
        write(output, atoms, format='extxyz')
    return output.getvalue()


def format_search_duration(seconds):
    """Format an in-memory dataset-query duration for display."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


def render_dataset_citation():
    """Show the MAD paper citation and official Materials Cloud downloads."""
    with st.expander("Dataset citation and official downloads"):
        st.markdown(
            "**Please cite:** A. Mazitov, S. Chorna, G. Fraux, M. Bercx, "
            "G. Pizzi, S. De, and M. Ceriotti, “Massive Atomic Diversity: a "
            "compact universal dataset for atomistic machine learning,” "
            f"*Scientific Data* **12**, 1857 (2025). [Paper]({MAD_PAPER_URL})"
        )
        st.markdown(f"**Official dataset record:** [Materials Cloud]({MAD_ARCHIVE_URL})")
        st.markdown(
            "**Direct extXYZ downloads:** "
            + " · ".join(
                f"[{label}]({url})" for label, url in MAD_DOWNLOAD_URLS.items()
            )
        )


# ======================================================================
# Main App
# ======================================================================

def main():
    st.title("🔬 MADv1 (PBEsol) Explorer")
    st.caption("Extended XYZ Trajectory Dataset Explorer for `mad-train.xyz`, `mad-test.xyz`, `mad-val.xyz`")
    render_dataset_citation()
    st.markdown("---")

    # --- Sidebar ---
    with st.sidebar:
        st.header("📂 Dataset")

        # Check which files exist
        available = {k: v for k, v in DATASET_FILES.items() if Path(v).exists()}
        missing = {k: v for k, v in DATASET_FILES.items() if not Path(v).exists()}

        if missing:
            st.warning(f"Missing files: {', '.join(missing.values())}")

        if not available:
            st.error("No dataset files found! Place mad-train.xyz, mad-test.xyz, and/or mad-val.xyz in the working directory.")
            return

        dataset_name = st.selectbox("Select dataset", options=list(available.keys()),
                                     format_func=lambda x: f"{x} ({available[x]})")

        if st.button("Load Dataset", type="primary"):
            with st.spinner(f"Loading {available[dataset_name]}..."):
                frames = load_trajectory(dataset_name)
                if frames is None or len(frames) == 0:
                    st.error("Failed to load or empty dataset.")
                else:
                    st.session_state.frames = frames
                    st.session_state.current_dataset = dataset_name
                    st.session_state.dataset_loaded = True
                    st.session_state.query_results = []
                    st.session_state.selected_indices = []
                    st.session_state.query_elapsed_seconds = None

                    with st.spinner("Computing statistics (cached on disk)..."):
                        stats = compute_statistics(dataset_name, frames)
                        st.session_state.stats = stats
                        st.session_state.all_elements = stats['elements']

                    with st.spinner("Building dimensionality index (cached on disk)..."):
                        dim_indices = get_or_build_dim_subsets(dataset_name, frames)
                        st.session_state.dim_indices = dim_indices

                    st.success(f"Loaded {len(frames)} structures!")
                    st.rerun()
                                        # Show detected property sources for debugging
                    with st.sidebar.expander("🔧 Detected Properties"):
                        a0 = frames[0]
                        st.write("**info keys:**", sorted(a0.info.keys()) if hasattr(a0, 'info') else [])
                        st.write("**arrays keys:**", sorted([k for k in a0.arrays.keys() if k not in ('numbers', 'positions')]) if hasattr(a0, 'arrays') else [])
                        if getattr(a0, "calc", None) is not None and hasattr(a0.calc, "results"):
                            st.write("**calc.results keys:**", sorted(a0.calc.results.keys()))
                        else:
                            st.write("**calc.results:** None")
                        
                        # Quick sanity check
                        e0 = extract_energy(a0)
                        f0 = extract_forces(a0)
                        st.write(f"**First frame energy:** {e0}")
                        st.write(f"**First frame forces shape:** {f0.shape if f0 is not None else None}")

        if st.session_state.dataset_loaded:
            st.success("✅ Dataset loaded")
            st.info(f"**File:** {DATASET_FILES[st.session_state.current_dataset]}")
            s = st.session_state.stats
            st.metric("Total Structures", s['total_structures'])
            st.metric("Unique Elements", len(s['elements']))
            st.metric("Unique Formulas", s['unique_formulas'])
            dim = s['dim_counts']
            st.markdown("**Dimensionality:**")
            for d in range(4):
                st.text(f"  {DIMENSIONALITY_LABELS[d]}: {dim[d]}")

    # --- Main content ---
    if not st.session_state.dataset_loaded:
        st.info("👈 Load a dataset from the sidebar to begin.")
        st.markdown("""
        ### Features
        - **Advanced Querying** with element, formula, atom-count, dimensionality, and property filters
        - **Dataset Statistics** with histograms of energy, forces, atom counts, element frequency, dimensionality
        - **Interactive 3D Visualization** via py3Dmol
        - **Export** selected structures as extended XYZ
        - **Cached statistics & dimensionality subsets** for fast repeat usage
        """)
        return

    frames = st.session_state.frames
    stats = st.session_state.stats
    dim_indices = st.session_state.dim_indices

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query", "📊 Results", "📈 Statistics", "ℹ️ Info"])

    # ============================ TAB 1: QUERY ============================
    with tab1:
        st.header("Query Dataset")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Element Filters")

            select_all_elements = st.checkbox("Select ALL elements", value=False)

            if select_all_elements:
                include_elements = list(stats['elements'])
                st.info(f"All {len(include_elements)} elements selected.")
            else:
                include_elements = st.multiselect(
                    "Contains Elements",
                    options=stats['elements'],
                    help="Structures must contain ALL selected elements"
                )

            exact_elements_only = st.toggle(
                "Exact elements only",
                value=False,
                disabled=not include_elements,
                help="When enabled, structures must contain EXACTLY the selected elements and no others."
            )
            if not include_elements:
                exact_elements_only = False

            exclude_elements = st.multiselect(
                "Exclude Elements",
                options=stats['elements'],
                help="Structures must NOT contain ANY of these elements"
            )

            exact_formula = st.text_input(
                "Exact Chemical Formula",
                placeholder="e.g., MoS2, Al2O3",
                help="Search for exact stoichiometric formula"
            )

        with col2:
            st.subheader("Structure Filters")

            # Dimensionality filter
            dim_options = ["Any"] + [DIMENSIONALITY_LABELS[d] for d in range(4)]
            dim_selection = st.selectbox("Dimensionality", options=dim_options, index=0)
            if dim_selection == "Any":
                dim_filter = None
            else:
                dim_filter = [k for k, v in DIMENSIONALITY_LABELS.items() if v == dim_selection][0]

            natoms_range = st.slider(
                "Number of Atoms",
                min_value=stats['natoms_range'][0],
                max_value=stats['natoms_range'][1],
                value=(stats['natoms_range'][0], stats['natoms_range'][1]),
            )

            max_results = st.number_input(
                "Maximum Results",
                min_value=10, max_value=50000, value=100, step=10,
            )

        # Advanced property filters
        with st.expander("🔧 Advanced Property Filters"):
            property_filters = {}
            if stats['scalar_properties']:
                selected_prop = st.selectbox("Property", options=[''] + stats['scalar_properties'])
                if selected_prop:
                    ca, cb = st.columns(2)
                    with ca:
                        prop_min = st.number_input(f"Min {selected_prop}", value=None, format="%.6f")
                    with cb:
                        prop_max = st.number_input(f"Max {selected_prop}", value=None, format="%.6f")
                    property_filters = {selected_prop: (prop_min, prop_max)}
            else:
                st.info("No scalar properties detected in info dict.")

        # Run query
        if st.button("🔍 Search", type="primary", use_container_width=True):
            query_params = {
                'elements': include_elements,
                'exclude_elements': exclude_elements,
                'exact_formula': exact_formula if exact_formula else None,
                'exact_elements_only': exact_elements_only,
                'natoms_min': natoms_range[0],
                'natoms_max': natoms_range[1],
                'max_results': max_results,
                'property_filters': property_filters,
                'dimensionality': dim_filter,
            }
            with st.spinner("Searching..."):
                search_start = time.perf_counter()
                results = query_frames(frames, dim_indices, query_params)
                search_elapsed = time.perf_counter() - search_start
                st.session_state.query_results = results
                st.session_state.query_elapsed_seconds = search_elapsed
                st.session_state.selected_indices = []
                st.session_state.viz_pos = 0
            st.success(
                f"Found {len(results)} structures in "
                f"{format_search_duration(search_elapsed)}."
            )
            if results:
                st.balloons()

    # ============================ TAB 2: RESULTS ============================
    with tab2:
        st.header("Query Results")
        query_elapsed = st.session_state.query_elapsed_seconds
        if query_elapsed is not None:
            st.caption(
                "Structure search time: "
                f"{format_search_duration(query_elapsed)} "
                "(result rendering and export excluded)."
            )

        if not st.session_state.query_results:
            st.info("No results yet. Run a query in the Query tab.")
        else:
            results = st.session_state.query_results
            st.success(f"Displaying {len(results)} structures")

            df_data = []
            for i, r in enumerate(results):
                row = {
                    'Index': i,
                    'Frame': r['frame_index'],
                    'Formula': r['formula'],
                    'N Atoms': r['natoms'],
                    'Elements': r['elements'],
                    'Dim': DIMENSIONALITY_LABELS.get(r['dimensionality'], '?'),
                }
                for k in ['energy', 'Energy', 'REF_energy']:
                    if k in r['properties']:
                        row['Energy'] = r['properties'][k]
                        break
                df_data.append(row)

            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True, height=400)

            # Selection
            st.subheader("Select Structures")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Select All"):
                    st.session_state.selected_indices = list(range(len(results)))
                    st.rerun()
            with c2:
                if st.button("Clear Selection"):
                    st.session_state.selected_indices = []
                    st.session_state.viz_pos = 0
                    st.rerun()
            with c3:
                select_range = st.text_input("Select by index (e.g., 0-5, 10)", placeholder="0-5, 10")

            if select_range:
                selected = []
                for part in select_range.split(','):
                    part = part.strip()
                    if '-' in part:
                        a, b = map(int, part.split('-'))
                        selected.extend(range(a, b + 1))
                    else:
                        selected.append(int(part))
                st.session_state.selected_indices = [i for i in selected if 0 <= i < len(results)]

            selected_manual = st.multiselect(
                "Or select manually:",
                options=list(range(len(results))),
                default=st.session_state.selected_indices,
                format_func=lambda x: f"{x}: {results[x]['formula']} ({results[x]['natoms']} atoms)"
            )
            st.session_state.selected_indices = selected_manual

            # Export
            if st.session_state.selected_indices:
                st.success(f"{len(st.session_state.selected_indices)} structure(s) selected")
                if st.button("📥 Export Selected as Extended XYZ", type="primary"):
                    content = export_to_extxyz(results, st.session_state.selected_indices)
                    st.download_button(
                        label="Download .extxyz file",
                        data=content,
                        file_name="selected_structures.extxyz",
                        mime="chemical/x-xyz"
                    )

            # Visualization
            st.markdown("---")
            st.subheader("🔬 Structure Visualization")

            if st.session_state.selected_indices:
                # Initialize viz position in session state
                if 'viz_pos' not in st.session_state:
                    st.session_state.viz_pos = 0

                sel = st.session_state.selected_indices
                n_sel = len(sel)

                # Clamp position
                if st.session_state.viz_pos >= n_sel:
                    st.session_state.viz_pos = n_sel - 1
                if st.session_state.viz_pos < 0:
                    st.session_state.viz_pos = 0

                # Navigation row
                nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 2, 1, 1])
                with nav1:
                    if st.button("⏮ First", use_container_width=True):
                        st.session_state.viz_pos = 0
                        st.rerun()
                with nav2:
                    if st.button("◀ Prev", use_container_width=True, disabled=(st.session_state.viz_pos == 0)):
                        st.session_state.viz_pos -= 1
                        st.rerun()
                with nav3:
                    st.markdown(
                        f"<div style='text-align:center; padding:0.5rem; font-size:1.1rem;'>"
                        f"<b>{st.session_state.viz_pos + 1}</b> / {n_sel}</div>",
                        unsafe_allow_html=True
                    )
                with nav4:
                    if st.button("Next ▶", use_container_width=True, disabled=(st.session_state.viz_pos >= n_sel - 1)):
                        st.session_state.viz_pos += 1
                        st.rerun()
                with nav5:
                    if st.button("Last ⏭", use_container_width=True):
                        st.session_state.viz_pos = n_sel - 1
                        st.rerun()

                viz_index = sel[st.session_state.viz_pos]
                r = results[viz_index]

                st.markdown(
                    f"**Result #{viz_index}** — **Formula:** {r['formula']}  |  "
                    f"**N Atoms:** {r['natoms']}  |  **Elements:** {r['elements']}  |  "
                    f"**Dim:** {DIMENSIONALITY_LABELS.get(r['dimensionality'], '?')}"
                )

                with st.expander("View Properties"):
                    st.json(r['properties'])

                visualize_structure(r['atoms'], viz_index, f"struct_{viz_index}", "viz1.html")
            else:
                st.info("Select at least one structure to visualize.")

    # ============================ TAB 3: STATISTICS ============================
    with tab3:
        st.header("📈 Dataset Statistics")

        # Overview metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Structures", stats['total_structures'])
        c2.metric("Unique Elements", len(stats['elements']))
        c3.metric("Unique Formulas", stats['unique_formulas'])
        c4.metric("Atom Range", f"{stats['natoms_range'][0]} – {stats['natoms_range'][1]}")

        st.markdown("---")

        # Dimensionality breakdown
        st.subheader("Dimensionality Breakdown")
        dim_df = pd.DataFrame({
            'Dimensionality': [DIMENSIONALITY_LABELS[d] for d in range(4)],
            'Count': [stats['dim_counts'][d] for d in range(4)],
        })
        col_dim1, col_dim2 = st.columns([2, 1])
        with col_dim1:
            st.bar_chart(dim_df.set_index('Dimensionality'))
        with col_dim2:
            st.dataframe(dim_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Element frequency
        st.subheader("Element Frequency (Total Atom Occurrences)")

        from ase.data import atomic_numbers
        import plotly.graph_objects as go

        elem_sorted = sorted(
            stats['element_counts'].items(),
            key=lambda x: atomic_numbers.get(x[0], 999)
        )

        elem_names = [e[0] for e in elem_sorted]
        elem_counts = [e[1] for e in elem_sorted]

        fig = go.Figure()

        fig.add_bar(
            x=elem_names,
            y=elem_counts,
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
        )

        fig.update_layout(
            height=500,
            xaxis_title="Element (by atomic number)",
            yaxis_title="Count",
            font=dict(size=16),  # 🔹 Larger global font
            xaxis=dict(
                tickangle=90,
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                tickfont=dict(size=14)
            ),
            margin=dict(l=40, r=40, t=40, b=120)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Number of atoms distribution
        st.subheader("Number of Atoms Distribution")
        natoms_arr = np.array(stats['natoms_list'])
        fig_natoms_data = np.histogram(natoms_arr, bins=min(50, max(1, int(natoms_arr.max() - natoms_arr.min()))))
        natoms_hist_df = pd.DataFrame({
            'Bin': [f"{int(fig_natoms_data[1][i])}-{int(fig_natoms_data[1][i+1])}" for i in range(len(fig_natoms_data[0]))],
            'Count': fig_natoms_data[0]
        })
        st.bar_chart(natoms_hist_df.set_index('Bin'))

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean N Atoms", f"{natoms_arr.mean():.1f}")
        c2.metric("Median N Atoms", f"{np.median(natoms_arr):.1f}")
        c3.metric("Std N Atoms", f"{natoms_arr.std():.1f}")

        st.markdown("---")

        # Energy distribution
        if stats['energies']:
            st.subheader("Energy Distribution")
            energy_arr = np.array(stats['energies'])
            nbins = min(80, max(10, len(energy_arr) // 20))
            energy_hist = np.histogram(energy_arr, bins=nbins)
            energy_hist_df = pd.DataFrame({
                'Bin': [f"{energy_hist[1][i]:.2f}" for i in range(len(energy_hist[0]))],
                'Count': energy_hist[0]
            })
            st.bar_chart(energy_hist_df.set_index('Bin'))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Energy", f"{energy_arr.mean():.4f}")
            c2.metric("Median Energy", f"{np.median(energy_arr):.4f}")
            c3.metric("Min Energy", f"{energy_arr.min():.4f}")
            c4.metric("Max Energy", f"{energy_arr.max():.4f}")

            # Per-atom energy
            if len(stats['energies']) == len(stats['natoms_list']):
                per_atom_energy = energy_arr / np.array(stats['natoms_list'])
                st.subheader("Per-Atom Energy Distribution")
                pa_hist = np.histogram(per_atom_energy, bins=nbins)
                pa_df = pd.DataFrame({
                    'Bin': [f"{pa_hist[1][i]:.3f}" for i in range(len(pa_hist[0]))],
                    'Count': pa_hist[0]
                })
                st.bar_chart(pa_df.set_index('Bin'))
                c1, c2 = st.columns(2)
                c1.metric("Mean E/atom", f"{per_atom_energy.mean():.4f}")
                c2.metric("Std E/atom", f"{per_atom_energy.std():.4f}")
        else:
            st.info("No energy data found in dataset info dicts.")

        st.markdown("---")

        # Force distributions
        if stats['max_forces']:
            st.subheader("Force Distributions")

            force_col1, force_col2 = st.columns(2)

            with force_col1:
                st.markdown("**Max Force per Structure**")
                max_f_arr = np.array(stats['max_forces'])
                nbins_f = min(80, max(10, len(max_f_arr) // 20))
                mf_hist = np.histogram(max_f_arr, bins=nbins_f)
                mf_df = pd.DataFrame({
                    'Bin': [f"{mf_hist[1][i]:.2f}" for i in range(len(mf_hist[0]))],
                    'Count': mf_hist[0]
                })
                st.bar_chart(mf_df.set_index('Bin'))
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Max Force", f"{max_f_arr.mean():.4f}")
                c2.metric("Median Max Force", f"{np.median(max_f_arr):.4f}")
                c3.metric("Max Max Force", f"{max_f_arr.max():.4f}")

            with force_col2:
                st.markdown("**Mean Force per Structure**")
                mean_f_arr = np.array(stats['mean_forces'])
                meanf_hist = np.histogram(mean_f_arr, bins=nbins_f)
                meanf_df = pd.DataFrame({
                    'Bin': [f"{meanf_hist[1][i]:.2f}" for i in range(len(meanf_hist[0]))],
                    'Count': meanf_hist[0]
                })
                st.bar_chart(meanf_df.set_index('Bin'))
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Avg Force", f"{mean_f_arr.mean():.4f}")
                c2.metric("Median Avg Force", f"{np.median(mean_f_arr):.4f}")
                c3.metric("Max Avg Force", f"{mean_f_arr.max():.4f}")
        else:
            st.info("No force data found in dataset arrays.")

        st.markdown("---")

        # Top formulas
        st.subheader("Most Common Formulas")
        if stats['formula_counts']:
            top_formulas = list(stats['formula_counts'].items())[:30]
            form_df = pd.DataFrame(top_formulas, columns=['Formula', 'Count'])
            st.bar_chart(form_df.set_index('Formula'))
            with st.expander("Show all (up to 200)"):
                st.dataframe(
                    pd.DataFrame(list(stats['formula_counts'].items()), columns=['Formula', 'Count']),
                    hide_index=True, use_container_width=True, height=400
                )

        st.markdown("---")

        st.subheader("All Elements")
        from ase.data import atomic_numbers
        sorted_elements = sorted(stats['elements'], key=lambda x: atomic_numbers.get(x, 999))
        st.write(", ".join(sorted_elements))

        # Properties list
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Scalar Properties (info)")
            if stats['scalar_properties']:
                for p in stats['scalar_properties']:
                    st.text(f"• {p}")
            else:
                st.info("None detected")
        with c2:
            st.subheader("Array Properties (arrays)")
            if stats['array_properties']:
                for p in stats['array_properties']:
                    st.text(f"• {p}")
            else:
                st.info("None detected")

    # ============================ TAB 4: INFO ============================
    with tab4:
        st.header("About MAD Explorer")
        st.markdown("""
        ### MAD Explorer — Extended XYZ Trajectory Dataset Explorer

        Explore, query, visualize and export structures from MAD datasets stored as 
        extended XYZ trajectories: `mad-train.xyz`, `mad-test.xyz`, `mad-val.xyz`.

        #### Key Features

        1. **Smart Caching**
           - Statistics computed once and saved to `.mad_explorer_cache/`
           - Dimensionality subsets indexed and stored for fast filtered queries
           - Cache invalidated automatically when source files change

        2. **Advanced Querying**
           - Filter by chemical elements (include/exclude/exact match) with "Select All" option
           - Dimensionality filter: 0D (molecules), 1D (wires), 2D (slabs), 3D (bulk)
           - Atom count range, exact formula, custom property range filters

        3. **Comprehensive Statistics**
           - Dimensionality breakdown (bar chart + table)
           - Element frequency across entire dataset
           - Number of atoms distribution with mean/median/std
           - Energy distribution (total and per-atom)
           - Force distributions (max force and mean force per structure)
           - Most common formulas

        4. **Interactive Visualization**
           - 3D structure rendering via py3Dmol
           - Ball & Stick, Stick, Sphere, Line styles
           - Unit cell display, spin animation

        5. **Data Export**
           - Export selected structures as extended XYZ
           - All properties preserved

        #### Dimensionality Classification

        Structures are classified based on periodic boundary conditions (pbc).  
        If pbc is not set, a vacuum-gap heuristic (12 Å threshold) is used:
        - **0D** — Molecule / cluster (no periodicity)
        - **1D** — Wire / chain (periodic in 1 direction)
        - **2D** — Slab / surface (periodic in 2 directions)
        - **3D** — Bulk crystal (periodic in all 3 directions)

        #### Cache Files

        All cached data is stored in `.mad_explorer_cache/` and keyed by a hash of
        the source file's path, size, and modification time. Delete this directory
        to force recomputation.

        #### Requirements

        ```
        streamlit
        py3Dmol
        ase
        pandas
        numpy
        ```
        """)


if __name__ == "__main__":
    main()
