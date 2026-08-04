"""Versioned elemental reference-energy data bundled with MLIP Studio."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .exceptions import TaskValidationError


REFERENCE_ENERGIES_PATH = Path(__file__).resolve().parent / "reference_energies.yaml"


@lru_cache(maxsize=1)
def load_element_reference_sets() -> dict[str, tuple[float, ...]]:
    """Load all elemental reference tables without doing work at API import."""

    try:
        import yaml
    except ImportError as exc:
        raise TaskValidationError(
            "Loading bundled reference energies requires PyYAML."
        ) from exc
    with REFERENCE_ENERGIES_PATH.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TaskValidationError("Bundled reference-energy data is invalid.")
    return {
        str(name): tuple(float(value) for value in values)
        for name, values in raw.items()
    }


def get_element_reference_set(name: str) -> tuple[float, ...]:
    """Return one reference table indexed by atomic number."""

    reference_sets = load_element_reference_sets()
    try:
        return reference_sets[name]
    except KeyError as exc:
        available = ", ".join(sorted(reference_sets))
        raise TaskValidationError(
            f"Unknown elemental reference set '{name}'. Available: {available}."
        ) from exc
