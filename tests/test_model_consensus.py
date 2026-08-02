import io

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

import model_consensus as consensus
from model_consensus import (
    ConsensusCalculationError,
    ConsensusModelSpec,
    ConsensusValidationError,
    ModelPrediction,
    PerturbationPrediction,
    SOURCE_FRAME_COUNT_KEY,
    generate_perturbed_configurations,
    get_consensus_datasets,
    get_consensus_models,
    make_consensus_fingerprint,
    make_consensus_extxyz,
    prepare_consensus_result,
    prepare_perturbation_result,
    run_model_consensus,
    validate_single_structure,
)


def model_spec(
    identifier,
    *,
    dataset="OMat24",
    family="MACE",
    label=None,
    metadata=None,
):
    return ConsensusModelSpec(
        id=identifier,
        label=label or identifier,
        family=family,
        dataset=dataset,
        model_name=identifier,
        force_mode="Conservative",
        metadata=metadata or {},
    )


def prediction(spec, energy, forces, *, error=None):
    return ModelPrediction(
        spec=spec,
        energy=energy,
        forces=None if forces is None else np.asarray(forces, dtype=float),
        stress=None,
        runtime_seconds=0.01,
        error=error,
    )


def h2():
    return Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    )


def two_valid_predictions(atoms):
    zeros = np.zeros((len(atoms), 3))
    return [
        prediction(model_spec("model-a"), -2.0, zeros),
        prediction(model_spec("model-b"), -1.0, zeros),
    ]


def test_identical_predictions_have_zero_energy_and_force_disagreement():
    atoms = h2()
    forces = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
    result = prepare_consensus_result(
        atoms,
        [
            prediction(model_spec("model-a"), -5.0, forces),
            prediction(model_spec("model-b"), -5.0, forces),
        ],
    )

    assert result.metrics["energy_std_per_atom"] == pytest.approx(0.0)
    assert result.metrics["energy_range_per_atom"] == pytest.approx(0.0)
    assert np.allclose(result.force_disagreement, 0.0)
    assert result.metrics["force_disagreement_max"] == pytest.approx(0.0)
    assert result.metrics["force_disagreement_rms"] == pytest.approx(0.0)


def test_known_two_model_force_disagreement_example():
    atoms = h2()
    force_a = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    force_b = np.array([[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    result = prepare_consensus_result(
        atoms,
        [
            prediction(model_spec("model-a"), 0.0, force_a),
            prediction(model_spec("model-b"), 0.0, force_b),
        ],
    )

    assert np.allclose(result.mean_forces, [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(result.force_disagreement, [1.0, 1.0])
    assert result.metrics["force_disagreement_mean"] == pytest.approx(1.0)
    assert result.metrics["force_disagreement_rms"] == pytest.approx(1.0)
    assert result.metrics["force_disagreement_p95"] == pytest.approx(1.0)


def test_energy_per_atom_is_only_total_system_normalization():
    atoms = h2()
    zeros = np.zeros((2, 3))
    result = prepare_consensus_result(
        atoms,
        [
            prediction(model_spec("model-a"), 4.0, zeros),
            prediction(model_spec("model-b"), 6.0, zeros),
        ],
    )

    assert result.model_table["Energy per Atom (eV/atom)"].tolist() == [2.0, 3.0]
    assert result.metrics["energy_std_per_atom"] == pytest.approx(0.5)
    assert result.metrics["energy_range_per_atom"] == pytest.approx(1.0)
    assert not any("Energy" in column for column in result.atom_table.columns)
    assert not hasattr(result, "atomwise_energies")


def test_pairwise_force_rmse_is_symmetric_with_zero_diagonal():
    atoms = h2()
    force_a = np.zeros((2, 3))
    force_b = np.ones((2, 3))
    force_c = np.arange(6, dtype=float).reshape(2, 3)
    result = prepare_consensus_result(
        atoms,
        [
            prediction(model_spec("model-a"), 0.0, force_a),
            prediction(model_spec("model-b"), 0.0, force_b),
            prediction(model_spec("model-c"), 0.0, force_c),
        ],
    )

    matrix = result.pairwise_force_rmse.to_numpy()
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] == pytest.approx(1.0)


def test_fewer_than_two_successful_models_raises_clear_error():
    atoms = h2()
    predictions = [
        prediction(model_spec("model-a"), -1.0, np.zeros((2, 3))),
        prediction(
            model_spec("model-b"),
            None,
            None,
            error="checkpoint could not be loaded",
        ),
    ]

    with pytest.raises(
        ConsensusCalculationError,
        match="at least two successful model predictions",
    ) as exc_info:
        prepare_consensus_result(atoms, predictions)

    assert exc_info.value.model_table is not None
    assert exc_info.value.model_table["Status"].tolist() == ["Success", "Failed"]


def test_failed_model_stays_in_status_table_but_is_excluded_from_metrics():
    atoms = h2()
    predictions = two_valid_predictions(atoms)
    predictions.insert(
        1,
        prediction(model_spec("model-failed"), None, None, error="load failed"),
    )
    result = prepare_consensus_result(atoms, predictions)

    assert len(result.predictions) == 3
    assert len(result.successful_predictions) == 2
    failed_row = result.model_table.loc[
        result.model_table["Model ID"] == "model-failed"
    ].iloc[0]
    assert failed_row["Status"] == "Failed"
    assert failed_row["Error"] == "load failed"
    assert result.pairwise_force_rmse.shape == (2, 2)


@pytest.mark.parametrize(
    ("bad_energy", "bad_forces", "error_text"),
    [
        (np.nan, np.zeros((2, 3)), "energy is non-finite"),
        (0.0, np.full((2, 3), np.inf), "forces contain non-finite"),
    ],
)
def test_non_finite_predictions_are_rejected(
    bad_energy,
    bad_forces,
    error_text,
):
    atoms = h2()
    predictions = two_valid_predictions(atoms)
    predictions.append(
        prediction(model_spec("bad-model"), bad_energy, bad_forces)
    )
    result = prepare_consensus_result(atoms, predictions)

    bad_row = result.model_table.loc[
        result.model_table["Model ID"] == "bad-model"
    ].iloc[0]
    assert bad_row["Status"] == "Failed"
    assert error_text in bad_row["Error"]
    assert len(result.successful_predictions) == 2


def test_incorrect_force_shape_is_rejected():
    atoms = h2()
    predictions = two_valid_predictions(atoms)
    predictions.append(
        prediction(model_spec("bad-shape"), 0.0, np.zeros((3, 3)))
    )
    result = prepare_consensus_result(atoms, predictions)

    bad_row = result.model_table.loc[
        result.model_table["Model ID"] == "bad-shape"
    ].iloc[0]
    assert bad_row["Status"] == "Failed"
    assert "expected (2, 3)" in bad_row["Error"]


def test_dataset_filtering_returns_only_explicitly_registered_models():
    assert get_consensus_datasets() == ["OMat24", "OMol25"]
    for dataset in get_consensus_datasets():
        models = get_consensus_models(dataset)
        assert len(models) >= 2
        assert all(model.dataset == dataset for model in models)
        assert len({model.id for model in models}) == len(models)

    assert get_consensus_models("not-registered") == []


def test_consensus_extxyz_has_shaped_force_arrays_and_safe_metadata():
    atoms = h2()
    result = prepare_consensus_result(atoms, two_valid_predictions(atoms))
    payload = make_consensus_extxyz(atoms, result)
    restored = read(io.StringIO(payload.decode("utf-8")), format="extxyz")

    assert restored.arrays["consensus_mean_forces"].shape == (2, 3)
    assert restored.arrays["force_disagreement"].shape == (2,)
    assert restored.info["consensus_dataset"] == "OMat24"
    assert isinstance(restored.info["consensus_models"], str)
    assert isinstance(restored.info["consensus_successful_models"], str)
    for key in (
        "energy_std_per_atom",
        "energy_range_per_atom",
        "force_disagreement_max",
        "force_disagreement_mean",
        "force_disagreement_rms",
        "force_disagreement_p95",
    ):
        assert np.isfinite(float(restored.info[key]))
    assert "atomwise_energy" not in restored.arrays
    assert "energies" not in restored.arrays


@pytest.mark.parametrize(
    "invalid",
    [
        lambda atoms: [atoms],
        lambda atoms: (atoms, atoms.copy()),
    ],
)
def test_batch_and_trajectory_inputs_are_rejected_defensively(invalid):
    with pytest.raises(ConsensusValidationError, match="exactly one structure"):
        validate_single_structure(invalid(h2()))


def test_multiframe_source_marker_is_rejected_defensively():
    atoms = h2()
    atoms.info[SOURCE_FRAME_COUNT_KEY] = 3

    with pytest.raises(ConsensusValidationError, match="contains 3 frames"):
        validate_single_structure(atoms)


def test_downloadable_app_does_not_impose_hosted_atom_limit():
    larger_structure = Atoms(
        numbers=[1] * 51,
        positions=np.zeros((51, 3)),
    )

    assert validate_single_structure(larger_structure) is larger_structure


class FakeCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy, forces, seen_metadata, fail=False):
        super().__init__()
        self.energy = energy
        self.forces = np.asarray(forces, dtype=float)
        self.seen_metadata = seen_metadata
        self.fail = fail

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.seen_metadata.append(dict(atoms.info))
        if self.fail:
            raise RuntimeError("synthetic model failure")
        self.results["energy"] = float(self.energy)
        self.results["forces"] = self.forces.copy()


def test_run_loop_copies_atoms_applies_omol_metadata_and_continues_failure(
    monkeypatch,
):
    atoms = h2()
    seen_metadata = []
    specs = [
        model_spec("model-a", dataset="OMol25"),
        model_spec("model-failed", dataset="OMol25"),
        model_spec("model-b", dataset="OMol25"),
    ]

    def fake_builder(spec, device):
        return FakeCalculator(
            energy={"model-a": -2.0, "model-b": -1.0}.get(spec.id, 0.0),
            forces=np.zeros((2, 3)),
            seen_metadata=seen_metadata,
            fail=spec.id == "model-failed",
        )

    monkeypatch.setattr(consensus, "build_consensus_calculator", fake_builder)
    result = run_model_consensus(
        atoms,
        specs,
        "cpu",
        charge=-1,
        spin=2,
    )

    assert result.model_table["Status"].tolist() == [
        "Success",
        "Failed",
        "Success",
    ]
    assert len(seen_metadata) == 3
    assert all(metadata["charge"] == -1 for metadata in seen_metadata)
    assert all(metadata["total_charge"] == -1 for metadata in seen_metadata)
    assert all(metadata["spin"] == 2 for metadata in seen_metadata)
    assert all(metadata["total_spin"] == 2 for metadata in seen_metadata)
    assert "total_charge" not in atoms.info
    assert "total_spin" not in atoms.info
    assert atoms.calc is None


def test_build_consensus_calculator_dispatches_without_loading_checkpoint(
    monkeypatch,
):
    sentinel = object()
    calls = []
    spec = model_spec("mace-test", family="MACE")

    def fake_mace_builder(received_spec, device):
        calls.append((received_spec, device))
        return sentinel

    monkeypatch.setattr(consensus, "_build_mace_calculator", fake_mace_builder)
    assert consensus.build_consensus_calculator(spec, "cpu") is sentinel
    assert calls == [(spec, "cpu")]


def test_unknown_calculator_family_has_readable_error():
    spec = model_spec("unknown", family="NotARealFamily")
    with pytest.raises(consensus.ConsensusModelError, match="Unsupported"):
        consensus.build_consensus_calculator(spec, "cpu")


def test_perturbation_path_is_reproducible_and_has_requested_rms_displacements():
    atoms = h2()
    original_positions = atoms.positions.copy()
    coordinates, configurations = generate_perturbed_configurations(
        atoms,
        configuration_count=7,
        max_rms_displacement=0.12,
        random_seed=19,
    )
    repeated_coordinates, repeated_configurations = (
        generate_perturbed_configurations(
            atoms,
            configuration_count=7,
            max_rms_displacement=0.12,
            random_seed=19,
        )
    )

    assert len(configurations) == 8
    assert np.allclose(coordinates, np.linspace(0.0, 0.12, 8))
    assert np.allclose(coordinates, repeated_coordinates)
    assert np.allclose(configurations[0].positions, original_positions)
    assert np.allclose(atoms.positions, original_positions)
    for coordinate, configuration, repeated in zip(
        coordinates,
        configurations,
        repeated_configurations,
    ):
        displacement = configuration.positions - original_positions
        displacement_rms = np.sqrt(
            np.mean(np.sum(displacement**2, axis=1))
        )
        assert displacement_rms == pytest.approx(coordinate)
        assert np.allclose(configuration.positions, repeated.positions)
        assert configuration.calc is None
        assert np.allclose(displacement.mean(axis=0), 0.0)


def test_perturbation_result_compares_relative_energies_and_forces():
    atoms = h2()
    coordinates = np.linspace(0.0, 0.08, 6)
    specs = [model_spec("model-a"), model_spec("model-b")]
    predictions = []
    for configuration_index, coordinate in enumerate(coordinates):
        predictions.extend(
            [
                PerturbationPrediction(
                    spec=specs[0],
                    configuration_index=configuration_index,
                    path_coordinate=float(coordinate),
                    energy=float(configuration_index),
                    forces=np.zeros((2, 3)),
                    runtime_seconds=0.01,
                    error=None,
                ),
                PerturbationPrediction(
                    spec=specs[1],
                    configuration_index=configuration_index,
                    path_coordinate=float(coordinate),
                    energy=float(2 * configuration_index),
                    forces=np.ones((2, 3)),
                    runtime_seconds=0.01,
                    error=None,
                ),
            ]
        )

    result = prepare_perturbation_result(
        atoms,
        predictions,
        coordinates,
        max_rms_displacement=0.08,
        random_seed=5,
    )

    final_rows = result.model_configuration_table[
        result.model_configuration_table["Configuration Index"] == 5
    ].set_index("Model ID")
    assert final_rows.loc["model-a", "Relative Energy (eV)"] == pytest.approx(5.0)
    assert final_rows.loc[
        "model-b", "Relative Energy / Atom (eV/atom)"
    ] == pytest.approx(5.0)
    assert final_rows.loc[
        "model-a", "Force RMSE vs Consensus (eV/Å)"
    ] == pytest.approx(0.5)
    assert final_rows.loc[
        "model-b", "Force RMSE vs Consensus (eV/Å)"
    ] == pytest.approx(0.5)
    assert result.configuration_summary_table.loc[
        5, "RMS Force Disagreement (eV/Å)"
    ] == pytest.approx(np.sqrt(0.75))


class PositionCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, offset, seen_positions):
        super().__init__()
        self.offset = float(offset)
        self.seen_positions = seen_positions

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=float)
        self.seen_positions.append(positions.copy())
        self.results["energy"] = float(np.sum(positions**2) + self.offset)
        self.results["forces"] = -2.0 * positions


def test_perturbation_run_loads_each_model_once_and_reuses_it_across_path(
    monkeypatch,
):
    atoms = h2()
    original_positions = atoms.positions.copy()
    specs = [model_spec("model-a"), model_spec("model-b")]
    build_calls = []
    seen_positions = []

    def fake_builder(spec, device):
        build_calls.append((spec.id, device))
        return PositionCalculator(
            offset=0.0 if spec.id == "model-a" else 1.0,
            seen_positions=seen_positions,
        )

    monkeypatch.setattr(consensus, "build_consensus_calculator", fake_builder)
    result = run_model_consensus(
        atoms,
        specs,
        "cpu",
        perturbation_count=5,
        perturbation_amplitude=0.04,
        perturbation_seed=11,
    )

    assert build_calls == [("model-a", "cpu"), ("model-b", "cpu")]
    assert len(seen_positions) == 12
    assert result.perturbation is not None
    assert result.perturbation.configuration_count == 6
    assert len(result.perturbation.model_configuration_table) == 12
    assert np.allclose(atoms.positions, original_positions)
    assert atoms.calc is None


def test_fingerprint_includes_perturbation_settings():
    atoms = h2()
    base = make_consensus_fingerprint(
        atoms,
        "OMat24",
        ["model-a", "model-b"],
        None,
        None,
        "cpu",
    )
    sampled = make_consensus_fingerprint(
        atoms,
        "OMat24",
        ["model-a", "model-b"],
        None,
        None,
        "cpu",
        run_perturbation_scan=True,
        perturbation_count=7,
        perturbation_amplitude=0.05,
        perturbation_seed=42,
    )
    changed_amplitude = make_consensus_fingerprint(
        atoms,
        "OMat24",
        ["model-a", "model-b"],
        None,
        None,
        "cpu",
        run_perturbation_scan=True,
        perturbation_count=7,
        perturbation_amplitude=0.08,
        perturbation_seed=42,
    )

    assert base != sampled
    assert sampled != changed_amplitude
