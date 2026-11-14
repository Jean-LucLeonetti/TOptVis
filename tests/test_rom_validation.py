"""Galerkin physics, design derivatives, held-out coverage and provenance."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.problem import CantileverProblem
from ml_opt.rom.training import SnapshotConfig, build_basis, generate_snapshots, read_metadata
from ml_opt.rom.validation import (GalerkinSolver, ValidationConfig, compare_evaluations,
                                   validate_basis, validation_cases, _relative_error)
from ml_opt.runner import optimise


def training(directory):
    config = SnapshotConfig(nx=6, ny=4, load_heights=(.5,), load_angles=(-90,),
                            volume_fractions=(.4,), iterations_per_case=3, snapshot_stride=1)
    generate_snapshots(config, directory)
    return build_basis(directory)


def options(**overrides):
    return ValidationConfig(**{**dict(load_heights=(.25, .75), load_angles=(-60,),
        volume_fractions=(.35,), iterations_per_case=3, comparison_stride=2), **overrides})


def test_complete_basis_matches_full_displacement_compliance_and_design_gradient():
    config = CantileverConfig(nx=4, ny=2, load_angle=-65, filter_radius=.12, projection_beta=3)
    with CantileverProblem(config) as full:
        basis = np.eye(len(full.mesh.free_dofs))
        with CantileverProblem(config, state_solver=GalerkinSolver(config, basis)) as reduced:
            design = np.random.default_rng(19).uniform(.3, .7, full.mesh.element_count)
            expected, actual = full.evaluate(design), reduced.evaluate(design)
            np.testing.assert_allclose(actual.displacement, expected.displacement, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(actual.compliance, expected.compliance, rtol=1e-10)
            np.testing.assert_allclose(actual.objective_gradient, expected.objective_gradient, rtol=1e-9, atol=1e-8)
            comparison = compare_evaluations(expected, actual, options())
            assert comparison["passed"]
            assert max(comparison[key] for key in ("displacement_error", "compliance_error", "gradient_error")) < 1e-9


@pytest.mark.parametrize("radius,beta", [(0, 0), (.13, 4)])
def test_reduced_design_gradient_matches_finite_differences(radius, beta):
    config = CantileverConfig(nx=6, ny=3, filter_radius=radius, projection_beta=beta, projection_eta=.4)
    with CantileverProblem(config) as full:
        first = full.evaluate(full.initial_design())
        column = first.displacement[full.mesh.free_dofs]
        basis = (column / np.linalg.norm(column))[:, None]
        with CantileverProblem(config, state_solver=GalerkinSolver(config, basis)) as reduced:
            design = np.random.default_rng(11).uniform(.3, .7, full.mesh.element_count)
            result = reduced.evaluate(design)
            assert result.compliance <= full.evaluate(design, False).compliance * (1+1e-10)
            for index in (0, 5, 11, 17):
                direction = np.zeros_like(design); direction[index] = 1e-6
                finite_difference = (reduced.evaluate(design+direction, False).compliance -
                                     reduced.evaluate(design-direction, False).compliance)/2e-6
                np.testing.assert_allclose(result.objective_gradient[index], finite_difference, rtol=3e-5, atol=1e-6)


def test_unseen_cases_reject_training_overlap_snapped_nodes_and_periodic_angles(tmp_path):
    metadata = training(tmp_path / "trained")
    with pytest.raises(ValueError, match="unseen"):
        validation_cases(metadata, options(load_heights=(.49,), load_angles=(-90,), volume_fractions=(.4,)))
    with pytest.raises(ValueError, match="distinct mesh rows"):
        validation_cases(metadata, options(load_heights=(.4, .5)))
    with pytest.raises(ValueError, match="distinct"):
        validation_cases(metadata, options(load_angles=(-180, 180)))
    alternate = {**metadata, "cases": [{**metadata["cases"][0], "load_angle": 180}]}
    with pytest.raises(ValueError, match="unseen"):
        validation_cases(alternate, options(load_heights=(.5,), load_angles=(-180,), volume_fractions=(.4,)))
    assert len(validation_cases(metadata, options())) == 2


def test_validation_metrics_match_saved_vectors_and_leave_basis_and_training_untouched(tmp_path):
    directory = tmp_path / "trained"
    training(directory)
    basis_bytes = (directory / "basis.npz").read_bytes()
    snapshots_bytes = (directory / "snapshots.npz").read_bytes()
    events = []
    metadata = validate_basis(directory, options(), emit=events.append)
    summary = metadata["validation"]
    assert summary["state_count"] == 6 and summary["completed_cases"] == 2
    assert summary["status"] == "complete"
    assert summary["basis_sha256"] == hashlib.sha256(basis_bytes).hexdigest()
    destination = directory / "validations" / summary["id"]
    report = json.loads((destination / "report.json").read_text())
    with np.load(destination / "comparisons.npz", allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata_json"])) == report
        np.testing.assert_array_equal(archive["iterations"], [0, 2, 3, 0, 2, 3])
        for index, row in enumerate(report["results"]):
            for name, array in (("displacement", "displacements"), ("gradient", "gradients")):
                reference = archive[f"full_{array}"][:, index]
                actual = archive[f"rom_{array}"][:, index]
                np.testing.assert_allclose(row[f"{name}_error"], np.linalg.norm(actual-reference)/np.linalg.norm(reference))
            np.testing.assert_allclose(row["compliance_error"], abs(row["rom_compliance"]-row["full_compliance"])/abs(row["full_compliance"]))
        for metric in ("displacement", "compliance", "gradient"):
            values = [row[f"{metric}_error"] for row in report["results"]]
            np.testing.assert_allclose(summary["metrics"][metric]["max"], max(values))
        assert summary["meets_tolerances"] == all(row["passed"] for row in report["results"])
    assert len([event for event in events if event["event"] == "validation_state"]) == 6
    assert (directory / "basis.npz").read_bytes() == basis_bytes
    assert (directory / "snapshots.npz").read_bytes() == snapshots_bytes
    assert read_metadata(directory) == metadata


def test_cancelled_validation_is_partial_and_rebuild_invalidates_report(tmp_path):
    directory = tmp_path / "partial"
    training(directory)
    events = []
    metadata = validate_basis(directory, options(), emit=events.append,
        stop_requested=lambda: any(event["event"] == "validation_state" for event in events))
    summary = metadata["validation"]
    assert summary["status"] == "cancelled" and summary["state_count"] == 1
    assert not summary["meets_tolerances"] and summary["completed_cases"] == 0
    assert validate_basis(directory, options(), stop_requested=lambda: True) is None
    assert read_metadata(directory) == metadata
    rebuilt = build_basis(directory, max_modes=1)
    assert "validation" not in rebuilt
    with np.load(directory / "basis.npz", allow_pickle=False) as model:
        assert "validation" not in json.loads(str(model["metadata_json"]))


def test_evaluation_callback_cannot_mutate_optimizer_state():
    config = CantileverConfig(nx=6, ny=2, max_iterations=3)
    def corrupt(row, design, evaluation):
        row.iteration = -99; design[:] = 0
        evaluation.objective_gradient[:] = 100
        evaluation.displacement[:] = -1; evaluation.density[:] = -1
    expected = optimise(config)
    result = optimise(config, evaluation_callback=corrupt)
    np.testing.assert_array_equal(expected.design, result.design)
    assert [row.iteration for row in result.history] == [0, 1, 2, 3]


@pytest.mark.parametrize("overrides", [{"gradient_tolerance": 0}, {"compliance_tolerance": float("nan")},
    {"comparison_stride": 0}, {"iterations_per_case": 300, "comparison_stride": 1},
    {"load_angles": (0, 0)}, {"load_heights": ()}])
def test_invalid_validation_options(overrides):
    with pytest.raises(ValueError):
        options(**overrides)


def test_zero_reference_error_is_explicit():
    assert _relative_error(np.zeros(2), np.zeros(2)) == 0
    assert _relative_error(np.ones(2), np.zeros(2)) is None


def test_invalid_basis_is_rejected():
    config = CantileverConfig(nx=4, ny=2)
    for basis in (np.zeros((24, 1)), np.ones((1, 1)), np.full((24, 1), np.nan)):
        with pytest.raises(ValueError):
            GalerkinSolver(config, basis)
