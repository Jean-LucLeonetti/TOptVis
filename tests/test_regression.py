"""Regression accuracy, whole-case isolation, result quality and persistence."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from ml_opt.regression import (RegressionSettings, generate_dataset, fit_models, predict,
                               read_dataset, read_models, split_cases)
from ml_opt.config import CantileverConfig
from ml_opt.runner import optimise


def synthetic(directory, count=12):
    directory.mkdir(exist_ok=True)
    settings = RegressionSettings(case_count=count)
    data = {"format": "toptvis-regression-v1", "id": directory.name, "status": "complete",
            "created_at": "2026-11-16T12:00:00+00:00", "settings": vars(settings),
            "generation_seconds": 2.0, "cases": [
                {"index": i, "volume_fraction": float(v), "compliance": float(10-5*v+12*v*v),
                 "converged": True, "feasible": True, "stop_reason": "design_and_objective_stable"}
                for i, v in enumerate(np.linspace(.25, .65, count))]}
    (directory / "dataset.json").write_text(json.dumps(data))
    return data


def test_quadratic_recovery_shared_split_baseline_and_reload(tmp_path):
    data = synthetic(tmp_path)
    result = fit_models(tmp_path)
    np.testing.assert_allclose(result["models"]["quadratic"]["coefficients"], [10, -5, 12], atol=1e-10)
    assert result["models"]["quadratic"]["test"]["rmse"] < 1e-12
    assert result["models"]["linear"]["test"]["rmse"] > 1e-3
    assert set(result["train_indices"]).isdisjoint(result["test_indices"])
    assert set(result["train_indices"] + result["test_indices"]) == set(range(12))
    assert 0 in result["train_indices"] and 11 in result["train_indices"]
    mean = np.mean([data["cases"][i]["compliance"] for i in result["train_indices"]])
    assert result["baseline"]["value"] == mean
    restored = read_models(tmp_path)
    assert restored == result
    np.testing.assert_allclose(predict(restored["models"]["quadratic"], [.3, .5]), [9.58, 10.5])
    assert fit_models(tmp_path)["test_indices"] == result["test_indices"]


def test_test_targets_cannot_change_fitted_coefficients(tmp_path):
    data = synthetic(tmp_path)
    before = fit_models(tmp_path)
    for index in before["test_indices"]:
        data["cases"][index]["compliance"] *= 10
    (tmp_path / "dataset.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="do not match"):
        read_models(tmp_path)
    after = fit_models(tmp_path)
    for name in ("linear", "quadratic"):
        np.testing.assert_array_equal(before["models"][name]["coefficients"], after["models"][name]["coefficients"])
        assert after["models"][name]["test"]["rmse"] > before["models"][name]["test"]["rmse"]
    assert before["baseline"]["value"] == after["baseline"]["value"]


def test_higher_orders_stable_predictions_shared_split_and_legacy_reload(tmp_path):
    data = synthetic(tmp_path, count=32)
    coefficients = [20, -8, 3, 2, -1, .5]
    for case in data["cases"]:
        case["compliance"] = float(np.polynomial.polynomial.polyval(case["volume_fraction"], coefficients))
    (tmp_path / "dataset.json").write_text(json.dumps(data))
    report = fit_models(tmp_path, degrees=[1, 2, 3, 5, 10])
    assert report["degrees"] == [1, 2, 3, 5, 10]
    assert set(report["models"]) == {"linear", "quadratic", "polynomial_3", "polynomial_5", "polynomial_10"}
    np.testing.assert_allclose(report["models"]["polynomial_5"]["coefficients"], coefficients, atol=1e-8)
    for order in (5, 10):
        fitted = read_models(tmp_path)["models"][f"polynomial_{order}"]
        assert fitted["test"]["rmse"] < 1e-10
        volumes = np.array([.2, .3, .5, .7])
        np.testing.assert_allclose(predict(fitted, volumes),
                                   np.polynomial.polynomial.polyval(volumes, coefficients), atol=1e-8)
    assert fit_models(tmp_path)["test_indices"] == report["test_indices"]
    np.testing.assert_allclose(predict({"coefficients": coefficients}, [.3, .5]),
                               np.polynomial.polynomial.polyval([.3, .5], coefficients))


@pytest.mark.parametrize("degrees", [[], [0], [True], [2.5], ["3"], [1, 1], None])
def test_invalid_polynomial_orders_preserve_existing_models(tmp_path, degrees):
    synthetic(tmp_path)
    before = fit_models(tmp_path)
    with pytest.raises(ValueError, match="Polynomial orders"):
        fit_models(tmp_path, degrees=degrees)
    assert read_models(tmp_path) == before


def test_order_requires_enough_training_cases_without_changing_split(tmp_path):
    synthetic(tmp_path, count=6)
    before = fit_models(tmp_path)
    assert len(before["train_indices"]) == 4
    with pytest.raises(ValueError, match="Order 4 needs at least 5 training cases"):
        fit_models(tmp_path, degrees=[1, 2, 3, 4])
    assert read_models(tmp_path) == before
    assert fit_models(tmp_path, degrees=[3])["test_indices"] == before["test_indices"]


def test_quality_filters_require_explicit_unfinished_opt_in(tmp_path):
    data = synthetic(tmp_path)
    data["cases"][0].update(converged=False, stop_reason="maximum_iterations")
    data["cases"][1].update(feasible=False)
    data["cases"][2].update(converged=False, stop_reason="cancelled")
    data["cases"][3].update(converged=False, feasible=False, stop_reason="failed")
    default, _, _ = split_cases(data)
    assert {case["index"] for case in default} == set(range(4, 12))
    included, _, _ = split_cases(data, include_unconverged=True)
    assert {case["index"] for case in included} == {0, *range(4, 12)}
    data["cases"] = data["cases"][:5]
    with pytest.raises(ValueError, match="six eligible"):
        split_cases(data, include_unconverged=True)


def test_cancelled_fit_preserves_saved_model(tmp_path):
    synthetic(tmp_path)
    before = fit_models(tmp_path)
    assert fit_models(tmp_path, seed=8, stop_requested=lambda: True) is None
    assert read_models(tmp_path) == before


def test_generation_saves_real_final_states_and_can_stop(tmp_path):
    settings = RegressionSettings(nx=6, ny=2, case_count=6, max_iterations=3, projection_beta=0)
    events = []
    data = generate_dataset(settings, tmp_path / "cases", emit=events.append)
    assert data["status"] == "complete" and len(data["cases"]) == 6
    assert read_dataset(tmp_path / "cases") == data
    for case in data["cases"]:
        assert case["iterations"] == 3 and case["stop_reason"] == "maximum_iterations"
        assert not case["converged"] and case["feasible"]
        assert len(case["density"]) == 12
        assert np.mean(case["density"]) == pytest.approx(case["physical_volume"])
    with pytest.raises(ValueError, match="six eligible"):
        fit_models(tmp_path / "cases")
    report = fit_models(tmp_path / "cases", include_unconverged=True)
    assert len(report["unfinished_indices"]) == 6
    # Stop during the first actual optimiser run: retain the evaluated design,
    # but never treat the cancelled target as a training example.
    stopped = []
    partial = generate_dataset(replace(settings, max_iterations=100), tmp_path / "partial",
        emit=stopped.append, stop_requested=lambda: any(event["event"] == "regression_iteration" for event in stopped))
    assert partial["status"] == "cancelled" and len(partial["cases"]) == 1
    assert partial["cases"][0]["stop_reason"] == "cancelled"
    assert read_dataset(tmp_path / "partial") == partial


@pytest.mark.parametrize("options", [{"case_count": 5}, {"nx": True}, {"volume_min": .7},
    {"volume_max": float("nan")}, {"load_height": 1.1}, {"material_model": "other"}])
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        RegressionSettings(**options)


@pytest.mark.parametrize("extra", [
    {"volume_constraint": "eq", "optimizer": "gcmma"},
    {"robust": True, "projection_beta": 1, "projection_schedule": "geometric",
     "projection_beta_interval": 1, "projection_beta_max": 4},
    {"max_solid_size": 1.1, "max_size_beta": 1, "max_size_tolerance": .001, "move_limit": 1e-6},
    {"max_void_size": 1.1, "max_void_beta": 1, "max_void_tolerance": .001, "move_limit": 1e-6},
])
def test_imported_configuration_and_quality_match_native_optimiser(tmp_path, extra):
    config = CantileverConfig(nx=6, ny=2, max_iterations=2, **extra)
    settings = RegressionSettings(case_count=6, problem=asdict(config))
    data = generate_dataset(settings, tmp_path / "imported")
    for volume in (settings.volume_min, settings.volume_max):
        assert settings.configuration(volume) == replace(config, volume_fraction=volume)
    assert data["fixed_config"] == json.loads(json.dumps(asdict(settings.configuration(.25))))
    for case in (data["cases"][0], data["cases"][-1]):
        reference = optimise(replace(config, volume_fraction=case["volume_fraction"]))
        evaluation = reference.evaluation
        assert case["compliance"] == pytest.approx(evaluation.compliance)
        np.testing.assert_allclose(case["density"], evaluation.density)
        constraints = [evaluation.constraint, evaluation.epigraph_constraint,
                       evaluation.max_size_constraint, evaluation.max_void_constraint]
        assert case["feasible"] == all(value <= config.feasibility_tolerance
                                       for value in constraints if value is not None)
        assert case["projection_beta"] == evaluation.projection_beta
        assert case["converged"] == reference.converged
    if config.max_solid_size or config.max_void_size:
        # These short runs meet the volume bound but still violate morphology.
        assert any(case["constraints"]["volume"] <= config.feasibility_tolerance
                   and not case["feasible"] for case in data["cases"])
