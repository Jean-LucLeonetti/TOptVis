"""Whole-case density regression and direct physical-field structural validation."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.problem import CantileverProblem
from ml_opt.regression import fit_models, generate_dataset, RegressionSettings
from ml_opt.topology_regression import (fit_topology_models, predict_topology,
    read_topology_models, evaluate_topology_prediction)


def synthetic(directory, *, fixed=False):
    config = CantileverConfig(nx=6, ny=2, filter_radius=.1, projection_beta=3,
        non_design_regions=[{"kind": "solid", "selection": {"kind": "surface", "start": [0, 0], "end": [1/6, 1]}}] if fixed else [])
    data = {"format": "toptvis-regression-v1", "id": directory.name,
            "status": "complete", "fixed_config": asdict(config), "cases": []}
    for i, v in enumerate(np.linspace(.3, .7, 16)):
        density = .1 + np.linspace(.2, .7, 12)*v + .1*v*v
        if fixed:
            density[[0, 6]] = 1
        data["cases"].append({"index": i, "volume_fraction": float(v), "compliance": 1+v*v,
            "density": density.tolist(), "feasible": True, "converged": True,
            "stop_reason": "design_and_objective_stable"})
    (directory / "dataset.json").write_text(json.dumps(data))
    return data, config


def test_density_recovery_shared_split_and_persistence(tmp_path):
    data, _ = synthetic(tmp_path)
    scalar = fit_models(tmp_path, degrees=[1, 2, 3])
    report = fit_topology_models(tmp_path, degrees=[1, 2, 3])
    assert report["train_indices"] == scalar["train_indices"]
    assert report["test_indices"] == scalar["test_indices"]
    assert report["models"]["2"]["test"]["mae"] < 1e-12
    assert report["models"]["1"]["test"]["mae"] > 1e-5
    assert read_topology_models(tmp_path) == report
    assert (tmp_path / "models.json").exists()
    assert predict_topology(report["models"]["2"], .5).shape == (12,)
    assert predict_topology(report["models"]["2"], [.4, .5]).shape == (2, 12)
    assert report["baseline"]["test"]["mae"] > report["models"]["2"]["test"]["mae"]
    for case in data["cases"]:
        np.testing.assert_allclose(predict_topology(report["models"]["2"], case["volume_fraction"]), case["density"], atol=1e-12)


def test_test_density_targets_never_enter_fitting(tmp_path):
    data, _ = synthetic(tmp_path)
    before = fit_topology_models(tmp_path)
    for index in before["test_indices"]:
        data["cases"][index]["density"] = [.9]*12
    (tmp_path / "dataset.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="do not match"):
        read_topology_models(tmp_path)
    after = fit_topology_models(tmp_path)
    for key in before["models"]:
        np.testing.assert_array_equal(before["models"][key]["coefficients"], after["models"][key]["coefficients"])
        assert after["models"][key]["test"]["mae"] > before["models"][key]["test"]["mae"]


def test_fixed_regions_clipping_and_cancel_preserve_saved_models(tmp_path):
    _, _ = synthetic(tmp_path, fixed=True)
    report = fit_topology_models(tmp_path)
    fitted = report["models"]["2"]
    predicted = predict_topology(fitted, [.01, .5, .99])
    np.testing.assert_array_equal(predicted[:, [0, 6]], 1)
    assert np.all((predicted >= 0) & (predicted <= 1))
    assert fit_topology_models(tmp_path, stop_requested=lambda: True) is None
    assert read_topology_models(tmp_path) == report
    raw = {"degree": 1, "input_offset": 0, "input_scale": 1, "coefficients": [[-2, 3], [0, 0]]}
    np.testing.assert_array_equal(predict_topology(raw, .5), [0, 1])
    np.testing.assert_array_equal(predict_topology(raw, .5, bounded=False), [-2, 3])


@pytest.mark.parametrize("bad", [None, [0]*3, [float("nan")]*12, [1.1]*12])
def test_invalid_density_targets_rejected(tmp_path, bad):
    data, _ = synthetic(tmp_path)
    data["cases"][0]["density"] = bad
    (tmp_path / "dataset.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="density|densities"):
        fit_topology_models(tmp_path)


@pytest.mark.parametrize("robust", [False, True])
def test_structural_check_uses_physical_nominal_density_once(tmp_path, robust):
    data, config = synthetic(tmp_path)
    config = replace(config, robust=robust, state_solver="fom")
    data["fixed_config"] = asdict(config)
    (tmp_path / "dataset.json").write_text(json.dumps(data))
    report = fit_topology_models(tmp_path)
    case = data["cases"][report["test_indices"][0]]
    volume = case["volume_fraction"]
    result = evaluate_topology_prediction(tmp_path, degree=2, volume=volume)
    direct = replace(config, volume_fraction=volume, robust=False, filter_radius=0, projection_beta=0)
    with CantileverProblem(direct) as problem:
        reference = problem.evaluate(np.array(case["density"]), sensitivities=False)
    assert result["prediction"]["objective"] == pytest.approx(reference.compliance, rel=1e-10)
    assert result["reference"]["objective"] == pytest.approx(reference.compliance)
    assert abs(result["relative_objective_change"]) < 1e-10
    assert result["reference_index"] == case["index"]
    assert result["robust_dataset"] is robust
    assert result["field"] == "nominal"
    assert evaluate_topology_prediction(tmp_path, degree=2, volume=volume, stop_requested=lambda: True) is None
    arbitrary = evaluate_topology_prediction(tmp_path, degree=2, volume=.512345)
    assert arbitrary["reference_index"] is None and "reference" not in arbitrary


def test_real_generated_topologies_can_be_fitted(tmp_path):
    directory = tmp_path / "actual"
    data = generate_dataset(RegressionSettings(nx=6, ny=2, case_count=6, max_iterations=2), directory)
    report = fit_topology_models(directory, include_unconverged=True)
    assert report["element_count"] == 12
    assert len(report["unfinished_indices"]) == 6
    volume = data["cases"][report["test_indices"][0]]["volume_fraction"]
    checked = evaluate_topology_prediction(directory, degree=1, volume=volume)
    assert checked["prediction"]["objective"] > 0
    assert checked["prediction"]["residual"] < 1e-8
