"""Ridge objective, unpenalised intercept, held-out isolation and uncapped orders."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.regression import fit_models, fit_polynomial, predict, read_models, volume_coefficients
from ml_opt.topology_regression import fit_topology_models, predict_topology, read_topology_models


def dataset(directory, count=16, narrow=False):
    config = CantileverConfig(nx=6, ny=2)
    volumes = np.linspace(.5-1e-5, .5+1e-5, count) if narrow else np.linspace(.25, .65, count)
    cases = [{"index": i, "volume_fraction": float(v), "compliance": float(10-5*v+12*v*v),
              "density": (.1+np.linspace(.2,.7,12)*v).tolist(),
              "converged": True, "feasible": True, "stop_reason": "design_and_objective_stable"}
             for i, v in enumerate(volumes)]
    data = {"format": "toptvis-regression-v1", "id": directory.name, "status": "complete",
            "fixed_config": asdict(config), "cases": cases}
    (directory / "dataset.json").write_text(json.dumps(data))
    return data


@pytest.mark.parametrize("alpha", [1e-4, .5, 1e6])
@pytest.mark.parametrize("outputs", [1, 3])
def test_ridge_matches_augmented_least_squares_and_leaves_intercept_free(alpha, outputs):
    rng = np.random.default_rng(42)
    x = np.linspace(-1, 1, 9)
    targets = rng.normal(size=(9, outputs)) + 10
    if outputs == 1:
        targets = targets[:, 0]
    degree = 12  # More coefficients than cases; ridge still has a unique fit.
    design = np.polynomial.polynomial.polyvander(x, degree)
    penalty = np.diag([0.] + [np.sqrt(alpha)]*degree)
    rhs = np.concatenate((targets, np.zeros((degree+1,)+targets.shape[1:])))
    expected = np.linalg.lstsq(np.vstack((design, penalty)), rhs, rcond=None)[0]
    actual = fit_polynomial(x, targets, degree, method="ridge", ridge_alpha=alpha)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)
    shifted = fit_polynomial(x, targets+500, degree, method="ridge", ridge_alpha=alpha)
    np.testing.assert_allclose(shifted[1:], actual[1:], atol=1e-10)
    np.testing.assert_allclose(shifted[0], actual[0]+500, atol=1e-10)


def test_ols_above_ten_and_zero_penalty_equivalence(tmp_path):
    dataset(tmp_path, count=40)
    ols = fit_models(tmp_path, degrees=[12])
    zero = fit_models(tmp_path, degrees=[12], method="ridge", ridge_alpha=0)
    assert ols["degrees"] == [12]
    np.testing.assert_array_equal(ols["models"]["polynomial_12"]["scaled_coefficients"],
                                  zero["models"]["polynomial_12"]["scaled_coefficients"])
    assert zero["method"] == "ridge" and zero["ridge_alpha"] == 0


def test_ridge_higher_order_than_cases_shrinks_coefficients_and_reloads(tmp_path):
    dataset(tmp_path, count=6)
    weak = fit_models(tmp_path, degrees=[1, 32], method="ridge", ridge_alpha=.001)
    strong = fit_models(tmp_path, degrees=[1, 32], method="ridge", ridge_alpha=1e3)
    model = strong["models"]["polynomial_32"]
    assert len(strong["train_indices"]) == 4
    assert np.linalg.norm(model["scaled_coefficients"][1:]) < np.linalg.norm(weak["models"]["polynomial_32"]["scaled_coefficients"][1:])
    assert np.all(np.isfinite(predict(model, [.25, .45, .65])))
    assert read_models(tmp_path) == strong
    assert fit_models(tmp_path, degrees=[64], method="ridge", stop_requested=lambda: True) is None
    assert read_models(tmp_path) == strong
    with pytest.raises(ValueError, match="needs at least"):
        fit_models(tmp_path, degrees=[32], method="ridge", ridge_alpha=0)


def test_both_ridge_targets_are_isolated_from_held_out_values(tmp_path):
    data = dataset(tmp_path)
    scalar = fit_models(tmp_path, degrees=[20], method="ridge")
    topology = fit_topology_models(tmp_path, degrees=[20], method="ridge")
    assert scalar["test_indices"] == topology["test_indices"]
    for i in scalar["test_indices"]:
        data["cases"][i]["compliance"] *= 10
        data["cases"][i]["density"] = [.99]*12
    (tmp_path / "dataset.json").write_text(json.dumps(data))
    changed_scalar = fit_models(tmp_path, degrees=[20], method="ridge")
    changed_topology = fit_topology_models(tmp_path, degrees=[20], method="ridge")
    np.testing.assert_array_equal(scalar["models"]["polynomial_20"]["scaled_coefficients"], changed_scalar["models"]["polynomial_20"]["scaled_coefficients"])
    np.testing.assert_array_equal(topology["models"]["20"]["coefficients"], changed_topology["models"]["20"]["coefficients"])
    assert changed_scalar["models"]["polynomial_20"]["test"]["mae"] > scalar["models"]["polynomial_20"]["test"]["mae"]
    assert changed_topology["models"]["20"]["test"]["mae"] > topology["models"]["20"]["test"]["mae"]


def test_high_order_topology_ridge_preserves_fixed_regions(tmp_path):
    data = dataset(tmp_path, count=6)
    config = CantileverConfig(nx=6, ny=2, non_design_regions=[
        {"kind":"solid", "selection":{"kind":"surface", "start":[0,0], "end":[1/6,1]}}])
    data["fixed_config"] = asdict(config)
    for case in data["cases"]:
        for i in (0, 6):
            case["density"][i] = 1
    (tmp_path / "dataset.json").write_text(json.dumps(data))
    report = fit_topology_models(tmp_path, degrees=[32], method="ridge", ridge_alpha=.1)
    assert read_topology_models(tmp_path) == report
    predicted = predict_topology(report["models"]["32"], [.3, .5, .6])
    np.testing.assert_array_equal(predicted[:, [0,6]], 1)
    assert np.all(np.isfinite(predicted)) and np.all((predicted>=0) & (predicted<=1))


def test_overflowing_original_coefficients_keep_scaled_predictions_and_export(tmp_path):
    dataset(tmp_path, count=6, narrow=True)
    report = fit_models(tmp_path, degrees=[128], method="ridge")
    model = report["models"]["polynomial_128"]
    assert model["coefficients"] is None
    assert np.isfinite(predict(model, .5))
    assert read_models(tmp_path) == report
    assert volume_coefficients(np.ones(400), .5, .001) is None
    overflowing = {"degree":2048, "input_offset":.5, "input_scale":.01,
                   "coefficients":np.ones((2049,1)).tolist()}
    with pytest.raises(ValueError, match="numerical range"):
        predict_topology(overflowing, .9)


@pytest.mark.parametrize("options", [{"method":"other"}, {"ridge_alpha":-1},
    {"ridge_alpha":float("nan")}, {"ridge_alpha":float("inf")}, {"ridge_alpha":True}])
def test_invalid_ridge_options_preserve_artifacts(tmp_path, options):
    dataset(tmp_path)
    before = fit_models(tmp_path)
    with pytest.raises(ValueError, match="method|penalty"):
        fit_models(tmp_path, **options)
    assert read_models(tmp_path) == before
