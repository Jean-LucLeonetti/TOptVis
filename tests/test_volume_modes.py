"""Signed volume bounds, smooth equality rows, real optimisers and streaming."""

from dataclasses import replace
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.optimizers import MovingAsymptotes
from ml_opt.output import save_results
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise
from ml_opt.web.app import create_app


@pytest.mark.parametrize("mode", ["lt", "gt", "eq"])
@pytest.mark.parametrize("robust", [False, True])
def test_volume_values_and_full_gradients(mode, robust):
    config = CantileverConfig(nx=8, ny=4, length=2, volume_constraint=mode,
                              filter_radius=.1, projection_beta=4, robust=robust)
    with CantileverProblem(config) as problem:
        design = np.random.default_rng(22).uniform(.45, .8, 32)
        current = problem.evaluate(design)
        fraction = current.field_volumes["dilated"] if robust else current.density.mean()
        error = fraction/config.volume_fraction-1
        expected = -error if mode == "gt" else abs(error) if mode == "eq" else error
        np.testing.assert_allclose(current.constraint, expected)
        np.testing.assert_allclose(current.volume_error, error)
        for index in (0, 12, 31):
            step = np.zeros(32)
            step[index] = 1e-5
            plus, minus = problem.evaluate(design+step, False), problem.evaluate(design-step, False)
            np.testing.assert_allclose(current.constraint_gradient[index],
                (plus.constraint-minus.constraint)/2e-5, rtol=1e-7)
            if mode == "eq":
                np.testing.assert_allclose(current.volume_gradient[index],
                    (plus.volume_error-minus.volume_error)/2e-5, rtol=1e-7)


def test_equality_uses_opposite_smooth_gradients_at_exact_target():
    config = CantileverConfig(nx=6, ny=2, volume_fraction=.5, volume_constraint="eq")
    with CantileverProblem(config) as problem:
        design = problem.initial_design()
        current = problem.evaluate(design)
    assert current.constraint == current.volume_error == 0
    np.testing.assert_array_equal(current.constraint_gradient, 0)
    assert np.linalg.norm(current.volume_gradient) > 0
    optimizer = MovingAsymptotes(design, volume_mode="eq", volume_tolerance=config.feasibility_tolerance)
    optimizer.scale = current.compliance
    np.testing.assert_array_equal(optimizer._values(current)[1], -config.feasibility_tolerance)
    assert optimizer.m == 2 and np.all(optimizer.a == 0)


@pytest.mark.parametrize("mode", ["lt", "gt", "eq"])
@pytest.mark.parametrize("method", ["mma", "gcmma"])
@pytest.mark.parametrize("robust", [False, True])
def test_both_optimizers_satisfy_selected_volume_mode(mode, method, robust):
    config = CantileverConfig(nx=18, ny=6, filter_radius=.05, projection_beta=4,
        max_iterations=110, optimizer=method, volume_constraint=mode, robust=robust)
    result = optimise(config)
    assert result.evaluation.constraint <= config.feasibility_tolerance
    error = result.evaluation.volume_error
    if mode == "gt":
        assert error >= -config.feasibility_tolerance
        if not robust:
            assert result.evaluation.volume_fraction > .95
    elif mode == "eq":
        assert abs(error) <= config.feasibility_tolerance
    else:
        assert error <= config.feasibility_tolerance
    assert result.evaluation.compliance < result.history[0].compliance


@pytest.mark.parametrize("mode", ["lt", "gt", "eq"])
@pytest.mark.parametrize("robust", [False, True])
def test_live_modes_keep_geometry_constraints_and_report_true_feasibility(mode, robust):
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["volume_modes_supported"]
        assert 'id="volume_constraint"' in client.get("/").text
        assert client.post("/api/run", json={"volume_constraint": "invalid"}).status_code == 422
        response = client.post("/api/run", json={"nx": 8, "ny": 4, "length": 2,
            "load_node_x": 8, "load_node_y": 2, "robust": robust,
            "optimizer": "gcmma" if robust else "mma", "projection_beta": 4,
            "volume_constraint": mode, "max_solid_size": .9, "max_void_size": .8, "max_iterations": 4})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["event"] == "complete"
        assert events[0]["config"]["volume_constraint"] == mode
        frames = [event for event in events if event["event"] == "iteration"]
        for frame in frames:
            row = frame["history"]
            error = row["volume_fraction"]/.4-1
            expected = -error if mode == "gt" else abs(error) if mode == "eq" else error
            np.testing.assert_allclose(row["constraint"], expected, atol=1e-13)
            np.testing.assert_allclose(row["volume_error"], error, atol=1e-13)
            assert row["max_size_constraint"] is not None and row["max_void_constraint"] is not None
        assert events[-1]["volume_feasible"] == (frames[-1]["history"]["constraint"] <= 1e-4)


@pytest.mark.parametrize("mode", ["gt", "eq"])
def test_volume_violation_prevents_convergence_and_exported_feasibility(mode, monkeypatch, tmp_path):
    from ml_opt.optimizers.moving_asymptotes import OptimizerStep

    class FrozenOptimizer:
        def __init__(self, *args, **kwargs):
            pass
        def step(self, design, *args):
            return OptimizerStep(design.copy())

    monkeypatch.setattr("ml_opt.runner.MovingAsymptotes", FrozenOptimizer)
    base = CantileverConfig(nx=6, ny=2, volume_fraction=.4, projection_beta=4,
        min_iterations=1, max_iterations=3, convergence_patience=1)
    assert optimise(base).converged  # The initial projected volume is below the upper bound.
    result = optimise(replace(base, volume_constraint=mode))
    assert not result.converged and result.evaluation.constraint > base.feasibility_tolerance
    monkeypatch.setattr("ml_opt.output._save_figures", lambda *_: None)
    save_results(result, tmp_path)
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert not summary["feasible"]
    assert summary["config"]["volume_constraint"] == mode
    assert summary["normalised_volume_error"] < 0


def test_invalid_volume_mode():
    with pytest.raises(ValueError, match="volume_constraint"):
        CantileverConfig(volume_constraint="invalid")
