"""Void morphology, exterior convention, adjoints, coupled constraints and streams."""

from dataclasses import replace
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from ml_opt.config import CantileverConfig
from ml_opt.fem.mesh import StructuredMesh
from ml_opt.morphology import MorphologicalSize
from ml_opt.optimizers import MovingAsymptotes
from ml_opt.output import save_results
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise
from ml_opt.web.app import create_app


def test_void_morphology_distinguishes_cavity_sizes_and_keeps_exterior_void():
    mesh = StructuredMesh(60, 30, 3, 1.5)
    operator = MorphologicalSize(mesh, .5, 16, .001)
    thin, thick = np.ones((30, 60)), np.ones((30, 60))
    thin[13:17, :] = 0
    thick[7:23, :] = 0
    def remaining(x):
        return operator.remaining_void(torch.tensor(x.ravel(), dtype=torch.float64)).numpy()
    np.testing.assert_array_equal(remaining(np.zeros_like(thin)), 1)
    assert remaining(thin).mean() < .001
    assert remaining(thick).mean() > .001
    assert np.all(remaining(thick) >= remaining(thin)-1e-14)
    # Material zero-extension must not create fictitious solid outside the mesh.
    solid = torch.ones(mesh.element_count, dtype=torch.float64)
    assert operator.remaining_void(solid)[0] > operator.eroded(1-solid)[0]
    larger = MorphologicalSize(mesh, 1.2, 16, .001)
    assert larger.remaining_void(torch.tensor(thick.ravel())).mean() < remaining(thick).mean()


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("solid", [False, True])
def test_void_gradient_through_projection_and_filter_and_independent_solid_constraint(robust, solid):
    config = CantileverConfig(nx=8, ny=4, length=2, robust=robust, filter_radius=.1,
        projection_beta=4, max_void_size=.8, max_solid_size=.9 if solid else 0)
    with CantileverProblem(config) as problem:
        design = np.random.default_rng(27).uniform(.08, .6, 32)
        current = problem.evaluate(design)
        field = current.field_densities["eroded"] if robust else current.density
        expected = problem.max_void.remaining_void(torch.tensor(field)).mean().item()
        np.testing.assert_allclose(current.max_void_remaining_volume, expected)
        assert np.linalg.norm(current.max_void_gradient) > 1
        assert current.max_void_gradient.max() < 1e-10
        for index in (0, 10, 18, 31):
            step = np.zeros(32)
            step[index] = 1e-5
            plus, minus = problem.evaluate(design+step, False), problem.evaluate(design-step, False)
            np.testing.assert_allclose(current.max_void_gradient[index],
                (plus.max_void_constraint-minus.max_void_constraint)/2e-5, rtol=2e-5, atol=1e-7)
        optimizer = MovingAsymptotes(design, robust=robust, max_size=solid, max_void=True)
        optimizer.scale = current.compliance
        assert optimizer.m == (4 if robust else 1) + int(solid) + 1
        assert optimizer.a[-1, 0] == 0
        assert optimizer._values(current)[1][-1, 0] == current.max_void_constraint
        if solid:
            assert optimizer.a[-2, 0] == 0
            assert optimizer._values(current)[1][-2, 0] == current.max_size_constraint


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_active_void_constraint_alone_and_with_solid_limit(method):
    base = CantileverConfig(nx=24, ny=8, filter_radius=.04, projection_beta=8,
                           volume_fraction=.5, max_iterations=130, optimizer=method)
    unrestricted = optimise(base)
    with CantileverProblem(replace(base, max_void_size=.8)) as reference:
        assert reference.evaluate(unrestricted.design, False).max_void_constraint > 0
    for solid in (0, .45):
        result = optimise(replace(base, max_void_size=.8, max_solid_size=solid))
        assert result.evaluation.max_void_constraint <= base.feasibility_tolerance
        assert result.evaluation.constraint <= base.feasibility_tolerance
        if solid:
            assert result.evaluation.max_size_constraint <= base.feasibility_tolerance
        assert not np.allclose(result.design, unrestricted.design)


def test_void_violation_prevents_convergence_and_is_exported(monkeypatch, tmp_path):
    from ml_opt.optimizers.moving_asymptotes import OptimizerStep

    class FrozenOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, design, *args):
            return OptimizerStep(design.copy())

    monkeypatch.setattr("ml_opt.runner.MovingAsymptotes", FrozenOptimizer)
    base = CantileverConfig(nx=8, ny=4, length=2, volume_fraction=.1, projection_beta=4,
        min_iterations=1, max_iterations=3, convergence_patience=1)
    assert optimise(base).converged
    result = optimise(replace(base, max_void_size=.8))
    assert not result.converged and result.stop_reason == "maximum_iterations"
    assert result.evaluation.constraint < base.feasibility_tolerance
    assert result.evaluation.max_void_constraint > base.feasibility_tolerance
    monkeypatch.setattr("ml_opt.output._save_figures", lambda *_: None)
    save_results(result, tmp_path)
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert not summary["feasible"] and not summary["max_void_feasible"]
    assert summary["max_void_field"] == "nominal"
    assert "max_void_constraint" in (tmp_path/"history.csv").read_text().splitlines()[0]


@pytest.mark.parametrize("method", ["mma", "gcmma"])
@pytest.mark.parametrize("robust", [False, True])
def test_live_void_and_solid_constraints_with_beta_scheduling(method, robust):
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["max_void_supported"]
        assert 'id="max_void_size"' in client.get("/").text
        response = client.post("/api/run", json={"nx": 8, "ny": 4, "length": 2,
            "load_node_x": 8, "load_node_y": 2, "max_solid_size": .9, "max_void_size": .8,
            "optimizer": method, "robust": robust, "projection_beta": 2,
            "projection_schedule": "geometric", "projection_beta_interval": 2,
            "projection_beta_max": 8, "max_iterations": 6})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["event"] == "complete"
        assert events[0]["config"]["max_void_size"] == .8
        operator = MorphologicalSize(StructuredMesh(8, 4, 2, 1), .8, 16, .001)
        frames = [event for event in events if event["event"] == "iteration"]
        assert [frame["history"]["projection_beta"] for frame in frames] == [2, 2, 2, 4, 4, 8, 8]
        for frame in frames:
            density = frame["field_densities"]["eroded"] if robust else frame["density"]
            expected = operator.remaining_void(torch.tensor(density, dtype=torch.float64)).mean().item()
            np.testing.assert_allclose(frame["history"]["max_void_remaining_volume"], expected, atol=1e-15)
            np.testing.assert_allclose(frame["history"]["max_void_constraint"], expected/.001-1, atol=1e-12)
        feasible = frames[-1]["history"]["max_void_constraint"] <= 1e-4
        assert events[-1]["max_void_feasible"] == feasible
        assert events[-1]["feasible"] == all(events[-1][name] for name in
            ("volume_feasible", "epigraph_feasible", "max_size_feasible", "max_void_feasible"))
        for options in ({"max_void_size": -.1}, {"max_void_size": .01},
                        {"max_void_tolerance": 0}, {"max_void_beta": 65}):
            assert client.post("/api/run", json=options).status_code == 422


@pytest.mark.parametrize("options", [{"max_void_size": -1}, {"max_void_size": .01},
    {"max_void_size": float("inf")}, {"max_void_tolerance": 0}, {"max_void_tolerance": 1},
    {"max_void_beta": 0}, {"max_void_beta": 65}, {"max_void_beta": float("nan")},
    {"max_void_size": 1e10}])
def test_invalid_void_configuration(options):
    with pytest.raises(ValueError):
        CantileverConfig(**options)
