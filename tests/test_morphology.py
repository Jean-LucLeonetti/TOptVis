"""Geometry, physical spacing, compact-support adjoints, and solver constraints."""

from dataclasses import replace
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.signal import convolve2d
import torch

from ml_opt.config import CantileverConfig
from ml_opt.fem.mesh import StructuredMesh
from ml_opt.morphology import MaximumSolidSize
from ml_opt.optimizers import MovingAsymptotes
from ml_opt.output import save_results
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise
from ml_opt.web.app import create_app


def test_compact_filter_matches_direct_convolution_and_transpose_on_rectangular_cells():
    mesh = StructuredMesh(17, 9, 3.4, .9)
    operator = MaximumSolidSize(mesh, .95, 16, .001)
    x = np.random.default_rng(17).uniform(size=mesh.element_count)
    radius, dx, dy = .475, .2, .1
    a, b = np.meshgrid(np.arange(-3, 4)*dx, np.arange(-5, 6)*dy)
    kernel = np.maximum(0, 1-np.hypot(a, b)/radius)
    kernel /= kernel.sum()
    expected = convolve2d(x.reshape(9, 17), kernel, mode="same", boundary="fill")
    np.testing.assert_allclose(operator.average(x), expected.ravel(), atol=4e-16)
    assert operator.average(np.ones_like(x))[0] < 1  # Outside the domain is void.
    y = np.random.default_rng(18).uniform(size=mesh.element_count)
    np.testing.assert_allclose(x @ operator.average(y), y @ operator.average(x), rtol=1e-14)


def test_erosion_distinguishes_thin_and_thick_solid_strips_and_preserves_void():
    mesh = StructuredMesh(60, 30, 3, 1.5)
    operator = MaximumSolidSize(mesh, .5, 16, .001)
    thin, thick = np.zeros((30, 60)), np.zeros((30, 60))
    thin[13:17, :] = 1  # 0.2 thick, less than the 0.5 disk diameter.
    thick[7:23, :] = 1  # 0.8 thick, greater than the disk diameter.
    def erode(x):
        return operator.eroded(torch.tensor(x.ravel(), dtype=torch.float64)).numpy()
    np.testing.assert_array_equal(erode(np.zeros_like(thin)), 0)
    assert erode(thin).mean() < .001
    assert erode(thick).mean() > .001
    assert np.all(erode(thick) >= erode(thin)-1e-14)
    larger = MaximumSolidSize(mesh, 1.2, 16, .001)
    assert larger.eroded(torch.tensor(thick.ravel())).mean() < erode(thick).mean()


def test_solid_interior_derivative_survives_fft_roundoff_at_density_one():
    mesh = StructuredMesh(20, 12, 2, 1.2)
    operator = MaximumSolidSize(mesh, .65, 16, .001)
    density = torch.ones(mesh.element_count, dtype=torch.float64, requires_grad=True)
    result = operator.eroded(density)
    gradient = torch.autograd.grad(result.sum(), density)[0].numpy()
    average = operator.average(np.ones(mesh.element_count))
    local = 16 / np.tanh(16) * (1 - np.tanh(16*(1-average))**2)
    np.testing.assert_allclose(gradient, operator.average(local), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("material", ["simp", "ramp"])
def test_size_gradient_includes_helmholtz_projection_and_dilated_field(robust, material):
    config = CantileverConfig(nx=8, ny=4, length=2, robust=robust, material_model=material,
        filter_radius=.1, projection_beta=4, max_solid_size=.8)
    with CantileverProblem(config) as problem:
        design = np.random.default_rng(19).uniform(.55, .95, 32)
        current = problem.evaluate(design)
        field = current.field_densities["dilated"] if robust else current.density
        expected = problem.max_size.eroded(torch.tensor(field)).mean().item()
        np.testing.assert_allclose(current.max_size_eroded_volume, expected)
        assert np.linalg.norm(current.max_size_gradient) > .01
        for index in (0, 10, 18, 31):
            step = np.zeros(32)
            step[index] = 1e-5
            plus, minus = problem.evaluate(design+step, False), problem.evaluate(design-step, False)
            np.testing.assert_allclose(current.max_size_gradient[index],
                (plus.max_size_constraint-minus.max_size_constraint)/2e-5, rtol=1e-5, atol=1e-7)
        optimizer = MovingAsymptotes(design, robust=robust, max_size=True)
        optimizer.scale = current.compliance
        assert optimizer.m == (5 if robust else 2)
        assert optimizer.a[-1, 0] == 0  # The epigraph cannot relax maximum size.
        assert optimizer._values(current)[1][-1, 0] == current.max_size_constraint


@pytest.mark.parametrize("options", [{"max_solid_size": -1}, {"max_solid_size": .01},
    {"max_solid_size": float("inf")}, {"max_size_tolerance": 0}, {"max_size_tolerance": 1},
    {"max_size_beta": 0}, {"max_size_beta": 65}, {"max_size_beta": float("nan")},
    {"max_solid_size": 1e10}])
def test_invalid_size_configuration(options):
    with pytest.raises(ValueError):
        CantileverConfig(**options)


def test_disabled_size_constraint_keeps_the_original_single_constraint():
    with CantileverProblem(CantileverConfig(nx=4, ny=2)) as problem:
        current = problem.evaluate(problem.initial_design())
        assert problem.max_size is None
        assert current.max_size_constraint is current.max_size_gradient is None
        assert MovingAsymptotes(problem.initial_design()).m == 1


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_active_constraint_limits_size_and_preserves_volume_feasibility(method):
    base = CantileverConfig(nx=24, ny=8, filter_radius=.04, projection_beta=8,
                           volume_fraction=.5, max_iterations=100, optimizer=method)
    unrestricted = optimise(base)
    config = replace(base, max_solid_size=.35)
    with CantileverProblem(config) as problem:
        unrestricted_size = problem.evaluate(unrestricted.design, False).max_size_constraint
    assert unrestricted_size > config.feasibility_tolerance
    limited = optimise(config)
    assert limited.evaluation.max_size_constraint <= config.feasibility_tolerance
    assert limited.evaluation.constraint <= config.feasibility_tolerance
    assert limited.evaluation.residual < 1e-7
    assert not np.allclose(limited.design, unrestricted.design)


def test_violated_size_cannot_be_reported_as_converged_or_feasible(monkeypatch, tmp_path):
    from ml_opt.optimizers.moving_asymptotes import OptimizerStep

    class FrozenOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, design, *args):
            return OptimizerStep(design.copy())

    monkeypatch.setattr("ml_opt.runner.MovingAsymptotes", FrozenOptimizer)
    config = CantileverConfig(nx=8, ny=4, length=2, volume_fraction=.8, projection_beta=4,
        min_iterations=1, max_iterations=3, convergence_patience=1, feasibility_tolerance=.2)
    assert optimise(config).converged
    result = optimise(replace(config, max_solid_size=.8))
    assert result.evaluation.constraint < config.feasibility_tolerance
    assert result.evaluation.max_size_constraint > config.feasibility_tolerance
    assert not result.converged
    assert result.stop_reason == "maximum_iterations"
    monkeypatch.setattr("ml_opt.output._save_figures", lambda *_: None)
    save_results(result, tmp_path)
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert not summary["feasible"] and not summary["max_size_feasible"]
    assert summary["max_size_field"] == "nominal"
    assert "max_size_constraint" in (tmp_path/"history.csv").read_text().splitlines()[0]


@pytest.mark.parametrize("method", ["mma", "gcmma"])
@pytest.mark.parametrize("robust", [False, True])
def test_api_streams_size_constraint_with_beta_scheduling(method, robust):
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["max_size_supported"]
        assert 'id="max_solid_size"' in client.get("/").text
        response = client.post("/api/run", json={"nx": 8, "ny": 4, "length": 2,
            "load_node_x": 8, "load_node_y": 2, "max_solid_size": .8,
            "optimizer": method, "robust": robust, "projection_beta": 2,
            "projection_schedule": "geometric", "projection_beta_interval": 2,
            "projection_beta_max": 8, "max_iterations": 6})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["event"] == "complete"
        assert events[0]["config"]["max_solid_size"] == .8
        operator = MaximumSolidSize(StructuredMesh(8, 4, 2, 1), .8, 16, .001)
        frames = [event for event in events if event["event"] == "iteration"]
        assert [frame["history"]["projection_beta"] for frame in frames] == [2, 2, 2, 4, 4, 8, 8]
        for frame in frames:
            density = frame["field_densities"]["dilated"] if robust else frame["density"]
            expected = operator.eroded(torch.tensor(density, dtype=torch.float64)).mean().item()
            np.testing.assert_allclose(frame["history"]["max_size_eroded_volume"], expected, atol=1e-15)
            np.testing.assert_allclose(frame["history"]["max_size_constraint"], expected/.001-1, atol=1e-12)
        feasible = frames[-1]["history"]["max_size_constraint"] <= 1e-4
        assert events[-1]["max_size_feasible"] == feasible
        assert events[-1]["feasible"] == (feasible and events[-1]["volume_feasible"] and events[-1]["epigraph_feasible"])
        for options in ({"max_solid_size": -.1}, {"max_solid_size": .01},
                        {"max_size_tolerance": 0}, {"max_size_beta": 65}):
            assert client.post("/api/run", json=options).status_code == 422
