"""Material adjoints, beta-stage transitions, stopping, exports and real streams."""

from dataclasses import replace
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from ml_opt.config import CantileverConfig
from ml_opt.material import elastic_modulus
from ml_opt.output import save_results
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise
from ml_opt.web.app import create_app


@pytest.mark.parametrize("model,penalty", [("simp", 1), ("simp", 3), ("ramp", 0), ("ramp", 3)])
def test_material_endpoints_and_exact_stiffness_derivative(model, penalty):
    config = CantileverConfig(material_model=model, penalisation=penalty,
                              young_modulus=2, minimum_modulus=.01)
    rho = torch.tensor([0, .2, .6, 1], dtype=torch.float64, requires_grad=True)
    modulus = elastic_modulus(rho, config)
    np.testing.assert_allclose(modulus.detach().numpy()[[0, -1]], [.01, 2])
    expected = (penalty * rho.detach().numpy()**(penalty-1) if model == "simp" else
                (1+penalty) / (1+penalty*(1-rho.detach().numpy()))**2)
    np.testing.assert_allclose(torch.autograd.grad(modulus.sum(), rho)[0], 1.99*expected)
    assert torch.all(torch.diff(modulus) > 0)


@pytest.mark.parametrize("model", ["simp", "ramp"])
@pytest.mark.parametrize("robust", [False, True])
def test_complete_filtered_projected_material_gradients_after_beta_change(model, robust):
    config = CantileverConfig(nx=6, ny=3, material_model=model, penalisation=2.5,
                              filter_radius=.12, projection_beta=2, robust=robust)
    with CantileverProblem(config) as problem:
        design = np.random.default_rng(19).uniform(.25, .75, 18)
        before = problem.evaluate(design)
        problem.projection_beta = 7
        current = problem.evaluate(design)
        assert current.projection_beta == 7
        assert not np.allclose(before.density, current.density)
        assert not np.allclose(before.objective_gradient, current.objective_gradient)
        for index in (0, 8, 17):
            step = np.zeros(18)
            step[index] = 1e-5
            plus, minus = problem.evaluate(design+step, False), problem.evaluate(design-step, False)
            np.testing.assert_allclose(current.objective_gradient[index],
                (plus.compliance-minus.compliance)/2e-5, rtol=4e-5)
            np.testing.assert_allclose(current.constraint_gradient[index],
                (plus.constraint-minus.constraint)/2e-5, rtol=1e-7)
            if robust:
                for name in current.field_gradients:
                    np.testing.assert_allclose(current.field_gradients[name][index],
                        (plus.field_compliances[name]-minus.field_compliances[name])/2e-5, rtol=4e-5)


@pytest.mark.parametrize("method", ["mma", "gcmma"])
@pytest.mark.parametrize("robust", [False, True])
def test_schedule_frames_are_evaluated_at_recorded_beta(method, robust, monkeypatch):
    import ml_opt.runner as runner

    config = CantileverConfig(nx=6, ny=3, material_model="ramp", optimizer=method, robust=robust,
        filter_radius=.12, projection_beta=2, projection_schedule="geometric",
        projection_beta_interval=2, projection_beta_multiplier=2, projection_beta_max=7, max_iterations=6)
    original_problem, original_optimizer = runner.CantileverProblem, runner.MovingAsymptotes
    problems, optimizers, snapshots = [], [], []

    class Problem(original_problem):
        def __init__(self, config):
            super().__init__(config)
            problems.append(self)

    class Optimizer(original_optimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            optimizers.append(self)

        def step(self, design, current, evaluate):
            # Verify that the local approximation never sees stale values/gradients.
            assert current.projection_beta == problems[-1].projection_beta
            fresh = evaluate(design)
            np.testing.assert_allclose(current.objective_gradient, fresh.objective_gradient)
            np.testing.assert_allclose(current.constraint_gradient, fresh.constraint_gradient)
            return super().step(design, current, evaluate)

    monkeypatch.setattr(runner, "CantileverProblem", Problem)
    monkeypatch.setattr(runner, "MovingAsymptotes", Optimizer)
    result = optimise(config, evaluation_callback=lambda row, design, evaluation:
                      snapshots.append((row, design, evaluation)))
    assert [row.projection_beta for row in result.history] == [2, 2, 2, 4, 4, 7, 7]
    assert len(optimizers) == 3
    assert result.config.projection_beta == 2  # Preserve the requested starting settings.
    assert result.evaluation.projection_beta == 7
    for row, design, evaluation in snapshots:
        with CantileverProblem(replace(config, projection_beta=row.projection_beta,
                                       projection_schedule="fixed")) as reference:
            expected = reference.evaluate(design)
        np.testing.assert_allclose(evaluation.density, expected.density, atol=1e-13)
        np.testing.assert_allclose(row.compliance, expected.compliance, rtol=1e-10)
        np.testing.assert_allclose(row.volume_fraction, expected.volume_fraction, atol=1e-13)


def test_schedule_defers_convergence_until_final_beta_and_resets_patience():
    config = CantileverConfig(nx=6, ny=3, projection_beta=2, projection_schedule="geometric",
        projection_beta_interval=2, projection_beta_max=8, max_iterations=20, min_iterations=1,
        design_tolerance=1, objective_tolerance=1e12, feasibility_tolerance=1e12, convergence_patience=2)
    result = optimise(config)
    assert result.converged
    assert result.history[-1].iteration == 6
    assert [row.projection_beta for row in result.history[-2:]] == [8, 8]


def test_schedule_cancellation_preserves_last_evaluated_beta_and_resets_next_run():
    config = CantileverConfig(nx=4, ny=2, projection_beta=2, projection_schedule="geometric",
        projection_beta_interval=1, projection_beta_max=8, max_iterations=10)
    rows = []
    result = optimise(config, callback=rows.append, stop_requested=lambda: len(rows) >= 3)
    assert result.stop_reason == "cancelled"
    assert result.evaluation.projection_beta == 4
    second = optimise(replace(config, max_iterations=1))
    assert [row.projection_beta for row in second.history] == [2, 2]


def test_parallel_robust_fields_match_sequential_across_beta_stages():
    config = CantileverConfig(nx=6, ny=3, robust=True, material_model="ramp", filter_radius=.12,
        projection_beta=2, projection_schedule="geometric", projection_beta_interval=1,
        projection_beta_max=7, max_iterations=3)
    sequential = optimise(config)
    parallel = optimise(replace(config, field_execution="parallel"))
    assert [row.projection_beta for row in parallel.history] == [2, 2, 4, 7]
    np.testing.assert_allclose(parallel.design, sequential.design, rtol=1e-9, atol=1e-10)
    for name in parallel.evaluation.field_compliances:
        np.testing.assert_allclose(parallel.evaluation.field_compliances[name],
                                   sequential.evaluation.field_compliances[name], rtol=1e-9)


@pytest.mark.parametrize("options", [
    {"material_model": "other"}, {"penalisation": .5},
    {"material_model": "ramp", "penalisation": -1}, {"penalisation": float("nan")},
    {"projection_schedule": "other"}, {"projection_schedule": "geometric"},
    {"projection_schedule": "geometric", "projection_beta": 4, "projection_beta_max": 2},
    {"projection_beta_interval": 0}, {"projection_beta_interval": True},
    {"projection_beta_multiplier": 1}, {"projection_beta_multiplier": float("inf")},
    {"projection_beta_max": 0},
])
def test_invalid_material_or_schedule(options):
    with pytest.raises(ValueError):
        CantileverConfig(**options)


def test_api_streams_material_and_actual_beta_and_validates_coupled_inputs():
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["material_schedule_supported"] is True
        page = client.get("/").text
        assert 'id="panel-material"' in page and 'id="projection_schedule"' in page
        for options in ({"penalisation": .5}, {"material_model": "ramp", "penalisation": -1},
                        {"projection_schedule": "geometric", "projection_beta": 0},
                        {"projection_schedule": "geometric", "projection_beta": 4, "projection_beta_max": 2}):
            assert client.post("/api/run", json=options).status_code == 422
        response = client.post("/api/run", json={"nx": 6, "ny": 3, "load_node_x": 6,
            "load_node_y": 1, "material_model": "ramp", "penalisation": 0,
            "projection_beta": 2, "projection_schedule": "geometric", "projection_beta_interval": 1,
            "projection_beta_max": 7, "max_iterations": 4})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["event"] == "complete"
        assert events[0]["config"]["material_model"] == "ramp"
        assert events[0]["config"]["penalisation"] == 0
        rows = [event["history"] for event in events if event["event"] == "iteration"]
        assert [row["projection_beta"] for row in rows] == [2, 2, 4, 7, 7]


def test_export_records_initial_settings_and_final_beta(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_opt.output._save_figures", lambda *_: None)
    result = optimise(CantileverConfig(nx=4, ny=2, material_model="ramp", penalisation=0,
        projection_beta=2, projection_schedule="geometric", projection_beta_interval=1,
        projection_beta_max=4, max_iterations=2))
    save_results(result, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["config"]["material_model"] == "ramp"
    assert summary["config"]["projection_beta"] == 2
    assert summary["final_projection_beta"] == 4
    assert "projection_beta" in (tmp_path / "history.csv").read_text().splitlines()[0]
