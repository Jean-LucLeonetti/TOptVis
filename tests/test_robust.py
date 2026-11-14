"""Three-field physics, full adjoints, execution parity, and worker lifecycle."""

from dataclasses import replace
import json
import multiprocessing as mp
import queue
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.output import save_results
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise
from ml_opt.web.app import create_app
from ml_opt.web.jobs import RunManager


def collect(job):
    events = []
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try:
            event = job.messages.get(timeout=1)
        except queue.Empty:
            continue
        events.append(event)
        if event["event"] in ("complete", "error"):
            return events
    pytest.fail("Robust solver did not produce a terminal event")


def configuration(**options):
    return replace(CantileverConfig(nx=6, ny=3, robust=True, filter_radius=.12,
                                   projection_beta=4, projection_eta=.45), **options)


def test_three_fields_match_independent_problems_and_full_finite_differences():
    config = configuration()
    design = np.random.default_rng(18).uniform(.25, .75, 18)
    with CantileverProblem(config) as problem:
        result = problem.evaluate(design)
        assert problem.solve_count == 3
        assert result.worst_field == "eroded"
        for name, eta in (("eroded", .55), ("nominal", .45), ("dilated", .35)):
            with CantileverProblem(replace(config, robust=False, projection_eta=eta)) as reference:
                expected = reference.evaluate(design)
            np.testing.assert_allclose(result.field_densities[name], expected.density, atol=1e-14)
            np.testing.assert_allclose(result.field_compliances[name], expected.compliance, rtol=1e-10)
            np.testing.assert_allclose(result.field_displacements[name], expected.displacement, rtol=1e-10)
            if name == "eroded":
                np.testing.assert_allclose(result.objective_gradient, expected.objective_gradient, rtol=1e-10)
            if name == "dilated":
                np.testing.assert_allclose(result.constraint_gradient, expected.constraint_gradient, rtol=1e-10)
        np.testing.assert_array_equal(result.density, result.field_densities["nominal"])
        assert np.all(result.field_densities["eroded"] <= result.density)
        assert np.all(result.density <= result.field_densities["dilated"])
        assert result.compliance == max(result.field_compliances.values())
        assert result.volume_fraction == result.field_volumes["dilated"]
        assert result.volume_fraction > result.density.mean()
        for index in (0, 5, 8, 17):
            step = np.zeros(18)
            step[index] = 1e-5
            plus, minus = problem.evaluate(design + step, False), problem.evaluate(design - step, False)
            np.testing.assert_allclose(result.objective_gradient[index],
                (plus.compliance - minus.compliance) / 2e-5, rtol=3e-5)
            np.testing.assert_allclose(result.constraint_gradient[index],
                (plus.constraint - minus.constraint) / 2e-5, rtol=1e-7)


def test_persistent_parallel_workers_match_sequential_values_and_gradients():
    config = configuration()
    workers = []
    with CantileverProblem(config) as sequential, CantileverProblem(
            replace(config, field_execution="parallel")) as parallel:
        workers = [p.pid for p in parallel.fields._workers]
        for scale, sensitivities in ((1., True), (.8, False), (.9, True)):
            design = np.linspace(.3, .7, 18) * scale
            actual, expected = parallel.evaluate(design, sensitivities), sequential.evaluate(design, sensitivities)
            assert [p.pid for p in parallel.fields._workers] == workers
            assert parallel.solve_count == sequential.solve_count
            for attribute in ("compliance", "constraint", "density", "displacement"):
                np.testing.assert_allclose(getattr(actual, attribute), getattr(expected, attribute), rtol=1e-10)
            if sensitivities:
                np.testing.assert_allclose(actual.objective_gradient, expected.objective_gradient, rtol=1e-10)
                np.testing.assert_allclose(actual.constraint_gradient, expected.constraint_gradient, rtol=1e-10)
            else:
                assert actual.objective_gradient is actual.constraint_gradient is None
    parallel.close()
    assert not set(workers) & {p.pid for p in mp.active_children()}


def test_parallel_failure_closes_remaining_workers():
    with CantileverProblem(configuration(field_execution="parallel")) as problem:
        workers = [p.pid for p in problem.fields._workers]
        with pytest.raises(RuntimeError, match="failed|exited"):
            problem.fields.solve(np.zeros((3, 18)))
        assert not set(workers) & {p.pid for p in mp.active_children()}


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_robust_optimizers_and_saved_fields(method, tmp_path):
    result = optimise(configuration(nx=12, ny=4, optimizer=method, max_iterations=35))
    assert result.evaluation.compliance < .85 * result.history[0].compliance
    assert result.evaluation.constraint <= result.config.feasibility_tolerance
    assert result.evaluation.residual < 1e-7
    assert all(entry.state_solves % 3 == 0 for entry in result.history)
    destination = save_results(result, tmp_path)
    with np.load(destination / "result.npz") as saved:
        for name in ("eroded", "nominal", "dilated"):
            np.testing.assert_array_equal(saved[f"density_{name}"].ravel(), result.evaluation.field_densities[name])
            np.testing.assert_array_equal(saved[f"displacement_{name}"].ravel(), result.evaluation.field_displacements[name])
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["constrained_volume_field"] == "dilated"
    assert summary["density_field"] == "nominal"
    assert (destination / "robust-fields.png").is_file()


@pytest.mark.parametrize("execution", ["sequential", "parallel"])
@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_api_runs_robust_fields_and_reports_dilated_volume(execution, method):
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["robust_supported"]
        response = client.post("/api/run", json={"nx": 6, "ny": 3,
            "load_node_x": 6, "load_node_y": 1, "max_iterations": 2,
            "robust": True, "projection_beta": 4, "field_execution": execution, "optimizer": method})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["event"] == "complete", events[-1]
        frames = [event for event in events if event["event"] == "iteration"]
        assert len(frames) == 3
        for frame in frames:
            assert set(frame["field_densities"]) == {"eroded", "nominal", "dilated"}
            np.testing.assert_array_equal(frame["density"], frame["field_densities"]["nominal"])
            np.testing.assert_allclose(frame["history"]["volume_fraction"],
                                       np.mean(frame["field_densities"]["dilated"]))
        solves = frames[-1]["history"]["state_solves"]
        if method == "mma":
            assert solves == 9
        else:
            assert solves > 9 and solves % 3 == 0
        assert not client.get("/api/health").json()["running"]


def test_parallel_cancellation_preserves_last_fields_and_worker_can_be_reused():
    manager = RunManager()
    config = configuration(field_execution="parallel", max_iterations=1000)
    try:
        job = manager.start(config)
        first = job.messages.get(timeout=40)
        assert first["event"] == "iteration"
        pid = job.process.pid
        manager.stop(job.id)
        events = collect(job)
        assert events[-1]["event"] == "complete", events[-1]
        assert events[-1]["stop_reason"] == "cancelled"
        manager.finish(job)
        job = manager.start(replace(config, field_execution="sequential", max_iterations=1))
        events = collect(job)
        assert events[-1]["event"] == "complete", events[-1]
        assert job.process.pid == pid
        assert [e["history"]["iteration"] for e in events if e["event"] == "iteration"] == [0, 1]
        manager.finish(job)
    finally:
        manager.shutdown()


@pytest.mark.parametrize("options", [{"projection_beta": 0}, {"robust_eta_offset": 0},
    {"robust_eta_offset": float("nan")}, {"projection_eta": .1},
    {"projection_eta": .95}, {"field_execution": "threads"}])
def test_invalid_robust_settings(options):
    with pytest.raises(ValueError):
        configuration(**options)


def test_api_rejects_invalid_coupled_thresholds_before_streaming():
    with TestClient(create_app()) as client:
        for options in ({"projection_beta": 0}, {"projection_eta": .95}):
            assert client.post("/api/run", json={"robust": True, **options}).status_code == 422
