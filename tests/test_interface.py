"""Load placement and real streamed solver iterations for the local interface."""

from dataclasses import replace
import json
import queue
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.fem.mesh import StructuredMesh
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise
from ml_opt.web.app import create_app
from ml_opt.web.jobs import RunManager


def test_selected_node_force_and_direction():
    mesh = StructuredMesh(6, 4, 3, 1)
    force = mesh.cantilever_force(2, 30, (3, 1)).reshape(-1, 2)
    node = 1 * 7 + 3
    np.testing.assert_allclose(force[node], [np.sqrt(3), 1])
    assert np.flatnonzero(np.linalg.norm(force, axis=1)).tolist() == [node]
    with pytest.raises(ValueError, match="clamped"):
        mesh.cantilever_force(1, node=(0, 1))


def test_load_changes_affect_the_actual_state_solve():
    base = CantileverConfig(nx=8, ny=4, load_node_x=8, load_node_y=2)
    def evaluate(config):
        problem = CantileverProblem(config)
        return problem.evaluate(problem.initial_design(), False)
    initial = evaluate(base)
    doubled = evaluate(replace(base, load=2))
    moved = evaluate(replace(base, load_node_x=4))
    horizontal = evaluate(replace(base, load_angle=0))
    np.testing.assert_allclose(doubled.compliance, 4 * initial.compliance, rtol=1e-10)
    assert moved.compliance < initial.compliance
    assert horizontal.compliance < initial.compliance


def test_snapshot_is_evaluated_and_cancellation_preserves_last_design():
    frames = []
    def snapshot(entry, density):
        frames.append((entry, density.copy()))
        density[:] = -100  # The callback receives a copy, not the solver's design.
    result = optimise(CantileverConfig(nx=6, ny=2), snapshot_callback=snapshot,
                       stop_requested=lambda: len(frames) >= 4)
    assert result.stop_reason == "cancelled"
    assert [entry.iteration for entry, _ in frames] == [0, 1, 2, 3]
    np.testing.assert_array_equal(result.evaluation.density, frames[-1][1])
    assert result.evaluation.compliance == frames[-1][0].compliance


def test_local_page_and_validation():
    with TestClient(create_app()) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/health").json() == {"status": "ready", "running": False}
        invalid = {"nx": 6, "ny": 2, "load_node_x": 0, "load_node_y": 1}
        assert client.post("/api/run", json=invalid).status_code == 422
        invalid["load_node_x"] = 7
        assert client.post("/api/run", json=invalid).status_code == 422
        assert client.post("/api/run/missing/stop").status_code == 404


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_api_streams_real_topology_frames(method):
    with TestClient(create_app()) as client:
        response = client.post("/api/run", json={"nx": 8, "ny": 3,
            "load_node_x": 5, "load_node_y": 1, "load_angle": -60,
            "max_iterations": 4, "optimizer": method})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[0]["event"] == "started"
        assert events[-1]["event"] == "complete"
        frames = [event for event in events if event["event"] == "iteration"]
        assert [frame["history"]["iteration"] for frame in frames] == [0, 1, 2, 3, 4]
        assert all(len(frame["density"]) == 24 for frame in frames)
        assert frames[-1]["history"]["compliance"] < frames[0]["history"]["compliance"]
        assert frames[-1]["density"] != frames[0]["density"]
        assert not client.get("/api/health").json()["running"]


def test_manager_serializes_and_cancels_its_worker():
    manager = RunManager()
    job = manager.start(CantileverConfig(nx=6, ny=2, max_iterations=1000))
    try:
        with pytest.raises(RuntimeError, match="already running"):
            manager.start(CantileverConfig())
        assert manager.stop(job.id)
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            try:
                event = job.messages.get(timeout=1)
            except queue.Empty:
                continue
            if event["event"] == "complete":
                assert event["stop_reason"] == "cancelled"
                break
            assert event["event"] != "error", event
        else:
            pytest.fail("The cancelled solver did not finish")
    finally:
        manager.finish(job)
    assert manager.active is None
