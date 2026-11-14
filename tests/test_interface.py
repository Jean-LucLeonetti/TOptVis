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
        assert 'id="filter_radius"' in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        health = client.get("/api/health").json()
        assert health["status"] == "ready" and not health["running"]
        assert isinstance(health["worker_ready"], bool)
        invalid = {"nx": 6, "ny": 2, "load_node_x": 0, "load_node_y": 1}
        assert client.post("/api/run", json=invalid).status_code == 422
        invalid["load_node_x"] = 7
        assert client.post("/api/run", json=invalid).status_code == 422
        assert client.post("/api/run/missing/stop").status_code == 404
        assert client.post("/api/run", json={"filter_radius": -.1}).status_code == 422


@pytest.mark.parametrize("method", ["mma", "gcmma"])
@pytest.mark.parametrize("radius", [0, .12])
def test_api_streams_real_topology_frames(method, radius):
    with TestClient(create_app()) as client:
        response = client.post("/api/run", json={"nx": 8, "ny": 3,
            "load_node_x": 5, "load_node_y": 1, "load_angle": -60,
            "max_iterations": 4, "optimizer": method, "filter_radius": radius})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[0]["event"] == "started"
        assert events[0]["config"]["filter_radius"] == radius
        assert events[-1]["event"] == "complete"
        frames = [event for event in events if event["event"] == "iteration"]
        assert [frame["history"]["iteration"] for frame in frames] == [0, 1, 2, 3, 4]
        assert all(len(frame["density"]) == 24 for frame in frames)
        assert frames[-1]["history"]["compliance"] < frames[0]["history"]["compliance"]
        assert frames[-1]["density"] != frames[0]["density"]
        expected = optimise(CantileverConfig(nx=8, ny=3, load_node_x=5, load_node_y=1,
                            load_angle=-60, max_iterations=4, optimizer=method, filter_radius=radius))
        np.testing.assert_allclose(frames[-1]["density"], expected.evaluation.density, rtol=1e-9, atol=1e-10)
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
        manager.shutdown()
    assert manager.active is None


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
    pytest.fail("Solver did not produce a terminal event")


def test_worker_reuses_process_and_resets_geometry_load_and_optimizer():
    manager = RunManager()
    base = CantileverConfig(nx=6, ny=2, max_iterations=3)
    try:
        manager.warmup()
        pid = manager._process.pid
        for config in (base, replace(base, nx=8, ny=3, optimizer="gcmma", load=2, filter_radius=.12,
                                     load_node_x=4, load_node_y=1), base):
            job = manager.start(config)
            assert job.process.pid == pid
            events = collect(job)
            manager.finish(job)
            assert events[-1]["event"] == "complete", events[-1]
            frames = [event for event in events if event["event"] == "iteration"]
            expected = optimise(config)
            assert [frame["history"]["iteration"] for frame in frames] == [0, 1, 2, 3]
            np.testing.assert_allclose(frames[-1]["density"], expected.evaluation.density, rtol=1e-9, atol=1e-10)
            assert manager.worker_ready and manager.active is None
    finally:
        manager.shutdown()


def test_disconnected_and_failed_runs_do_not_poison_next_run():
    manager = RunManager()
    config = CantileverConfig(nx=6, ny=2, max_iterations=1000)
    try:
        job = manager.start(config)
        assert job.messages.get(timeout=40)["event"] == "iteration"
        pid = job.process.pid
        manager.finish(job)  # Simulate a disconnected stream, with unread frames.
        assert manager.active is None
        assert manager._process.pid == pid
        # An invalid worker command must emit an error without killing the worker.
        job = manager.start(config)
        manager._commands.put(("invalid-command", {"nx": 0}))
        manager.stop(job.id)
        assert collect(job)[-1]["event"] == "complete"
        manager.finish(job)
        # A new mailbox discards the preceding, separately tagged error.
        job = manager.start(replace(config, max_iterations=2))
        events = collect(job)
        manager.finish(job)
        assert job.process.pid == pid
        assert events[-1]["event"] == "complete"
        assert len([event for event in events if event["event"] == "iteration"]) == 3
    finally:
        manager.shutdown()


def test_worker_recovers_after_process_crash():
    manager = RunManager()
    config = CantileverConfig(nx=6, ny=2, max_iterations=2)
    try:
        job = manager.start(config)
        assert job.messages.get(timeout=40)["event"] == "iteration"
        pid = job.process.pid
        job.process.terminate()
        job.process.join(timeout=10)
        manager.finish(job)
        job = manager.start(config)
        assert job.process.pid != pid
        events = collect(job)
        manager.finish(job)
        assert events[-1]["event"] == "complete"
    finally:
        manager.shutdown()
