"""Streamed offline jobs, persistent downloads, and worker sharing."""

import io
import json
import queue
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.web.app import create_app,RunRequest
from ml_opt.web.rom import SnapshotRequest
from ml_opt.web.jobs import RunManager


SETTINGS = {"nx": 6, "ny": 4, "load_heights": [.5], "load_angles": [-90, 0],
            "volume_fractions": [.4], "iterations_per_case": 3, "snapshot_stride": 2}


def events(response):
    assert response.status_code == 200, response.text
    result = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert result[0]["event"] == "started"
    assert result[-1]["event"] == "complete", result[-1]
    return result


def test_api_generate_build_download_rebuild_and_reload(tmp_path):
    with TestClient(create_app(training_root=tmp_path)) as client:
        assert client.get("/api/health").json()["rom_training_supported"]
        assert client.get("/api/rom/trainings").json()["trainings"] == []
        page = client.get("/").text
        assert 'id="rom-snapshot-form"' in page and 'id="rom-basis-form"' in page
        assert client.get("/static/rom-training.js").status_code == 200
        generated = events(client.post("/api/rom/snapshots", json=SETTINGS))
        training_id = generated[0]["training_id"]
        assert generated[-1]["training"]["snapshot_count"] == 6
        archive = client.get(f"/api/rom/{training_id}/artifacts/snapshots")
        assert "attachment" in archive.headers["content-disposition"]
        with np.load(io.BytesIO(archive.content), allow_pickle=False) as data:
            assert data["displacements"].shape == (60, 6)
        built = events(client.post(f"/api/rom/{training_id}/basis", json={"energy_target": .999, "max_modes": 4}))
        assert built[-1]["training"]["basis"]["rank"] <= 4
        artifact = client.get(f"/api/rom/{training_id}/artifacts/basis")
        with np.load(io.BytesIO(artifact.content), allow_pickle=False) as model:
            basis = model["basis"]
            np.testing.assert_allclose(basis.T@basis, np.eye(basis.shape[1]), atol=1e-14)
        rebuilt = events(client.post(f"/api/rom/{training_id}/basis", json={"max_modes": 1}))
        assert rebuilt[-1]["training"]["basis"]["rank"] == 1
        assert not client.get("/api/health").json()["running"]
    with TestClient(create_app(training_root=tmp_path)) as client:
        records = client.get("/api/rom/trainings").json()["trainings"]
        assert len(records) == 1 and records[0]["id"] == training_id
        assert records[0]["basis"]["rank"] == 1
        metadata = client.get(f"/api/rom/{training_id}/artifacts/metadata").json()
        assert metadata == records[0]
        # Existing full-order optimisation still uses the same persistent worker.
        events(client.post("/api/run", json={"nx": 6, "ny": 2, "load_node_x": 6,
                                             "load_node_y": 1, "max_iterations": 2}))


def test_api_rejects_invalid_and_unknown_training_inputs(tmp_path):
    with TestClient(create_app(training_root=tmp_path)) as client:
        for extra in ({"load_heights": [1.2]}, {"volume_fractions": []}, {"snapshot_stride": 0},
                      {"load_angles": [0, 0]}, {"nx": True}, {"ny": 1.5},
                      {"iterations_per_case": 0},
                      {"root": "elsewhere"}):
            assert client.post("/api/rom/snapshots", json={**SETTINGS, **extra}).status_code == 422
        assert client.post("/api/rom/invalid/basis", json={}).status_code == 404
        assert client.get("/api/rom/invalid/artifacts/metadata").status_code == 404
        generated = events(client.post("/api/rom/snapshots", json=SETTINGS))
        training_id = generated[0]["training_id"]
        assert client.post(f"/api/rom/{training_id}/basis", json={"energy_target": 0}).status_code == 422
        assert client.post(f"/api/rom/{training_id}/basis", json={"max_modes": 0}).status_code == 422
        assert client.get(f"/api/rom/{training_id}/artifacts/basis").status_code == 404


def test_api_generates_more_than_24_cases_and_512_snapshots(tmp_path):
    settings=dict(nx=3,ny=2,load_heights=[.5],load_angles=list(range(-90,91,7))[:25],
        volume_fractions=[.4],iterations_per_case=20,snapshot_stride=1)
    with TestClient(create_app(training_root=tmp_path)) as client:
        generated=events(client.post('/api/rom/snapshots',json=settings))
        metadata=generated[-1]['training']
        assert metadata['case_count']==25 and metadata['snapshot_count']==525
        assert metadata['estimated_snapshot_bytes']==8*(18+6)*525
        archive=client.get(f'/api/rom/{metadata["id"]}/artifacts/snapshots')
        with np.load(io.BytesIO(archive.content),allow_pickle=False) as arrays:
            assert arrays['displacements'].shape==(18,525)
            assert np.max(arrays['residuals'])<1e-7
        built=events(client.post(f'/api/rom/{metadata["id"]}/basis',json={'max_modes':4}))
        assert built[-1]['training']['basis']['rank']>0


def test_large_training_requests_and_matching_run_geometry_are_accepted():
    request=SnapshotRequest(nx=240,ny=120,length=2001,height=.001,filter_radius=2000,
        projection_beta=100,load_angles=list(range(25)),iterations_per_case=1000,snapshot_stride=1)
    config=request.configuration()
    assert config.estimated_snapshot_bytes>128*1024**2
    assert RunRequest(nx=240,ny=120,length=2001,height=.001,load_node_x=240,load_node_y=60).configuration().nx==240


def test_experimental_model_routes_are_removed_but_rom_remains(tmp_path):
    with TestClient(create_app(training_root=tmp_path)) as client:
        page=client.get('/').text
        assert 'id="tab-model-lab"' not in page
        assert 'id="tab-rom-training"' in page and 'id="tab-cantilever"' in page
        assert 'id="tab-ml"' in page
        assert client.get('/api/ml/catalog').status_code==404
        assert client.get('/api/ml/models').status_code==404
        assert client.post('/api/ml/train',json={}).status_code==404
        assert client.get('/static/model-lab.js').status_code==404
        assert 'model_lab_supported' not in client.get('/api/health').json()
    manager=RunManager()
    with pytest.raises(ValueError,match='Unknown training task'):
        manager.start_task('ml_train',{})


def test_training_shares_worker_lock_and_stops_without_poisoning_next_job(tmp_path):
    manager = RunManager()
    try:
        job = manager.start_task("rom_snapshots", {"settings": {**SETTINGS, "iterations_per_case": 200,
                    "load_angles": [-90], "snapshot_stride": 2}, "root": str(tmp_path)})
        with pytest.raises(RuntimeError, match="already running"):
            manager.start(CantileverConfig(nx=6, ny=2))
        # Stop after a real snapshot, preserving nonempty reusable data.
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            try:
                event = job.messages.get(timeout=1)
            except queue.Empty:
                continue
            assert event["event"] != "error", event
            if event["event"] == "snapshot":
                assert manager.stop(job.id)
                break
        else:
            pytest.fail("Training did not produce a snapshot")
        while time.monotonic() < deadline:
            try:
                event = job.messages.get(timeout=1)
            except queue.Empty:
                continue
            assert event["event"] != "error", event
            if event["event"] == "complete":
                assert event["stop_reason"] == "cancelled"
                assert event["training"]["snapshot_count"] > 0
                break
        else:
            pytest.fail("Training did not cancel")
        pid = job.process.pid
        manager.finish(job)
        next_job = manager.start_task("rom_basis", {"training_id": job.id, "root": str(tmp_path), "options": {}})
        assert next_job.process.pid == pid
        while True:
            event = next_job.messages.get(timeout=40)
            assert event["event"] != "error", event
            if event["event"] == "complete":
                assert event["training"]["basis"]["rank"] >= 1
                break
        manager.finish(next_job)
    finally:
        manager.shutdown()


def test_interrupted_manifest_is_visible_but_cannot_build_a_basis(tmp_path):
    directory = tmp_path / ("a" * 32)
    directory.mkdir()
    metadata = {"id": directory.name, "status": "generating", "created_at": "2026-11-14T12:00:00+00:00",
                "snapshot_count": 0, "settings": SETTINGS}
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    # Malformed and unrelated files cannot break or enter the dataset library.
    invalid = tmp_path / ("b" * 32)
    invalid.mkdir()
    (invalid / "metadata.json").write_text("{", encoding="utf-8")
    with TestClient(create_app(training_root=tmp_path)) as client:
        records = client.get("/api/rom/trainings").json()["trainings"]
        assert len(records) == 1 and records[0]["status"] == "interrupted"
        assert client.post(f"/api/rom/{directory.name}/basis", json={}).status_code == 409
