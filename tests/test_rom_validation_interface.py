"""Held-out validation streams and saved report lifecycle through the real worker."""

import io
import json

from fastapi.testclient import TestClient
import numpy as np

from ml_opt.web.app import create_app


SETTINGS = {"nx": 6, "ny": 4, "load_heights": [.5], "load_angles": [-90],
            "volume_fractions": [.4], "iterations_per_case": 3, "snapshot_stride": 1}
VALIDATION = {"load_heights": [.25, .75], "load_angles": [-60], "volume_fractions": [.35],
              "iterations_per_case": 3, "comparison_stride": 2}


def collect(response):
    assert response.status_code == 200, response.text
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[0]["event"] == "started"
    assert events[-1]["event"] == "complete", events[-1]
    return events


def test_api_unseen_validation_download_reload_and_invalidation(tmp_path):
    with TestClient(create_app(training_root=tmp_path)) as client:
        assert client.get("/api/health").json()["rom_validation_supported"]
        assert 'id="rom-validation-form"' in client.get("/").text
        generated = collect(client.post("/api/rom/snapshots", json=SETTINGS))
        training_id = generated[0]["training_id"]
        prefix = f"/api/rom/{training_id}"
        assert client.post(prefix+"/validate", json=VALIDATION).status_code == 409
        collect(client.post(prefix+"/basis", json={}))
        overlap = client.post(prefix+"/validate", json={**VALIDATION, "load_heights": [.5],
                             "load_angles": [-90], "volume_fractions": [.4]})
        assert overlap.status_code == 422 and "unseen" in overlap.json()["detail"]
        assert not client.get("/api/health").json()["running"]
        for invalid in ({"gradient_tolerance": 0}, {"load_heights": []},
                        {"iterations_per_case": 300, "comparison_stride": 1}, {"root": "elsewhere"}):
            assert client.post(prefix+"/validate", json={**VALIDATION, **invalid}).status_code == 422
        events = collect(client.post(prefix+"/validate", json=VALIDATION))
        assert sum(event["event"] == "validation_state" for event in events) == 6
        report = client.get(prefix+"/validation").json()
        assert report["state_count"] == 6 and report["status"] == "complete"
        assert report == client.get(prefix+"/artifacts/validation").json()
        artifact = client.get(prefix+"/artifacts/comparisons")
        assert "attachment" in artifact.headers["content-disposition"]
        with np.load(io.BytesIO(artifact.content), allow_pickle=False) as archive:
            assert archive["full_gradients"].shape == (24, 6)
            assert archive["rom_displacements"].shape == (60, 6)
            assert json.loads(str(archive["metadata_json"])) == report
    with TestClient(create_app(training_root=tmp_path)) as client:
        assert client.get(prefix+"/validation").json() == report
        record = client.get("/api/rom/trainings").json()["trainings"][0]
        assert record["validation"]["id"] == report["id"]
        collect(client.post(prefix+"/basis", json={"max_modes": 1}))
        assert client.get(prefix+"/validation").status_code == 404
        assert client.get(prefix+"/artifacts/comparisons").status_code == 404
        assert "validation" not in client.get("/api/rom/trainings").json()["trainings"][0]
