"""Real streamed regression jobs and compatibility with the other workspaces."""

import json
import queue
import time
from dataclasses import asdict

from fastapi.testclient import TestClient
import numpy as np
import pytest

from ml_opt.web.app import create_app
from ml_opt.web.app import RunRequest
from ml_opt.web.jobs import RunManager
from ml_opt.runner import optimise


SETTINGS = {"nx": 6, "ny": 2, "case_count": 6, "max_iterations": 3, "projection_beta": 0}


def events(response):
    assert response.status_code == 200, response.text
    result = [json.loads(line) for line in response.text.splitlines() if line]
    assert result[0]["event"] == "started"
    assert result[-1]["event"] == "complete", result[-1]
    return result


def test_generate_fit_download_reload_and_existing_routes(tmp_path):
    root = tmp_path / "regression"
    with TestClient(create_app(training_root=tmp_path / "rom", regression_root=root)) as client:
        page = client.get("/").text
        assert page.index('id="tab-regression"') < page.index('id="tab-rom-training"')
        assert 'id="workspace-rom-training"' in page
        assert client.get("/static/regression.js").status_code == 200
        assert client.get("/api/health").json()["regression_supported"]
        assert client.get("/api/regression/datasets").json() == {"datasets": []}
        generated = events(client.post("/api/regression/generate", json=SETTINGS))
        dataset_id = generated[0]["dataset_id"]
        assert sum(event["event"] == "regression_case" for event in generated) == 6
        endpoint = f"/api/regression/{dataset_id}"
        data = client.get(endpoint).json()
        assert len(data["dataset"]["cases"]) == 6 and data["models"] is None
        assert client.post(endpoint + "/fit", json={}).status_code == 422
        fitted = events(client.post(endpoint + "/fit", json={"include_unconverged": True}))
        assert fitted[0]["dataset_id"] == dataset_id
        report = client.get(endpoint).json()["models"]
        assert report["unfinished_indices"] == list(range(6))
        assert set(report["models"]) == {"linear", "quadratic"}
        saved = client.get(endpoint + "/artifacts/dataset")
        assert saved.json() == data["dataset"]
        assert "attachment" in saved.headers["content-disposition"]
        assert client.get(endpoint + "/artifacts/models").json() == report
        assert client.post(endpoint + "/topology/evaluate", json={"degree": 1, "volume": .4}).status_code == 422
        assert client.post(endpoint + "/topology/fit", json={}).status_code == 422
        topology_options = {"include_unconverged": True, "degrees": [1, 2, 3]}
        events(client.post(endpoint + "/topology/fit", json=topology_options))
        topology_report = client.get(endpoint).json()["topology_models"]
        assert topology_report["test_indices"] == report["test_indices"]
        assert topology_report["train_indices"] == report["train_indices"]
        assert topology_report["element_count"] == 12
        assert set(topology_report["models"]) == {"1", "2", "3"}
        assert client.get(endpoint + "/artifacts/topology-models").json() == topology_report
        assert client.get(endpoint).json()["models"] == report
        volume = data["dataset"]["cases"][topology_report["test_indices"][0]]["volume_fraction"]
        checked = events(client.post(endpoint + "/topology/evaluate", json={"degree": 2, "volume": volume}))
        evaluation = next(event["evaluation"] for event in checked if event["event"] == "topology_evaluation")
        assert evaluation["prediction"]["objective"] > 0
        assert evaluation["reference"]["objective"] > 0
        assert evaluation["prediction"]["residual"] < 1e-8
        assert client.post(endpoint + "/topology/evaluate", json={"degree": 10, "volume": volume}).status_code == 422
        assert client.post(endpoint + "/topology/evaluate", json={"degree": True, "volume": volume}).status_code == 422
        assert client.post(endpoint + "/topology/evaluate", json={"degree": 2, "volume": 0}).status_code == 422
        records = client.get("/api/regression/datasets").json()["datasets"]
        assert records[0]["has_models"] and records[0]["completed_cases"] == 6
        assert "cases" not in records[0]
        assert client.get("/api/rom/trainings").json() == {"trainings": []}
        assert not client.get("/api/health").json()["running"]
        for degrees in ([], [0], [11], [True], ["3"], [2.5], [1, 1], [4]):
            rejected = client.post(endpoint + "/fit", json={"include_unconverged": True, "degrees": degrees})
            assert rejected.status_code == 422, rejected.text
            assert client.get(endpoint).json()["models"] == report
            assert not client.get("/api/health").json()["running"]
        events(client.post(endpoint + "/fit", json={"include_unconverged": True, "degrees": [1, 2, 3]}))
        report = client.get(endpoint).json()["models"]
        assert report["degrees"] == [1, 2, 3]
        assert set(report["models"]) == {"linear", "quadratic", "polynomial_3"}
        assert client.get(endpoint + "/artifacts/models").json() == report
    with TestClient(create_app(training_root=tmp_path / "rom", regression_root=root)) as client:
        assert client.get(endpoint).json()["models"] == report
        assert client.get(endpoint).json()["topology_models"]["element_count"] == 12
        assert client.get(endpoint).json()["dataset"]["settings"]["nx"] == 6


def test_uncapped_ridge_fit_requests_and_high_order_structural_check(tmp_path):
    with TestClient(create_app(regression_root=tmp_path / "regression")) as client:
        generated = events(client.post("/api/regression/generate", json=SETTINGS))
        endpoint = f"/api/regression/{generated[0]['dataset_id']}"
        options = {"include_unconverged": True, "degrees": list(range(1, 13)),
                   "method": "ridge", "ridge_alpha": .1}
        for suffix, key in (("/fit", "models"), ("/topology/fit", "topology_models")):
            for invalid in ({"method": "other"}, {"ridge_alpha": -1}, {"ridge_alpha": True}):
                assert client.post(endpoint + suffix, json={**options, **invalid}).status_code == 422
            events(client.post(endpoint + suffix, json=options))
            report = client.get(endpoint).json()[key]
            assert report["method"] == "ridge" and report["ridge_alpha"] == .1
            assert report["degrees"] == list(range(1, 13)) and len(report["models"]) == 12
        checked = events(client.post(endpoint + "/topology/evaluate", json={"degree": 12, "volume": .4}))
        result = next(event["evaluation"] for event in checked if event["event"] == "topology_evaluation")
        assert result["degree"] == 12 and result["prediction"]["residual"] < 1e-8


def test_rejects_invalid_requests_and_unknown_files(tmp_path):
    # No worker is needed for request validation and missing-file checks.
    client = TestClient(create_app(regression_root=tmp_path))
    for options in ({"case_count": 5}, {"volume_min": .8}, {"nx": True}, {"unexpected": "value"}):
        assert client.post("/api/regression/generate", json={**SETTINGS, **options}).status_code == 422
    assert client.get("/api/regression/not-a-dataset").status_code == 404
    assert client.post("/api/regression/" + "0"*32 + "/fit", json={}).status_code == 404
    assert client.get("/api/regression/" + "0"*32 + "/artifacts/dataset").status_code == 404


def test_both_neural_targets_stream_save_and_preserve_polynomial_models(tmp_path):
    root=tmp_path/"regression"
    with TestClient(create_app(regression_root=root)) as client:
        generated=events(client.post("/api/regression/generate",json=SETTINGS))
        endpoint=f"/api/regression/{generated[0]['dataset_id']}"
        options={"include_unconverged":True,"method":"neural","neurons":4,"epochs":30,
                 "learning_rate":.01,"weight_penalty":.001,"network_seed":9,"validation_fraction":.25}
        assert client.post(endpoint+"/topology/evaluate",json={"model_key":"neural_network","volume":.4}).status_code==422
        for suffix,target,key in (("/fit","compliance","models"),("/topology/fit","topology","topology_models")):
            events(client.post(endpoint+suffix,json={"include_unconverged":True}))
            polynomial=client.get(endpoint).json()[key]
            for invalid in ({"neurons":0},{"epochs":True},{"learning_rate":0},{"validation_fraction":.01}):
                assert client.post(endpoint+suffix,json={**options,**invalid}).status_code==422
            streamed=events(client.post(endpoint+suffix,json=options))
            progress=[event for event in streamed if event["event"]=="neural_training"]
            assert progress[0]["epoch"]==0 and progress[-1]["epoch"]==30
            data=client.get(endpoint).json();neural=data["neural_models"][target]
            assert data[key]==polynomial
            assert neural["test_indices"]==polynomial["test_indices"]
            assert neural["validation_indices"] and neural["models"]["neural_network"]["weights"]
            assert client.get(endpoint+f"/artifacts/neural-{target}").json()==neural
        checked=events(client.post(endpoint+"/topology/evaluate",json={"model_key":"neural_network","volume":.4}))
        result=next(event["evaluation"] for event in checked if event["event"]=="topology_evaluation")
        assert result["model_key"]=="neural_network" and result["prediction"]["residual"]<1e-8
        assert client.post(endpoint+"/topology/evaluate",json={"model_key":"neural_network","degree":2,"volume":.4}).status_code==422
        saved=client.get(endpoint).json()["neural_models"]
    with TestClient(create_app(regression_root=root)) as client:
        assert client.get(endpoint).json()["neural_models"]==saved


def test_regression_shares_worker_lock_and_preserves_cancelled_cases(tmp_path):
    manager = RunManager()
    try:
        job = manager.start_task("regression_generate", {"root": str(tmp_path),
            "settings": {**SETTINGS, "max_iterations": 200}})
        with pytest.raises(RuntimeError, match="already running"):
            manager.start_task("rom_basis", {})
        deadline = time.monotonic() + 45
        received = []
        while time.monotonic() < deadline:
            try:
                event = job.messages.get(timeout=.5)
            except queue.Empty:
                continue
            received.append(event)
            assert event["event"] != "error", event
            if event["event"] == "regression_iteration":
                assert manager.stop(job.id)
            if event["event"] == "complete":
                break
        else:
            pytest.fail("Regression job did not stop")
        assert received[-1]["stop_reason"] == "cancelled"
        manager.finish(job)
        saved = json.loads((tmp_path / job.id / "dataset.json").read_text())
        assert saved["status"] == "cancelled" and saved["cases"]
        assert manager.active is None
    finally:
        manager.shutdown()


@pytest.mark.parametrize("motion", [False, True])
def test_saved_setup_generation_preserves_boundaries_and_reloads(tmp_path, motion):
    conditions = [
        {"kind": "fixed", "selection": {"kind": "line", "start": [0, 0], "end": [0, 1]}},
        {"kind": "force", "selection": {"kind": "line", "start": [1, 0], "end": [1, 1]},
         "magnitude": 2, "angle": -60, "case_id": "down"},
        {"kind": "force", "selection": {"kind": "point", "start": [1, 1]},
         "magnitude": 3, "angle": 45, "case_id": "up"},
    ]
    if motion:
        conditions.append({"kind": "displacement", "selection": {"kind": "line", "start": [1, 0], "end": [1, 1]},
                           "components": "x", "ux": .03})
    problem = {"nx": 6, "ny": 2, "max_iterations": 2, "filter_radius": .05,
               "projection_beta": 1, "projection_schedule": "geometric", "projection_beta_interval": 1,
               "boundary_conditions": conditions, "load_aggregation": "worst",
               "load_cases": [{"id": "down", "name": "Down", "weight": 1}, {"id": "up", "name": "Up", "weight": 2}]}
    with TestClient(create_app(regression_root=tmp_path)) as client:
        assert client.get("/api/health").json()["regression_problem_import_supported"]
        validated = client.post("/api/project/validate", json=problem)
        assert validated.status_code == 200, validated.text
        source = validated.json()["model"]
        result = events(client.post("/api/regression/generate", json={"case_count": 6, "problem": source,
                                                                     "problem_name": "Saved cantilever.json"}))
        endpoint = f'/api/regression/{result[0]["dataset_id"]}'
        dataset = client.get(endpoint).json()["dataset"]
        assert dataset["settings"]["source_problem"] == source
        assert dataset["settings"]["problem_name"] == "Saved cantilever.json"
        assert dataset["objective_quantity"] == ("elastic_work" if motion else "compliance")
        started = [row for row in result if row["event"] == "regression_case_started"]
        for row, case in zip(started, dataset["cases"], strict=True):
            expected = RunRequest(**{**source, "volume_fraction": case["volume_fraction"]}).configuration()
            assert row["config"] == json.loads(json.dumps(asdict(expected)))
        reference = optimise(RunRequest(**{**source, "volume_fraction": .25}).configuration())
        np.testing.assert_allclose(dataset["cases"][0]["density"], reference.evaluation.density)
        assert dataset["cases"][0]["compliance"] == pytest.approx(reference.evaluation.compliance)
        assert set(dataset["cases"][0]["load_case_compliances"]) == {"down", "up"}
    with TestClient(create_app(regression_root=tmp_path)) as client:
        assert client.get(endpoint).json()["dataset"] == dataset


@pytest.mark.parametrize("problem, code", [
    ({"nx": 6, "ny": 2}, 422),  # Missing load coordinates cannot be silently remapped.
    ({"boundary_conditions": []}, 422),
    ({"rom_basis_path": "/tmp/untrusted.npz"}, 422),
    ({"state_solver": "rom", "rom_training_id": "a" * 32}, 404),
    ({"state_solver": "rom"}, 422),
])
def test_bad_import_does_not_start_solver(tmp_path, problem, code):
    app = create_app(training_root=tmp_path / "rom", regression_root=tmp_path / "regression")
    with TestClient(app) as client:
        response = client.post("/api/regression/generate", json={"problem": problem, "case_count": 6})
        assert response.status_code == code, response.text
        assert app.state.manager.active is None
        assert not (tmp_path / "regression").exists()


def test_saved_rom_setup_is_resolved_and_fingerprinted(tmp_path):
    with TestClient(create_app(training_root=tmp_path / "rom", regression_root=tmp_path / "regression")) as client:
        snapshots = events(client.post("/api/rom/snapshots", json={"nx": 6, "ny": 2,
            "load_heights": [.5], "load_angles": [-90], "volume_fractions": [.4],
            "iterations_per_case": 2, "snapshot_stride": 1}))
        training_id = snapshots[0]["training_id"]
        events(client.post(f"/api/rom/{training_id}/basis", json={"max_modes": 2}))
        problem = {"nx": 6, "ny": 2, "load_node_x": 6, "load_node_y": 1, "max_iterations": 1,
                   "state_solver": "rom", "rom_training_id": training_id}
        generated = events(client.post("/api/regression/generate", json={"case_count": 6, "problem": problem}))
        dataset = client.get(f'/api/regression/{generated[0]["dataset_id"]}').json()["dataset"]
        fixed = dataset["fixed_config"]
        assert fixed["state_solver"] == "rom" and len(fixed["rom_basis_sha256"]) == 64
        assert fixed["rom_basis_path"] == str(tmp_path / "rom" / training_id / "basis.npz")
        assert all("compliance" in case for case in dataset["cases"])
        assert dataset["settings"]["source_problem"]["rom_training_id"] == training_id
        assert "rom_basis_path" not in dataset["settings"]["source_problem"]
        response = client.post("/api/regression/generate", json={"problem": {**problem, "length": 4}})
        assert response.status_code == 422
