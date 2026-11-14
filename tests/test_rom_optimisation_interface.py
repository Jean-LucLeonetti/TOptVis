"""The app runs a chosen saved ROM and rejects incompatible selections upfront."""

import hashlib
import json

from fastapi.testclient import TestClient
import numpy as np

from ml_opt.web.app import create_app


def events(response):
    assert response.status_code == 200, response.text
    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert rows[-1]["event"] == "complete", rows[-1]
    return rows


def test_saved_rom_runs_both_optimisers_and_robust_fields_and_fom_remains_available(tmp_path):
    with TestClient(create_app(training_root=tmp_path)) as client:
        assert client.get("/api/health").json()["rom_optimisation_supported"]
        assert 'id="state_solver"' in client.get("/").text
        assert client.get("/static/rom-choice.js").status_code == 200
        snapshots = {"nx": 6, "ny": 4, "load_heights": [.5], "load_angles": [-90],
                     "volume_fractions": [.4], "iterations_per_case": 4, "snapshot_stride": 1}
        generated = events(client.post("/api/rom/snapshots", json=snapshots))
        training_id = generated[0]["training_id"]
        base = {"nx": 6, "ny": 4, "load_node_x": 6, "load_node_y": 2, "load_angle": -60,
                "max_iterations": 2, "filter_radius": .1, "projection_beta": 3}
        selection = {**base, "state_solver": "rom", "rom_training_id": training_id}
        assert client.post("/api/run", json=selection).status_code == 409
        built = events(client.post(f"/api/rom/{training_id}/basis", json={"max_modes": 2}))[-1]["training"]
        digest = hashlib.sha256((tmp_path / training_id / "basis.npz").read_bytes()).hexdigest()
        nominal_rom_compliance = None
        for robust, method, execution in ((False, "mma", "sequential"), (False, "gcmma", "sequential"),
            (True, "gcmma", "sequential"), (True, "mma", "parallel")):
            run = events(client.post("/api/run", json={**selection, "robust": robust,
                                "optimizer": method, "field_execution": execution}))
            assert run[0]["config"]["rom_basis_sha256"] == digest
            assert run[-1]["rom_basis"]["rank"] == built["basis"]["rank"]
            frames = [row for row in run if row["event"] == "iteration"]
            assert [row["history"]["iteration"] for row in frames] == [0, 1, 2]
            assert all(row["history"]["state_solver"] == "rom" for row in frames)
            if not robust:
                nominal_rom_compliance = frames[0]["history"]["compliance"]
            assert frames[-1]["history"]["linear_residual"] > 1e-7
            if robust:
                assert set(frames[-1]["field_densities"]) == {"eroded", "nominal", "dilated"}
        fom = events(client.post("/api/run", json=base))
        assert fom[-1]["state_solver"] == "fom" and fom[-1]["rom_basis"] is None
        assert fom[1]["history"]["compliance"] >= nominal_rom_compliance * (1-1e-9)
        assert not client.get("/api/health").json()["running"]
        for invalid, code in (({**selection, "nx": 8}, 422), ({**selection, "length": 4}, 422),
                              ({**base, "state_solver": "rom"}, 422),
                              ({**selection, "rom_training_id": "invalid"}, 422),
                              ({**selection, "rom_training_id": "f"*32}, 404),
                              ({**base, "rom_training_id": training_id}, 422),
                              ({**selection, "rom_basis_path": "elsewhere.npz"}, 422)):
            assert client.post("/api/run", json=invalid).status_code == code
        assert not client.get("/api/health").json()["running"]
