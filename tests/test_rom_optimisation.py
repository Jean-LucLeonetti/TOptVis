"""Online ROM solver selection, genuine reduced updates, and basis provenance."""

from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.fem.elasticity import ElasticitySolver
from ml_opt.fem.mesh import StructuredMesh
from ml_opt.output import save_results
from ml_opt.problem import CantileverProblem
from ml_opt.rom.solver import load_basis
from ml_opt.runner import optimise


def save_basis(path, config, basis):
    mesh = StructuredMesh(config.nx, config.ny, config.length, config.height)
    metadata = {"schema_version": 1, "id": "a"*32, "created_at": "2026-11-14T12:00:00+00:00",
        "status": "basis_ready", "cases": [asdict(replace(config, robust=False, state_solver="fom",
                                                           rom_basis_path=None, rom_basis_sha256=None))],
        "settings": {key: getattr(config, key) for key in ("nx", "ny", "length", "height")},
        "basis": {"rank": basis.shape[1], "built_at": "2026-11-14T12:00:00+00:00"}}
    np.savez_compressed(path, basis=basis, free_dofs=mesh.free_dofs, metadata_json=np.array(json.dumps(metadata)))
    return replace(config, state_solver="rom", rom_basis_path=str(path))


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_rom_full_basis_matches_fom_optimisation_without_full_elasticity_solves(tmp_path, monkeypatch, method):
    config = CantileverConfig(nx=6, ny=2, max_iterations=4, optimizer=method, filter_radius=.12, projection_beta=3)
    expected = optimise(config)
    free = len(StructuredMesh(config.nx, config.ny, config.length, config.height).free_dofs)
    reduced = save_basis(tmp_path / "basis.npz", config, np.eye(free))
    def forbidden(*args, **kwargs):
        raise AssertionError("ROM optimisation must not call the full-order elasticity solver")
    monkeypatch.setattr(ElasticitySolver, "solve", forbidden)
    result = optimise(reduced)
    np.testing.assert_allclose(result.design, expected.design, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(result.evaluation.compliance, expected.evaluation.compliance, rtol=1e-8)
    assert all(row.state_solver == "rom" and row.rom_basis_rank == free for row in result.history)
    assert result.config.rom_basis_sha256 == hashlib.sha256((tmp_path / "basis.npz").read_bytes()).hexdigest()
    assert result.rom_basis["basis_sha256"] == result.config.rom_basis_sha256


def test_low_rank_rom_drives_design_updates_and_saves_model_identity(tmp_path, monkeypatch):
    config = CantileverConfig(nx=6, ny=3, max_iterations=5, filter_radius=.1, projection_beta=2)
    with CantileverProblem(config) as full:
        state = full.evaluate(full.initial_design())
        vector = state.displacement[full.mesh.free_dofs]
    rom = save_basis(tmp_path / "basis.npz", config, (vector/np.linalg.norm(vector))[:, None])
    monkeypatch.setattr(ElasticitySolver, "solve", lambda *args: pytest.fail("Unexpected full-order state solve"))
    result = optimise(rom)
    assert result.rom_basis["rank"] == 1
    assert result.history[-1].state_solves == 6
    assert not np.allclose(result.design, config.volume_fraction)
    assert result.evaluation.compliance < result.history[0].compliance
    monkeypatch.setattr("ml_opt.output._save_figures", lambda *args: None)
    save_results(result, tmp_path / "result")
    summary = json.loads((tmp_path / "result/summary.json").read_text())
    assert summary["state_solver"] == "rom" and summary["rom_basis"] == result.rom_basis
    with np.load(tmp_path / "result/result.npz", allow_pickle=False) as archive:
        assert str(archive["state_solver"]) == "rom"
        assert json.loads(str(archive["rom_basis_json"])) == result.rom_basis
    assert "state_solver" in (tmp_path / "result/history.csv").read_text().splitlines()[0]


def test_robust_rom_parallel_and_sequential_fields_match_without_fom_fallback(tmp_path):
    base = CantileverConfig(nx=6, ny=3, filter_radius=.1, projection_beta=3, max_iterations=2)
    with CantileverProblem(base) as full:
        evaluation = full.evaluate(full.initial_design())
        vector = evaluation.displacement[full.mesh.free_dofs]
    config = save_basis(tmp_path / "basis.npz", replace(base, robust=True), (vector/np.linalg.norm(vector))[:, None])
    design = np.linspace(.25, .75, 18)
    with CantileverProblem(config) as sequential:
        expected = sequential.evaluate(design)
        assert sequential.solve_count == 3
        assert expected.residual > 1e-5
    with CantileverProblem(replace(config, field_execution="parallel")) as parallel:
        actual = parallel.evaluate(design)
        assert parallel.solve_count == 3
        assert actual.residual > 1e-5
        for field in ("eroded", "nominal", "dilated"):
            np.testing.assert_allclose(actual.field_compliances[field], expected.field_compliances[field], rtol=1e-12)
            np.testing.assert_allclose(actual.field_gradients[field], expected.field_gradients[field], rtol=1e-12)


@pytest.mark.parametrize("key,value", [("nx", 8), ("ny", 4), ("length", 4), ("height", 2),
                                      ("poisson_ratio", .2), ("thickness", 2)])
def test_incompatible_basis_rejected(tmp_path, key, value):
    base = CantileverConfig(nx=6, ny=2)
    rom = save_basis(tmp_path / "basis.npz", base, np.eye(36))
    with pytest.raises(ValueError, match=key):
        load_basis(rom.rom_basis_path, replace(rom, **{key: value}))


def test_changed_or_corrupt_basis_is_rejected(tmp_path):
    base = CantileverConfig(nx=6, ny=2)
    path = tmp_path / "basis.npz"
    rom = save_basis(path, base, np.eye(36))
    _, information = load_basis(path, rom)
    path.write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="changed"):
        optimise(replace(rom, rom_basis_sha256=information["basis_sha256"]))
    with pytest.raises(ValueError):
        load_basis(path, rom)


def test_basis_orthogonal_to_the_load_cannot_drive_optimisation(tmp_path):
    base = CantileverConfig(nx=6, ny=2)
    rom = save_basis(tmp_path / "basis.npz", base, np.eye(36)[:, :1])
    with pytest.raises(ValueError, match="cannot represent this load"):
        optimise(rom)


@pytest.mark.parametrize("options", [{"state_solver": "unknown"}, {"state_solver": "rom"},
    {"rom_basis_path": "basis.npz"}, {"state_solver": "rom", "rom_basis_path": "basis.npz", "rom_basis_sha256": "bad"}])
def test_invalid_solver_configuration(options):
    with pytest.raises(ValueError):
        CantileverConfig(**options)
