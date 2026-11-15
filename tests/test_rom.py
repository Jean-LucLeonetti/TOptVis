"""Physical snapshot provenance, POD truncation, and portable training artifacts."""

from dataclasses import replace
import json

import numpy as np
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.fem.mesh import StructuredMesh
from ml_opt.problem import CantileverProblem
from ml_opt.rom.training import SnapshotConfig, build_basis, generate_snapshots, pod_basis, read_metadata
from ml_opt.runner import optimise


def small_config(**options):
    return SnapshotConfig(nx=6, ny=4, load_heights=(.5,), load_angles=(-90, 0),
                          volume_fractions=(.4,), iterations_per_case=3, snapshot_stride=2, **options)


def test_snapshots_are_real_filtered_projected_equilibrium_states(tmp_path):
    config = small_config()
    events = []
    directory = tmp_path / "dataset"
    metadata = generate_snapshots(config, directory, emit=events.append)
    assert metadata["snapshot_count"] == 6
    assert metadata["completed_cases"] == 2
    assert metadata["status"] == "snapshots_ready"
    with np.load(directory / "snapshots.npz", allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata_json"])) == metadata
        mesh = StructuredMesh(6, 4, 3, 1)
        np.testing.assert_array_equal(archive["free_dofs"], mesh.free_dofs)
        assert archive["displacements"].shape == (len(mesh.free_dofs), 6)
        np.testing.assert_array_equal(archive["iterations"], [0, 2, 3, 0, 2, 3])
        assert np.max(archive["residuals"]) < 1e-7
        for column in range(6):
            case = config.cases()[archive["case_indices"][column]]
            with CantileverProblem(case) as problem:
                rho = archive["densities"][:, column]
                moduli = case.minimum_modulus + (case.young_modulus-case.minimum_modulus)*rho**case.penalisation
                full = problem.solver.solve(moduli)
                np.testing.assert_allclose(full.displacement[mesh.free_dofs], archive["displacements"][:, column], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(full.compliance, archive["compliances"][column], rtol=1e-10)
        expected = optimise(config.cases()[0])
        np.testing.assert_allclose(archive["densities"][:, 2], expected.evaluation.density)
    assert sum(event["event"] == "snapshot" for event in events) == 6


def test_state_callback_copies_cannot_change_optimisation():
    config = CantileverConfig(nx=6, ny=2, max_iterations=3)
    def corrupt(row, displacement, density):
        displacement[:] = -123
        density[:] = -123
    expected = optimise(config)
    actual = optimise(config, state_snapshot_callback=corrupt)
    np.testing.assert_array_equal(actual.design, expected.design)
    np.testing.assert_array_equal(actual.evaluation.displacement, expected.evaluation.displacement)


def test_stop_keeps_partial_snapshots_and_can_build_basis(tmp_path):
    events = []
    directory = tmp_path / "partial"
    metadata = generate_snapshots(small_config(), directory, emit=events.append,
                                  stop_requested=lambda: any(e["event"] == "snapshot" for e in events))
    assert metadata["status"] == "cancelled"
    assert metadata["snapshot_count"] == 1
    assert metadata["completed_cases"] == 0
    result = build_basis(directory)
    assert result["basis"]["rank"] == 1
    assert result["basis"]["max_training_error"] < 1e-12


def test_stop_before_first_case_does_not_make_empty_model(tmp_path):
    directory = tmp_path / "empty"
    metadata = generate_snapshots(small_config(), directory, stop_requested=lambda: True)
    assert metadata["snapshot_count"] == 0
    assert not (directory / "snapshots.npz").exists()
    with pytest.raises(ValueError, match="nonempty"):
        build_basis(directory)


def test_pod_energy_rank_cap_and_normalisation():
    snapshots = np.diag([10., 2., 1.])
    basis, _, errors, summary = pod_basis(snapshots, energy_target=.99, max_modes=1, normalize=False)
    assert summary["rank"] == 1 and not summary["target_met"]
    np.testing.assert_allclose(summary["captured_energy"], 100/105)
    np.testing.assert_allclose(errors, [0, 1, 1])
    np.testing.assert_allclose(basis.T @ basis, np.eye(1), atol=1e-14)
    _, _, _, equal = pod_basis(snapshots, energy_target=.99, max_modes=1, normalize=True)
    np.testing.assert_allclose(equal["captured_energy"], 1/3)
    full, _, _, all_modes = pod_basis(snapshots, energy_target=1, max_modes=256, normalize=False)
    assert all_modes["rank"] == 3 and all_modes["target_met"]
    np.testing.assert_allclose(full @ full.T @ snapshots, snapshots, atol=1e-14)


def test_pod_rank_deficiency_and_training_error_match_direct_projection():
    rng = np.random.default_rng(17)
    left, _ = np.linalg.qr(rng.normal(size=(23, 4)))
    right = rng.normal(size=(4, 12))
    snapshots = left @ right
    basis, norms, errors, summary = pod_basis(snapshots, energy_target=1)
    assert summary["rank"] == summary["numerical_rank"] == 4
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(errors, np.linalg.norm(snapshots-basis@(basis.T@snapshots), axis=0)/norms, atol=1e-15)
    assert summary["max_training_error"] < 1e-13


def test_basis_roundtrip_and_cancelled_rebuild_preserve_previous_model(tmp_path):
    directory = tmp_path / "trained"
    generate_snapshots(small_config(), directory)
    result = build_basis(directory, energy_target=.9999, max_modes=4)
    with np.load(directory / "basis.npz", allow_pickle=False) as model:
        basis = model["basis"]
        assert json.loads(str(model["metadata_json"])) == result
        mesh = StructuredMesh(6, 4, 3, 1)
        lifted = np.zeros((mesh.dof_count, basis.shape[1]))
        lifted[model["free_dofs"]] = basis
        np.testing.assert_array_equal(lifted[mesh.fixed_dofs], 0)
        np.testing.assert_allclose(basis.T@basis, np.eye(basis.shape[1]), atol=1e-14)
    original = (directory / "basis.npz").read_bytes()
    events = []
    assert build_basis(directory, emit=events.append, stop_requested=lambda: bool(events)) is None
    assert read_metadata(directory) == result
    assert (directory / "basis.npz").read_bytes() == original
    rebuilt = build_basis(directory, max_modes=1)
    assert rebuilt["basis"]["rank"] == 1


def test_stop_during_basis_save_does_not_publish_a_partial_rebuild(tmp_path, monkeypatch):
    directory = tmp_path / "saved"
    generate_snapshots(small_config(), directory)
    original_metadata = build_basis(directory)
    original_bytes = (directory / "basis.npz").read_bytes()
    stopped = False
    save = np.savez_compressed
    def stop_after_save(*args, **kwargs):
        nonlocal stopped
        save(*args, **kwargs)
        stopped = True
    monkeypatch.setattr(np, "savez_compressed", stop_after_save)
    assert build_basis(directory, max_modes=1, stop_requested=lambda: stopped) is None
    assert (directory / "basis.npz").read_bytes() == original_bytes
    assert read_metadata(directory) == original_metadata
    assert not (directory / "basis.tmp").exists()


@pytest.mark.parametrize("options", [dict(load_heights=()), dict(load_heights=(.1, .11)),
    dict(load_angles=(float("nan"),)), dict(volume_fractions=(0,)),
    dict(load_angles=(0, 0)), dict(snapshot_stride=0), dict(iterations_per_case=0),
    dict(nx=0), dict(ny=True), dict(nx=2.5), dict(length=0), dict(height=float('inf')),
    dict(filter_radius=-.1), dict(projection_beta=-1), dict(projection_eta=1)])
def test_invalid_snapshot_requests(options):
    with pytest.raises(ValueError):
        replace(SnapshotConfig(), **options)


def test_snapshot_workload_has_no_fixed_resource_caps(monkeypatch):
    # This intentionally exceeds all former ceilings, without allocating the
    # estimated arrays or enumerating every case just to validate its settings.
    monkeypatch.setattr(SnapshotConfig,'cases',lambda *_: pytest.fail('Validation must not enumerate cases'))
    config=SnapshotConfig(nx=240,ny=120,load_angles=tuple(range(25)),
        iterations_per_case=1000,snapshot_stride=1)
    assert config.case_count==225
    assert config.maximum_snapshots==225225
    assert config.estimated_snapshot_bytes==8*(2*240*121+240*120)*225225
    assert config.estimated_snapshot_bytes>128*1024**2
    assert replace(config,snapshot_stride=1001).maximum_snapshots==450


def test_minimal_mesh_and_periodic_load_angles_are_valid():
    config=SnapshotConfig(nx=1,ny=1,load_heights=(.5,),load_angles=(270.,),
        volume_fractions=(.4,),length=.001,height=2001.,filter_radius=2000.,projection_beta=100.)
    assert config.cases()[0].load_angle==270.
    assert config.cases()[0].load_node_y==1


@pytest.mark.parametrize("snapshots", [np.zeros((3, 2)), np.ones((0, 2)),
                                        np.array([[1, np.nan]]), np.ones(3)])
def test_invalid_snapshot_matrix(snapshots):
    with pytest.raises(ValueError):
        pod_basis(snapshots)
