"""Affine ROM assembly matches an independent full sparse Galerkin reference."""

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from ml_opt.config import CantileverConfig
from ml_opt.fem.state_solver import create_state_solver
from ml_opt.rom import operators as operator_module
from ml_opt.rom.operators import prepare_element_operators
from ml_opt.rom.solver import GalerkinSolver, load_basis
from ml_opt.rom.training import SnapshotConfig, build_basis, generate_snapshots, read_metadata


@pytest.mark.parametrize("rank", [1, 7, 18])
def test_affine_solution_energy_and_matrix_free_residual_match_full_assembly(rank):
    config = CantileverConfig(nx=7, ny=3, length=4.2, height=.8,
        poisson_ratio=.23, thickness=1.7, load_angle=-63, load_node_x=5, load_node_y=2)
    rng = np.random.default_rng(29)
    basis, _ = np.linalg.qr(rng.normal(size=(56, rank)))
    solver = GalerkinSolver(config, basis)
    mesh = solver.mesh
    lifted = np.zeros((mesh.dof_count, rank)); lifted[mesh.free_dofs] = basis
    rows = np.repeat(mesh.element_dofs, 8, axis=1).ravel()
    columns = np.tile(mesh.element_dofs, (1, 8)).ravel()
    # Repeated solves vary the material over four orders of magnitude.
    for _ in range(3):
        moduli = 10**rng.uniform(-4, 0, mesh.element_count)
        matrix = coo_matrix(((moduli[:,None,None]*solver.element_stiffness).ravel(),
                             (rows, columns)), shape=(mesh.dof_count, mesh.dof_count)).tocsr()
        reduced = lifted.T @ (matrix @ lifted)
        expected_u = lifted @ np.linalg.solve(reduced, lifted.T @ solver.force)
        expected_energy = np.einsum("ei,ij,ej->e", expected_u[mesh.element_dofs],
                                    solver.element_stiffness, expected_u[mesh.element_dofs])
        expected_residual = np.linalg.norm((matrix @ expected_u - solver.force)[mesh.free_dofs]) / np.linalg.norm(solver.rhs)
        actual = solver.solve(moduli)
        np.testing.assert_allclose(actual.displacement, expected_u, rtol=2e-11, atol=1e-11)
        np.testing.assert_allclose(actual.element_energy, expected_energy, rtol=2e-11, atol=1e-11)
        np.testing.assert_allclose(actual.compliance, solver.force @ expected_u, rtol=2e-11)
        np.testing.assert_allclose(actual.relative_residual, expected_residual, rtol=2e-11)
        assert not hasattr(solver, "rows") and not hasattr(solver, "columns")
    assert solver.solve_count == 3


def trained(directory):
    generate_snapshots(SnapshotConfig(nx=6, ny=3, load_heights=(.5,),
        volume_fractions=(.4,), iterations_per_case=3, snapshot_stride=1), directory)
    return build_basis(directory, energy_target=1, max_modes=4)


def test_training_persists_operators_and_loading_does_not_rebuild(tmp_path, monkeypatch):
    directory = tmp_path/'trained'; metadata = trained(directory)
    config = replace(CantileverConfig(**metadata['cases'][0]), state_solver='rom',
                     rom_basis_path=str(directory/'basis.npz'))
    basis, information, operators = load_basis(config.rom_basis_path, config, include_operators=True)
    np.testing.assert_allclose(operators, prepare_element_operators(config, basis), atol=1e-14)
    assert information['reduced_assembly'] == 'saved element operators'
    assert metadata['basis']['operator_bytes'] == operators.nbytes
    def forbidden(*args, **kwargs):
        pytest.fail('Saved operators must be used without repeating offline projection')
    monkeypatch.setattr('ml_opt.rom.solver.prepare_element_operators', forbidden)
    solver = create_state_solver(config)
    assert np.isfinite(solver.solve(np.ones(18)).compliance)


@pytest.mark.parametrize('damage', ['shape', 'nonfinite', 'missing', 'format'])
def test_malformed_operator_artifact_is_rejected(tmp_path, damage):
    directory = tmp_path/'trained'; metadata = trained(directory)
    path = directory/'basis.npz'
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    if damage == 'shape': arrays['reduced_element_stiffness'] = np.ones((1, 1))
    if damage == 'nonfinite': arrays['reduced_element_stiffness'][0, 0] = np.nan
    if damage == 'missing': arrays.pop('reduced_element_stiffness')
    if damage == 'format':
        metadata['basis']['operator_format'] = 'unknown'
        arrays['metadata_json'] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='operator'):
        load_basis(path, CantileverConfig(**metadata['cases'][0]), include_operators=True)


def test_operator_budget_and_cancellation_preserve_previous_basis(tmp_path, monkeypatch):
    directory = tmp_path/'trained'; metadata = trained(directory)
    original = (directory/'basis.npz').read_bytes()
    events = []
    result = build_basis(directory, emit=events.append,
        stop_requested=lambda: any(e['event']=='operators_started' for e in events))
    assert result is None
    assert read_metadata(directory) == metadata
    assert (directory/'basis.npz').read_bytes() == original
    monkeypatch.setattr(operator_module, 'MAX_OPERATOR_BYTES', 1)
    with pytest.raises(ValueError, match='fewer basis modes'):
        build_basis(directory)
    assert read_metadata(directory) == metadata
    assert (directory/'basis.npz').read_bytes() == original
