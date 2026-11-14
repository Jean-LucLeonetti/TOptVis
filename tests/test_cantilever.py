"""Numerical checks of elasticity, the implicit gradient, and both optimisers."""

from dataclasses import replace

import numpy as np
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.fem.element import q4_stiffness
from ml_opt.fem.mesh import StructuredMesh
from ml_opt.optimizers import MovingAsymptotes
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise


def test_q4_rigid_modes_and_energy():
    stiffness = q4_stiffness(0.7, 0.4, 0.3, 1.0)
    coordinates = np.array([[0, 0], [0.7, 0], [0.7, 0.4], [0, 0.4]])
    modes = np.column_stack((np.tile([1, 0], 4), np.tile([0, 1], 4),
                             np.column_stack((-coordinates[:, 1], coordinates[:, 0])).ravel()))
    np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-14)
    np.testing.assert_allclose(stiffness @ modes, 0, atol=1e-14)
    assert np.count_nonzero(np.linalg.eigvalsh(stiffness) > 1e-10) == 5


@pytest.mark.parametrize("ny", [3, 4])
def test_cantilever_support_and_load(ny):
    mesh = StructuredMesh(6, ny, 3.0, 1.0)
    force = mesh.cantilever_force(2.0).reshape(-1, 2)
    np.testing.assert_allclose(force.sum(axis=0), [0, -2])
    assert np.all(mesh.coordinates[mesh.fixed_dofs // 2, 0] == 0)
    active = np.flatnonzero(force[:, 1])
    assert np.all(mesh.coordinates[active, 0] == mesh.length)
    np.testing.assert_allclose(-force[:, 1] @ mesh.coordinates[:, 1] / 2, mesh.height / 2)


def test_petsc_solution_against_independent_dense_assembly():
    problem = CantileverProblem(CantileverConfig(nx=4, ny=2))
    mesh, solver = problem.mesh, problem.solver
    moduli = np.linspace(0.2, 1.0, mesh.element_count)
    state = solver.solve(moduli)
    stiffness = np.zeros((mesh.dof_count, mesh.dof_count))
    for modulus, dofs in zip(moduli, mesh.element_dofs):
        stiffness[np.ix_(dofs, dofs)] += modulus * solver.element_stiffness
    expected = np.linalg.solve(stiffness[np.ix_(mesh.free_dofs, mesh.free_dofs)],
                               solver.force[mesh.free_dofs])
    np.testing.assert_allclose(state.displacement[mesh.free_dofs], expected, rtol=1e-10)
    np.testing.assert_array_equal(state.displacement[mesh.fixed_dofs], 0)
    np.testing.assert_allclose(state.compliance, moduli @ state.element_energy, rtol=1e-11)
    assert state.relative_residual < 1e-10


def test_torch_sensitivities_against_central_differences_without_filtering():
    problem = CantileverProblem(CantileverConfig(nx=6, ny=3))
    rng = np.random.default_rng(7)
    design = rng.uniform(0.3, 0.8, problem.mesh.element_count)
    evaluation = problem.evaluate(design)
    np.testing.assert_array_equal(evaluation.density, design)
    np.testing.assert_allclose(evaluation.volume_fraction, design.mean())
    for index in (0, 5, 8, 17):
        perturbation = np.zeros_like(design)
        perturbation[index] = 1e-5
        plus = problem.evaluate(design + perturbation, False)
        minus = problem.evaluate(design - perturbation, False)
        dc = (plus.compliance - minus.compliance) / 2e-5
        dg = (plus.constraint - minus.constraint) / 2e-5
        np.testing.assert_allclose(evaluation.objective_gradient[index], dc, rtol=2e-5)
        np.testing.assert_allclose(evaluation.constraint_gradient[index], dg, rtol=1e-8)


def test_petsc_reuses_symbolic_factors_but_updates_numeric_factors():
    from petsc4py import PETSc
    PETSc.Log.begin()
    events = {name: PETSc.Log.Event(name) for name in ("MatLUFactorSym", "MatLUFactorNum")}
    before = {name: event.getPerfInfo()["count"] for name, event in events.items()}
    with CantileverProblem(CantileverConfig(nx=6, ny=3)) as problem:
        mesh, solver = problem.mesh, problem.solver
        for moduli in (np.linspace(.05, 1, mesh.element_count),
                       np.linspace(1, .01, mesh.element_count), np.full(mesh.element_count, .3)):
            state = solver.solve(moduli)
            stiffness = np.zeros((mesh.dof_count, mesh.dof_count))
            for modulus, dofs in zip(moduli, mesh.element_dofs):
                stiffness[np.ix_(dofs, dofs)] += modulus * solver.element_stiffness
            expected = np.linalg.solve(stiffness[np.ix_(mesh.free_dofs, mesh.free_dofs)],
                                       solver.force[mesh.free_dofs])
            np.testing.assert_allclose(state.displacement[mesh.free_dofs], expected, rtol=1e-10)
        assert solver.solve_count == 3
    assert events["MatLUFactorSym"].getPerfInfo()["count"] - before["MatLUFactorSym"] == 1
    assert events["MatLUFactorNum"].getPerfInfo()["count"] - before["MatLUFactorNum"] == 3
    problem.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.solve(moduli)


def test_runner_releases_native_objects_when_callback_fails(monkeypatch):
    from ml_opt.fem.elasticity import ElasticitySolver
    closed = []
    original = ElasticitySolver.close
    def close(solver):
        if not solver._closed:
            closed.append(solver)
        original(solver)
    monkeypatch.setattr(ElasticitySolver, "close", close)
    def broken_callback(entry):
        raise RuntimeError("callback failed")
    with pytest.raises(RuntimeError, match="callback failed"):
        optimise(CantileverConfig(nx=4, ny=2), callback=broken_callback)
    assert len(closed) == 1 and closed[0]._matrix is None


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_optimisers_reduce_compliance_and_respect_volume(method):
    config = CantileverConfig(nx=18, ny=6, optimizer=method, max_iterations=35)
    result = optimise(config)
    assert result.evaluation.compliance < 0.65 * result.history[0].compliance
    assert result.evaluation.constraint <= config.feasibility_tolerance
    assert np.all((result.design >= 0) & (result.design <= 1))
    np.testing.assert_array_equal(result.evaluation.density, result.design)
    assert result.evaluation.residual < 1e-7
    assert max(entry.design_change for entry in result.history) <= config.move_limit + 1e-10
    if method == "gcmma":
        assert result.history[-1].state_solves > len(result.history)


def test_gcmma_never_accepts_a_nonconservative_trial():
    problem = CantileverProblem(CantileverConfig(nx=4, ny=2))
    design = problem.initial_design()
    current = problem.evaluate(design)
    optimizer = MovingAsymptotes(design, "gcmma", max_inner_iterations=1)
    def nonconservative_trial(design, sensitivities):
        return replace(current, compliance=1e20, constraint=1e10)
    with pytest.raises(RuntimeError, match="conservative"):
        optimizer.step(design, current, nonconservative_trial)


@pytest.mark.parametrize("options", [{"volume_fraction": 0}, {"move_limit": -0.1},
                                     {"minimum_modulus": 2}, {"nx": 0}])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        CantileverConfig(**options)
