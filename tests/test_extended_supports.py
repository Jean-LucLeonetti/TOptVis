"""Directional constraints, imposed motion, and independent-load sensitivities."""

from dataclasses import replace
import json
import numpy as np
import pytest

from ml_opt.boundaries import BoundaryCondition as BC, MeshSelection as Selection, LoadCase
from ml_opt.config import CantileverConfig
from ml_opt.fem.mesh import StructuredMesh
from ml_opt.fem.state_solver import create_state_solver
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise
from ml_opt.output import save_results


def solver(config):
    return create_state_solver(config)


def clamp():
    return BC('fixed',Selection('line',(0,0),(0,1)))


def force(case='case-1', magnitude=1, angle=-90):
    return BC('force',Selection('point',(1,1)),magnitude,angle,case_id=case)


def check_gradient(config):
    state_solver=solver(config)
    try:
        moduli=np.linspace(.3,1,config.nx*config.ny)
        state=state_solver.solve(moduli)
        for index in range(len(moduli)):
            delta=np.zeros_like(moduli); delta[index]=1e-6
            difference=(state_solver.solve(moduli+delta).compliance-state_solver.solve(moduli-delta).compliance)/2e-6
            np.testing.assert_allclose(-state.element_energy[index],difference,rtol=3e-5,atol=1e-6)
        return state
    finally:
        state_solver.close()


@pytest.mark.parametrize('linear_solver', ['lu', 'cholesky', 'cg-icc', 'cg-gamg'])
def test_pin_and_roller_with_odd_free_dof_count(linear_solver):
    config=CantileverConfig(nx=3,ny=2,linear_solver=linear_solver,boundary_conditions=[
        BC('fixed',Selection('point',(0,0))), BC('restrain_y',Selection('point',(1,0))),force()])
    mesh=StructuredMesh(3,2,3,1,config.boundary_conditions)
    assert len(mesh.free_dofs)%2==1
    state=check_gradient(config)
    np.testing.assert_array_equal(state.displacement[mesh.fixed_dofs],0)
    assert state.relative_residual<1e-7


@pytest.mark.parametrize('applied_force',[False,True])
def test_nonzero_prescribed_displacement_has_consistent_adjoint(applied_force):
    conditions=[clamp(),BC('displacement',Selection('line',(1,0),(1,1)),components='x',ux=.03)]
    if applied_force: conditions.append(force())
    config=CantileverConfig(nx=3,ny=2,boundary_conditions=conditions)
    state=check_gradient(config)
    right_nodes=np.arange(3,12,4)
    np.testing.assert_allclose(state.displacement[2*right_nodes],.03)
    assert state.compliance>0


@pytest.mark.parametrize('aggregation',['weighted','worst'])
def test_load_cases_are_independent_and_gradients_follow_aggregation(aggregation):
    cases=(LoadCase('a','Downward',1),LoadCase('b','Upward',3))
    conditions=[clamp(),force('a',1,-90),force('b',2,90)]
    config=CantileverConfig(nx=3,ny=2,boundary_conditions=conditions,load_cases=cases,load_aggregation=aggregation)
    state=check_gradient(config)
    assert state.case_compliances['a']>0
    np.testing.assert_allclose(state.case_compliances['b'],4*state.case_compliances['a'],rtol=1e-10)
    expected=(state.case_compliances['a']+3*state.case_compliances['b'])/4 if aggregation=='weighted' else state.case_compliances['b']
    np.testing.assert_allclose(state.compliance,expected)
    assert state.active_case=='b' and state.state_solves==2


def test_conflicting_components_and_unrestrained_rotation_are_rejected():
    with pytest.raises(ValueError,match='Conflicting'):
        CantileverConfig(nx=3,ny=2,boundary_conditions=[clamp(),BC('displacement',Selection('point',(0,0)),components='x',ux=.1),force()])
    with pytest.raises(ValueError,match='rigid motion'):
        CantileverConfig(nx=3,ny=2,boundary_conditions=[BC('restrain_x',Selection('line',(0,0),(0,1))),force()])


def test_unknown_and_unloaded_cases_are_rejected():
    with pytest.raises(ValueError,match='unknown load'):
        CantileverConfig(nx=3,ny=2,boundary_conditions=[clamp(),force('missing')])
    with pytest.raises(ValueError,match='Second'):
        CantileverConfig(nx=3,ny=2,boundary_conditions=[clamp(),force()],load_cases=[LoadCase(),LoadCase('second','Second')])


def test_custom_load_cases_and_motion_match_across_robust_worker_processes():
    config = CantileverConfig(nx=3, ny=2, robust=True, projection_beta=2,
        boundary_conditions=[clamp(), force('a'), force('b',2,90),
            BC('displacement',Selection('line',(1,0),(1,1)),components='x',ux=.03)],
        load_cases=[LoadCase('a','Down'),LoadCase('b','Up',3)])
    design = np.linspace(.3,.7,6)
    with CantileverProblem(config) as sequential, CantileverProblem(replace(config,field_execution='parallel')) as parallel:
        expected, actual = sequential.evaluate(design), parallel.evaluate(design)
        assert actual.field_compliances == pytest.approx(expected.field_compliances)
        assert actual.load_case_compliances == pytest.approx(expected.load_case_compliances)
        np.testing.assert_allclose(actual.objective_gradient, expected.objective_gradient, rtol=1e-10)
        np.testing.assert_allclose(actual.constraint_gradient, expected.constraint_gradient, rtol=1e-10)
        assert parallel.solve_count == sequential.solve_count == 12  # State and adjoint for two cases, three fields.
        delta = np.zeros(6); delta[2] = 1e-6
        difference = (parallel.evaluate(design+delta,False).compliance-parallel.evaluate(design-delta,False).compliance)/2e-6
        assert actual.objective_gradient[2] == pytest.approx(difference,rel=3e-5)


@pytest.mark.parametrize('motion', [False, True])
def test_native_optimisation_exports_prescribed_motion_and_all_load_cases(tmp_path, motion):
    conditions = [clamp(),force('a'),force('b',2,90)]
    if motion:
        conditions.append(BC('displacement',Selection('line',(1,0),(1,1)),components='x',ux=.03))
    config = CantileverConfig(nx=3,ny=2,max_iterations=2,boundary_conditions=conditions,
        load_cases=[LoadCase('a','Down'),LoadCase('b','Up',3)])
    result = optimise(config)
    output = save_results(result,tmp_path)
    assert (output/'density.png').stat().st_size > 0
    summary = json.loads((output/'summary.json').read_text())
    assert summary['objective_quantity'] == ('elastic_work' if motion else 'compliance')
    assert summary['load_case_compliances'] == result.evaluation.load_case_compliances
    assert all(set(row.load_case_compliances)=={'a','b'} for row in result.history)
    with np.load(output/'result.npz',allow_pickle=False) as archive:
        assert archive['forces'].shape == (2,12,2)
        np.testing.assert_allclose(archive['forces'].sum(axis=1),[[0,-1],[0,2]],atol=1e-14)
        np.testing.assert_allclose(archive['prescribed_displacement'][[3,7,11],0],.03 if motion else 0)
