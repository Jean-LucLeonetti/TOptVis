"""Fixed physical regions, reduced design variables, and consistent sensitivities."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from ml_opt.boundaries import MeshSelection
from ml_opt.config import CantileverConfig
from ml_opt.optimizers import MovingAsymptotes
from ml_opt.problem import CantileverProblem
from ml_opt.regions import NonDesignRegion, region_densities
from ml_opt.runner import optimise


REGIONS = [NonDesignRegion('solid',MeshSelection('surface',(0,0),(1/6,.5))),
           NonDesignRegion('void',MeshSelection('surface',(.5,0),(2/3,.25)))]


def configuration(**options):
    return CantileverConfig(nx=6,ny=4,filter_radius=.15,projection_beta=2,
        non_design_regions=REGIONS,**options)


def test_region_mapping_roundtrip_conflicts_and_volume_validation():
    config=configuration()
    assert CantileverConfig(**json.loads(json.dumps(asdict(config))))==config
    fixed=region_densities(config.non_design_regions,6,4)
    np.testing.assert_array_equal(np.flatnonzero(fixed==1),[0,6])
    np.testing.assert_array_equal(np.flatnonzero(fixed==0),[3])
    assert np.count_nonzero(np.isnan(fixed))==21
    with pytest.raises(ValueError,match='overlap'):
        replace(config,non_design_regions=[REGIONS[0],replace(REGIONS[0],kind='void')])
    with pytest.raises(ValueError,match='one element'):
        replace(config,nx=1,ny=1)
    with pytest.raises(ValueError,match='available for design'):
        replace(config,non_design_regions=[NonDesignRegion('solid',MeshSelection('surface',(0,0),(1,1)))])
    with pytest.raises(ValueError,match='exceed'):
        replace(config,volume_fraction=.05)
    with pytest.raises(ValueError,match='prevent'):
        replace(config,volume_fraction=.99,volume_constraint='gt')
    with pytest.raises(ValueError,match='surface'):
        NonDesignRegion('solid',MeshSelection('line',(0,0),(0,1)))


@pytest.mark.parametrize('robust',[False,True])
def test_fixed_cells_have_zero_derivatives_and_active_gradients_match_differences(robust):
    with CantileverProblem(configuration(robust=robust)) as problem:
        design=problem.initial_design()
        evaluation=problem.evaluate(design)
        mask=problem.design_mask
        assert np.mean(design)==pytest.approx(problem.config.volume_fraction)
        for field in (evaluation.field_densities or {'nominal':evaluation.density}).values():
            np.testing.assert_array_equal(field[~mask],problem.fixed_density[~mask])
        np.testing.assert_array_equal(evaluation.objective_gradient[~mask],0)
        np.testing.assert_array_equal(evaluation.constraint_gradient[~mask],0)
        for index in (0,3,10):
            # Fixed raw coordinates can be perturbed without changing physics.
            plus=design.copy();minus=design.copy()
            if mask[index]:
                plus[index]+=1e-6;minus[index]-=1e-6;denominator=2e-6
            else:
                plus[index]=.7;minus[index]=.3;denominator=.4
            upper,lower=problem.evaluate(plus,False),problem.evaluate(minus,False)
            assert evaluation.objective_gradient[index]==pytest.approx((upper.compliance-lower.compliance)/denominator,rel=4e-5,abs=1e-6)
            assert evaluation.constraint_gradient[index]==pytest.approx((upper.constraint-lower.constraint)/denominator,rel=4e-5,abs=1e-8)
        assert MovingAsymptotes(design,design_mask=mask).n==21


@pytest.mark.parametrize('method',['mma','gcmma'])
@pytest.mark.parametrize('robust',[False,True])
def test_every_iterate_keeps_regions_fixed_through_continuation(method,robust):
    config=configuration(optimizer=method,robust=robust,max_iterations=4,projection_schedule='geometric',
                         projection_beta_interval=2,projection_beta_max=4)
    samples=[]
    result=optimise(config,evaluation_callback=lambda row,design,state:samples.append((design,state)))
    fixed=region_densities(config.non_design_regions,6,4);mask=np.isfinite(fixed)
    assert len(samples)==5
    for design,state in samples:
        np.testing.assert_array_equal(design[mask],fixed[mask])
        for field in (state.field_densities or {'nominal':state.density}).values():
            np.testing.assert_array_equal(field[mask],fixed[mask])
        np.testing.assert_array_equal(state.objective_gradient[mask],0)
        assert state.residual<1e-7
    assert np.max(np.abs(result.design[~mask]-samples[0][0][~mask]))>1e-4


def test_parallel_robust_and_reduced_solves_preserve_regions():
    from ml_opt.rom.solver import GalerkinSolver
    config=configuration(robust=True)
    with CantileverProblem(config) as first, CantileverProblem(replace(config,field_execution='parallel')) as second:
        expected=first.evaluate(first.initial_design());actual=second.evaluate(second.initial_design())
        np.testing.assert_allclose(actual.objective_gradient,expected.objective_gradient,rtol=1e-10)
        assert actual.field_volumes==expected.field_volumes
    config=replace(config,robust=False)
    with CantileverProblem(config) as full:
        expected=full.evaluate(full.initial_design())
        basis=np.eye(len(full.mesh.free_dofs))
    with CantileverProblem(config,state_solver=GalerkinSolver(config,basis)) as reduced:
        actual=reduced.evaluate(reduced.initial_design())
        assert actual.compliance==pytest.approx(expected.compliance,rel=1e-9)
        np.testing.assert_allclose(actual.objective_gradient,expected.objective_gradient,rtol=1e-8)


@pytest.mark.parametrize('method',['mma','gcmma'])
def test_single_remaining_design_variable_uses_a_nondegenerate_subproblem(method):
    config=CantileverConfig(nx=4,ny=2,max_iterations=2,optimizer=method,
        non_design_regions=[NonDesignRegion('void',MeshSelection('surface',(0,0),(.75,1))),
                            NonDesignRegion('solid',MeshSelection('surface',(.75,0),(1,.5)))])
    result=optimise(config)
    assert np.all(np.isfinite(result.design))
    np.testing.assert_array_equal(result.design[[0,1,2,3,4,5,6]],[0,0,0,1,0,0,0])
