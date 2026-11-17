"""Diffusion physics, adjoint gradients, robust mapping and MMA constraints."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from ml_opt.config import CantileverConfig
from ml_opt.problem import CantileverProblem
from ml_opt.fem.connection import ConnectionDiffusion
from ml_opt.connectivity import connection_elements, connected_targets
from ml_opt.boundaries import MeshSelection
from ml_opt.runner import optimise


def config(**options):
    return CantileverConfig(nx=12,ny=4,connection_anchors=[{'kind':'line','start':[0,0],'end':[0,1]}],
        connection_targets=[{'kind':'point','start':[1,.5]}],**options)


def test_diffusion_detects_cuts_and_normalises_full_solid():
    cfg=config()
    with CantileverProblem(cfg) as problem:
        solid=problem.evaluate(np.ones(48),False)
        assert solid.connection_ratios['target-1']==pytest.approx(1)
        assert solid.connection_connected=={'target-1':True}
        assert solid.connection_constraints['target-1']==pytest.approx(1/cfg.connection_max_temperature-1)
        density=np.ones((4,12));density[:,6]=0
        cut=problem.evaluate(density.ravel(),False)
        assert cut.connection_ratios['target-1']>1000
        assert cut.connection_constraints['target-1']>0
        assert cut.connection_connected=={'target-1':False}
        density[:,6]=.05
        grey=problem.evaluate(density.ravel(),False)
        assert grey.connection_ratios['target-1']<cut.connection_ratios['target-1']
        assert grey.connection_connected=={'target-1':False}
        assert np.all(cut.connection_temperatures['target-1']>=-1e-7)


@pytest.mark.parametrize('robust',[False,True])
def test_diffusion_gradients_include_filter_projection_and_fixed_regions(robust):
    cfg=config(filter_radius=.08,projection_beta=3,robust=robust,
        non_design_regions=[{'kind':'solid','selection':{'kind':'surface','start':[0,0],'end':[1/12,1]}}])
    x=np.random.default_rng(4).uniform(.3,.8,48)
    with CantileverProblem(cfg) as problem:
        evaluation=problem.evaluate(x)
        gradient=evaluation.connection_gradients['target-1']
        assert np.max(np.abs(gradient[[0,12,24,36]]))==0
        for index in (2,17,31,47):
            plus=x.copy();minus=x.copy();plus[index]+=1e-6;minus[index]-=1e-6
            expected=(problem.evaluate(plus,False).connection_constraints['target-1']-
                problem.evaluate(minus,False).connection_constraints['target-1'])/2e-6
            assert gradient[index]==pytest.approx(expected,rel=2e-5,abs=1e-7)
        if robust:
            direct=replace(cfg,filter_radius=0,projection_beta=0,robust=False)
            with CantileverProblem(direct) as physical:
                e=physical.evaluate(evaluation.field_densities['eroded'],False)
            assert e.connection_ratios==pytest.approx(evaluation.connection_ratios)


def test_multiple_targets_have_independent_constraints_and_adjoint_states():
    cfg=replace(config(),connection_targets=[{'kind':'point','start':[1,0]},{'kind':'point','start':[1,1]}])
    with CantileverProblem(cfg) as problem:
        x=np.random.default_rng(2).uniform(.4,.8,48);e=problem.evaluate(x)
        assert len(e.connection_constraints)==2
        assert not np.allclose(e.connection_temperatures['target-1'],e.connection_temperatures['target-2'])
        for key in e.connection_constraints:
            plus=x.copy();minus=x.copy();plus[7]+=1e-6;minus[7]-=1e-6
            fd=(problem.evaluate(plus,False).connection_constraints[key]-problem.evaluate(minus,False).connection_constraints[key])/2e-6
            assert e.connection_gradients[key][7]==pytest.approx(fd,rel=2e-5,abs=1e-7)


@pytest.mark.parametrize('method',['mma','gcmma'])
def test_optimisation_enforces_connection_rows(method):
    cfg=config(optimizer=method,max_iterations=30,connection_max_temperature=20,filter_radius=.08,volume_fraction=.45)
    result=optimise(cfg)
    assert result.evaluation.connection_constraints['target-1']<=cfg.feasibility_tolerance
    assert result.history[-1].connection_constraints==result.evaluation.connection_constraints
    assert result.evaluation.constraint<=cfg.feasibility_tolerance


@pytest.mark.parametrize('change',[
    {'connection_targets':[]},{'connection_anchors':[]},
    {'connection_targets':[{'kind':'point','start':[0,.5]}]},
    {'connection_min_conductivity':0},{'connection_penalisation':True},{'connection_max_temperature':.5},
    {'non_design_regions':[{'kind':'void','selection':{'kind':'surface','start':[11/12,0],'end':[1,1]}}]}
])
def test_invalid_regions_and_parameters(change):
    with pytest.raises(ValueError):replace(config(),**change)


def test_mapping_and_geometric_check_use_faces_not_diagonal_contacts():
    region=MeshSelection('surface',[.25,.25],[.75,.75])
    np.testing.assert_array_equal(connection_elements(region,4,4),[5,6,9,10])
    x=np.eye(3).ravel()
    assert connected_targets(x,3,3,[0],[np.array([8])])==[False]
