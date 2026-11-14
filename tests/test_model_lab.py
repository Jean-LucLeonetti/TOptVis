"""Every family actually trains; gradients, correction, splits and provenance."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from ml_opt.config import CantileverConfig
from ml_opt.fem.elasticity import ElasticitySolver
from ml_opt.problem import CantileverProblem
from ml_opt.rom.training import SnapshotConfig,generate_snapshots,read_metadata
from ml_opt.rom.validation import ValidationConfig
from ml_opt.model_lab.config import FAMILIES,POD_FAMILIES,TrainOptions
from ml_opt.model_lab.data import file_hash,load_dataset,split_cases
from ml_opt.model_lab.experiments import run_experiment,validate_model
from ml_opt.model_lab.inference import ModelProblem,Predictor
from ml_opt.model_lab.training import train_model


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    root=tmp_path_factory.mktemp('model-lab')
    source=root/'source'
    generate_snapshots(SnapshotConfig(nx=6,ny=3,load_heights=(.25,.75),load_angles=(-90.,-45.),
        volume_fractions=(.4,),iterations_per_case=3,snapshot_stride=1),source)
    models={}
    for family in sorted(FAMILIES):
        directory=root/family
        metadata=train_model(source,directory,TrainOptions(family=family,epochs=3,batch_size=4,width=8,
            layers=1,max_modes=3,energy_loss_weight=.01))
        models[family]=(directory,metadata)
    return root,source,models


def config():
    return CantileverConfig(nx=6,ny=3,load_node_x=6,load_node_y=2,load_angle=-65,
                            filter_radius=.1,projection_beta=2)


@pytest.mark.parametrize('family',sorted(FAMILIES))
def test_each_family_trains_persists_and_predicts_with_exact_clamp(fitted,family):
    _,source,models=fitted; directory,metadata=models[family]
    assert metadata['status']=='trained' and metadata['checkpoint_ready']
    assert metadata['history'] and metadata['source_sha256']==file_hash(source/'snapshots.npz')
    assert set(metadata['split']['train_case_indices']).isdisjoint(metadata['split']['validation_case_indices'])
    predictor=Predictor(directory,config())
    try:
        state=predictor.raw_state(np.full(18,.1))
        assert np.isfinite(state.compliance) and np.isfinite(state.relative_residual)
        np.testing.assert_array_equal(state.displacement[predictor.mesh.fixed_dofs],0)
        assert state.displacement.shape==(56,)
    finally: predictor.close()


@pytest.mark.parametrize('family',sorted(FAMILIES-POD_FAMILIES))
def test_raw_neural_gradient_is_derivative_of_its_own_objective(fitted,family):
    _,_,models=fitted; directory,_=models[family]
    settings=config()
    with ModelProblem(settings,Predictor(directory,settings)) as problem:
        design=np.random.default_rng(5).uniform(.35,.65,18)
        value=problem.evaluate(design)
        for index in (1,9,17):
            delta=np.zeros(18);delta[index]=.002
            difference=(problem.evaluate(design+delta,False).compliance-problem.evaluate(design-delta,False).compliance)/.004
            np.testing.assert_allclose(value.objective_gradient[index],difference,rtol=.03,atol=.01)
        assert problem.solver.fom_solves==0


@pytest.mark.parametrize('family',sorted(FAMILIES))
def test_fem_correction_matches_full_displacement_compliance_and_gradient(fitted,family):
    _,_,models=fitted; directory,_=models[family]; settings=config()
    with CantileverProblem(settings) as full, ModelProblem(settings,Predictor(directory,settings,mode='corrected')) as corrected:
        design=np.linspace(.3,.7,18)
        expected=full.evaluate(design); actual=corrected.evaluate(design)
        for key in ('displacement','compliance','objective_gradient','constraint_gradient'):
            np.testing.assert_allclose(getattr(actual,key),getattr(expected,key),rtol=2e-7,atol=2e-7)
        assert actual.residual<1e-7
        assert corrected.solver.stats()['predictions']==1 and corrected.solver.fom_solves==1


def test_adaptive_basis_changes_only_at_accepted_states(fitted):
    _,_,models=fitted; directory,_=models['adaptive_pod']; settings=config()
    predictor=Predictor(directory,settings)
    try:
        original=predictor.bases[0].copy()
        moduli=np.linspace(.01,.4,18)
        predictor.solve(moduli); reference=predictor.solve(moduli)
        assert predictor.fom_solves==2
        np.testing.assert_array_equal(predictor.bases[0],original)
        predictor.accept()
        assert predictor.basis_updates==1 and predictor.bases[0].shape[1]==original.shape[1]+1
        np.testing.assert_allclose(predictor.raw_state(moduli).displacement,reference.displacement,rtol=2e-8,atol=1e-7)
    finally: predictor.close()


def test_splits_group_duplicate_cases_and_never_split_a_trajectory():
    first=config(); second=replace(first,load_angle=-30)
    cases=[first,replace(first,load=2),second]
    indices=np.array([0,0,1,1,2,2])
    train,validation,split=split_cases(cases,indices,.25,7)
    assert ((0 in train)==(2 in train)) and set(train).isdisjoint(validation)
    assert split['independent_case_count']==2
    with pytest.raises(ValueError,match='independent cases'):
        split_cases([first],np.array([0,0]),.25,7)


def test_unseen_validation_and_experiment_save_real_fom_checks(fitted,tmp_path):
    _,source,models=fitted; directory,metadata=models['cnn']
    before=file_hash(directory/'model.pt')
    options=ValidationConfig(load_heights=(.5,),load_angles=(-65.,),volume_fractions=(.35,),iterations_per_case=2,comparison_stride=1)
    report=validate_model(directory,source,options)
    assert report['summaries']['surrogate']['solver_statistics']['fom_solves']==0
    assert report['summaries']['corrected']['meets_tolerances']
    assert file_hash(directory/'model.pt')==before
    with np.load(directory/'validations'/report['id']/'arrays.npz',allow_pickle=False) as arrays:
        assert arrays['designs'].shape==(18,6)
        assert np.max(np.abs(arrays['full_displacements'][:,arrays['modes']=='corrected']-
            arrays['model_displacements'][:,arrays['modes']=='corrected']))<1e-6
    experiment=run_experiment(directory,dict(mode='corrected',max_iterations=2))
    assert experiment['iterations']==2 and experiment['final_fom_check']['passed']
    assert experiment['solver_statistics']['fom_solves']==3
    assert all(row['state_solver']=='cnn/corrected' for row in experiment['history'])


def test_cancellation_preserves_checkpoint_and_bad_geometry_is_rejected(fitted,tmp_path):
    _,source,_=fitted; events=[]; directory=tmp_path/'cancelled'
    metadata=train_model(source,directory,TrainOptions(family='cnn',width=8,layers=1,epochs=10),
        emit=events.append,stop_requested=lambda:any(e['event']=='ml_epoch' for e in events))
    assert metadata['status']=='cancelled' and metadata['checkpoint_ready'] and len(metadata['history'])==1
    with pytest.raises(ValueError,match='nx must match'):
        Predictor(directory,replace(config(),nx=7))
    path=directory/'model.pt';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):
        Predictor(directory,config())


def test_correction_lu_fallback_is_checked_and_counted(fitted,monkeypatch):
    _,_,models=fitted; directory,_=models['cnn']; settings=config()
    original=ElasticitySolver.solve
    def fail_iterative(solver,moduli):
        if solver.linear_solver=='cg-gamg': raise RuntimeError('Synthetic iterative failure')
        return original(solver,moduli)
    monkeypatch.setattr(ElasticitySolver,'solve',fail_iterative)
    with ModelProblem(settings,Predictor(directory,settings,mode='corrected')) as problem:
        evaluation=problem.evaluate(problem.initial_design())
        assert evaluation.residual<1e-7 and problem.solver.lu_fallbacks==1
        assert problem.solver.corrector.linear_solver=='lu'
        problem.evaluate(np.full(18,.45))
        assert problem.solver.lu_fallbacks==1 and problem.solver.fom_solves==2


@pytest.mark.parametrize('family',['fno','adaptive_pod'])
def test_gcmma_experiment_preserves_consistency_with_state_cache(fitted,family):
    _,_,models=fitted; directory,_=models[family]
    result=run_experiment(directory,dict(mode='corrected',optimizer='gcmma',max_iterations=2))
    assert result['final_fom_check']['passed']
    assert all(row['state_solver']==f'{family}/corrected' for row in result['history'])


@pytest.mark.parametrize('options',[dict(family='unknown'),dict(width=7),dict(family='transformer',width=10),
    dict(epochs=0),dict(validation_fraction=0),dict(learning_rate=float('nan'))])
def test_invalid_training_options(options):
    with pytest.raises(ValueError): TrainOptions(**options)
