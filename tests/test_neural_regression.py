"""Neural fitting, independent splits, deterministic checkpoints and portable inference."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from ml_opt.config import CantileverConfig
from ml_opt.neural_regression import (NeuralSettings, fit_neural_models, neural_split,
                                     predict_network, read_neural_models)
from ml_opt.regression import fit_models
from ml_opt.topology_regression import fit_topology_models, evaluate_topology_prediction


def dataset(directory, *, count=20, fixed=False):
    config = CantileverConfig(nx=6, ny=2, filter_radius=.1, projection_beta=3,
        non_design_regions=[
            {"kind":"solid","selection":{"kind":"surface","start":[0,0],"end":[1/6,1]}},
            {"kind":"void","selection":{"kind":"surface","start":[5/6,0],"end":[1,1]}}
        ] if fixed else [])
    cases=[]
    for i,v in enumerate(np.linspace(.25,.65,count)):
        density=.1+v*np.linspace(.3,.8,12)
        if fixed:
            density[[0,6]]=1;density[[5,11]]=0
        cases.append({"index":i,"volume_fraction":float(v),"compliance":float(10-5*v+12*v*v),
            "density":density.tolist(),"converged":True,"feasible":True,"stop_reason":"design_and_objective_stable"})
    data={"format":"toptvis-regression-v1","id":directory.name,"status":"complete",
          "fixed_config":asdict(config),"cases":cases}
    (directory/'dataset.json').write_text(json.dumps(data))
    return data


@pytest.mark.parametrize("target", ["compliance", "topology"])
def test_neural_fit_checkpoint_reloads_and_matches_browser_inference(tmp_path,target):
    dataset(tmp_path,fixed=True)
    polynomial=fit_models(tmp_path) if target=="compliance" else fit_topology_models(tmp_path)
    artifact=tmp_path/("models.json" if target=="compliance" else "topology-models.json")
    previous=artifact.read_bytes()
    rng=torch.random.get_rng_state().clone();threads=torch.get_num_threads()
    events=[]
    report=fit_neural_models(tmp_path,target=target,epochs=700,emit=events.append)
    assert artifact.read_bytes()==previous
    assert torch.equal(torch.random.get_rng_state(),rng) and torch.get_num_threads()==threads
    assert read_neural_models(tmp_path,target)==report
    model=report['models']['neural_network']
    assert model['best_validation_loss']==min(row['validation_loss'] for row in model['training_history'])
    assert model['training_history'][-1]['epoch']==700
    assert set(report['train_indices']).isdisjoint(report['validation_indices'])
    assert set(report['test_indices']).isdisjoint(report['validation_indices']+report['train_indices'])
    assert report['test_indices']==polynomial['test_indices']
    assert 0 in report['train_indices'] and 19 in report['train_indices']
    assert model['test']['mae']<report['baseline']['test']['mae']
    assert events[0]['epoch']==0 and events[-1]['epoch']==700
    assert len(events)<=202 and all(event['event']=='neural_training' for event in events)
    values=[.2,.3,.5,.7]
    expected=predict_network(model,values)
    if target=='topology':
        assert expected.shape==(4,12)
        np.testing.assert_array_equal(expected[:,[0,6]],1)
        np.testing.assert_array_equal(expected[:,[5,11]],0)
        assert np.all((expected>=0)&(expected<=1))
        checked=evaluate_topology_prediction(tmp_path,model_key='neural_network',volume=.5)
        assert checked['model_key']=='neural_network' and checked['prediction']['residual']<1e-8
    path=tmp_path/f'neural-{target}.json'
    module=(Path(__file__).parents[1]/'src/ml_opt/web/static/regression-models.js').as_uri()
    script='''import {readFileSync} from "node:fs";
import {predictNeural} from MODULE;
const model=JSON.parse(readFileSync(process.argv[1],"utf8")).models.neural_network;
process.stdout.write(JSON.stringify([.2,.3,.5,.7].map(v=>predictNeural(model,v))));'''.replace('MODULE',json.dumps(module))
    browser=json.loads(subprocess.check_output(['node','--input-type=module','-e',script,str(path)],text=True))
    np.testing.assert_allclose(browser,expected,atol=1e-12,rtol=1e-12)
    repeated=fit_neural_models(tmp_path,target=target,epochs=700)
    assert repeated['models']['neural_network']['weights']==model['weights']
    saved=path.read_bytes()
    assert fit_neural_models(tmp_path,target=target,epochs=700,stop_requested=lambda:True) is None
    assert path.read_bytes()==saved


@pytest.mark.parametrize('target',['compliance','topology'])
def test_test_targets_never_change_weights_scaling_or_validation_choice(tmp_path,target):
    data=dataset(tmp_path)
    before=fit_neural_models(tmp_path,target=target,epochs=150)
    for i in before['test_indices']:
        data['cases'][i]['compliance']*=10
        data['cases'][i]['density']=[.99]*12
    (tmp_path/'dataset.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='does not match'):
        read_neural_models(tmp_path,target)
    after=fit_neural_models(tmp_path,target=target,epochs=150)
    a,b=before['models']['neural_network'],after['models']['neural_network']
    assert a['weights']==b['weights']
    assert a['input_offset']==b['input_offset'] and a['input_scale']==b['input_scale']
    assert a['target_offset']==b['target_offset'] and a['target_scale']==b['target_scale']
    assert a['best_epoch']==b['best_epoch']
    assert a['test']['mae']<b['test']['mae']
    assert before['baseline']['train']==after['baseline']['train']


def test_cancellation_during_epochs_preserves_previous_model(tmp_path):
    dataset(tmp_path)
    fit_neural_models(tmp_path,target='compliance',epochs=30)
    saved=(tmp_path/'neural-compliance.json').read_bytes()
    received=[]
    assert fit_neural_models(tmp_path,target='compliance',epochs=300,emit=received.append,
        stop_requested=lambda:any(row['epoch']>=10 for row in received)) is None
    assert (tmp_path/'neural-compliance.json').read_bytes()==saved
    assert received[-1]['epoch']<300


@pytest.mark.parametrize('target',['compliance','topology'])
def test_validation_targets_only_select_checkpoint_not_training_updates(tmp_path,target):
    data=dataset(tmp_path)
    before=fit_neural_models(tmp_path,target=target,epochs=100)
    for i in before['validation_indices']:
        data['cases'][i]['compliance']*=10
        data['cases'][i]['density']=[.99]*12
    (tmp_path/'dataset.json').write_text(json.dumps(data))
    after=fit_neural_models(tmp_path,target=target,epochs=100)
    a,b=before['models']['neural_network'],after['models']['neural_network']
    assert a['input_offset']==b['input_offset'] and a['input_scale']==b['input_scale']
    assert a['target_offset']==b['target_offset'] and a['target_scale']==b['target_scale']
    assert [row['train_loss'] for row in a['training_history'] if row['epoch'] in (0,100)]==[
        row['train_loss'] for row in b['training_history'] if row['epoch'] in (0,100)]
    assert a['best_validation_loss']!=b['best_validation_loss']


def test_split_reserves_validation_only_from_polynomial_training_cases(tmp_path):
    data=dataset(tmp_path,count=6)
    _,train,validation,test=neural_split(data)
    assert len(train)==3 and len(validation)==1 and len(test)==2
    assert 0 in train and 5 in train
    assert set(train)|set(validation)|set(test)==set(range(6))


@pytest.mark.parametrize('options',[{'neurons':0},{'neurons':True},{'epochs':0},{'epochs':1.5},
    {'learning_rate':0},{'learning_rate':float('nan')},{'weight_penalty':-1},
    {'validation_fraction':.01},{'network_seed':-1},{'network_seed':True}])
def test_invalid_network_settings(options):
    with pytest.raises(ValueError):NeuralSettings(**options)
