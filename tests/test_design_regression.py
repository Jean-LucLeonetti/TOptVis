"""Full density features, independent families and actual structural labels."""

from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from ml_opt.config import CantileverConfig
from ml_opt.design_regression import (DesignSettings, analysis_config, generate_designs, evaluate_design,
    split_design_cases, fit_design_polynomial, fit_design_models, predict_design, read_design_models, validate_density)


def source(directory, *, count=8, fixed=True):
    directory.mkdir(exist_ok=True)
    config=CantileverConfig(nx=6,ny=2,projection_beta=3,filter_radius=.15,
        non_design_regions=[{"kind":"solid","selection":{"kind":"surface","start":[0,0],"end":[1/6,1]}}] if fixed else [])
    rng=np.random.default_rng(9)
    cases=[]
    for i in range(count):
        density=rng.uniform(.2,.9,12)
        if fixed:density[[0,6]]=1
        cases.append({"index":i,"density":density.tolist(),"compliance":-999})
    data={"format":"toptvis-regression-v1","id":directory.name,"status":"complete","fixed_config":asdict(config),"cases":cases}
    (directory/'dataset.json').write_text(json.dumps(data))
    return data


def synthetic(directory):
    data=source(directory,count=12)
    data.update(format="toptvis-design-regression-v1",status="complete")
    for case in data["cases"]:
        case.update(status="complete",family=str(case["index"]),variant=0,
            compliance=float(np.exp(2+np.array(case["density"])@np.linspace(-.3,.7,12))))
    # Multiple designs in each family, not independent random rows.
    for base in list(data["cases"]):
        density=np.array(base["density"]);density[1]+=.01
        data["cases"].append({**base,"index":len(data["cases"]),"variant":1,"density":density.tolist(),
            "compliance":float(np.exp(2+density@np.linspace(-.3,.7,12)))})
    (directory/'dataset.json').write_text(json.dumps(data))
    return data


@pytest.mark.parametrize("alpha",[0.,.07])
@pytest.mark.parametrize("degree",[1,2,5])
def test_kernel_solution_matches_explicit_weighted_polynomial_basis(alpha,degree):
    rng=np.random.default_rng(7);x=rng.uniform(.1,.9,(9,2));y=rng.normal(size=9);query=rng.uniform(.1,.9,(4,2))
    def expanded(values):
        z=np.column_stack([np.ones(len(values)),(2*values-1)/math.sqrt(2)])
        powers=[p for p in itertools.product(range(degree+1),repeat=3) if sum(p)==degree]
        return np.column_stack([math.sqrt(math.factorial(degree)/math.prod(math.factorial(v) for v in p)/2**degree)*np.prod(z**p,axis=1) for p in powers])
    features=expanded(x);offset=features.mean(axis=0);a=features-offset
    if alpha:
        coefficients=np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@(y-y.mean()))
    else:coefficients=np.linalg.lstsq(a,y-y.mean(),rcond=None)[0]
    expected=(expanded(query)-offset)@coefficients+y.mean()
    model=fit_design_polynomial(x,y,degree,alpha)
    model.update(input_indices=[0,1],target_transform="identity")
    np.testing.assert_allclose(predict_design(model,query),expected,rtol=1e-8,atol=1e-9)


@pytest.mark.parametrize("robust",[False,True])
def test_physical_generator_recomputes_labels_preserves_regions_and_source(tmp_path,robust):
    base=tmp_path/'source';original=source(base,count=4);before=(base/'dataset.json').read_bytes()
    original['fixed_config']['robust']=robust
    (base/'dataset.json').write_text(json.dumps(original));before=(base/'dataset.json').read_bytes()
    events=[];directory=tmp_path/'generated'
    data=generate_designs(base,directory,DesignSettings(variants=2),emit=events.append)
    assert data['status']=='complete' and len(data['cases'])==12
    assert (base/'dataset.json').read_bytes()==before
    assert data['fixed_config']['projection_beta']==data['fixed_config']['filter_radius']==0
    assert data['source_robust']==robust
    assert not data['fixed_config']['robust'] and data['fixed_config']['state_solver']=='fom'
    for case in data['cases']:
        assert case['status']=='complete' and case['compliance']>0 and case['residual']<1e-8
        assert np.array(case['density'])[[0,6]].tolist()==[1,1]
    np.testing.assert_allclose(data['cases'][0]['density'],original['cases'][0]['density'])
    checked=evaluate_design(directory,data['cases'][0]['density'])
    assert checked['compliance']==pytest.approx(data['cases'][0]['compliance'])
    changed=data['cases'][0]['density'].copy();changed[3]=.05
    assert evaluate_design(directory,changed)['compliance']!=pytest.approx(checked['compliance'])
    changed[0]=.5
    with pytest.raises(ValueError,match='fixed'):evaluate_design(directory,changed)
    partial=generate_designs(base,tmp_path/'partial',DesignSettings(),stop_requested=lambda:len(events)>13,emit=events.append)
    assert partial['status']=='cancelled' and len(partial['cases'])==2


def test_grouped_split_unites_duplicates_and_keeps_families_together(tmp_path):
    data=synthetic(tmp_path)
    # An identical density in a different family must unite both families.
    data['cases'][1]['density']=data['cases'][0]['density']
    cases,train,val,test,groups=split_design_cases(data)
    assert groups[0]==groups[1]
    assert all(set(groups[i] for i in a).isdisjoint(groups[i] for i in b) for a,b in ((train,val),(train,test),(val,test)))
    for family in {c['family'] for c in cases}:
        members={i for i,c in enumerate(cases) if c['family']==family}
        assert sum(bool(members.intersection(subset)) for subset in (train,val,test))==1
    assert sum(map(len,(train,val,test)))==len(cases)


@pytest.mark.parametrize("method",['ols','ridge','neural'])
def test_fits_portable_full_density_models_without_test_leakage(tmp_path,method):
    data=synthetic(tmp_path);events=[]
    report=fit_design_models(tmp_path,method=method,epochs=150,emit=events.append,degrees=[1,2,13])
    assert read_design_models(tmp_path,method)==report
    assert set(report['train_indices']).isdisjoint(report['test_indices']+report['validation_indices'])
    inputs=[case['density'] for case in data['cases']]
    for model in report['models'].values():
        prediction=predict_design(model,inputs)
        assert np.isfinite(prediction).all() and (prediction>0).all()
        assert 0 not in model['input_indices'] and 6 not in model['input_indices']
        code="import {predictDesign} from './src/ml_opt/web/static/design-models.js';let s='';for await(const c of process.stdin)s+=c;const [m,x]=JSON.parse(s);console.log(JSON.stringify(x.map(v=>predictDesign(m,v))));"
        result=subprocess.run(['node','--input-type=module','-e',code],input=json.dumps([model,inputs]),text=True,capture_output=True,check=True)
        np.testing.assert_allclose(json.loads(result.stdout),prediction,rtol=1e-11,atol=1e-11)
        if method=='neural':
            assert len(model['weights']['hidden_weight'][0])==10
            assert model['best_validation_loss']==min(row['validation_loss'] for row in model['training_history'])
        if method=='ols' and model['degree']==1:assert model['train']['mean_relative_error']<1e-8
    # Alter held-out targets only: fit parameters / NN checkpoint cannot change.
    for case in data['cases']:
        if case['index'] in report['test_indices']:case['compliance']*=100
    (tmp_path/'dataset.json').write_text(json.dumps(data))
    refit=fit_design_models(tmp_path,method=method,epochs=150,degrees=[1,2,13])
    for key,model in report['models'].items():
        for parameter in ('weights','dual_coefficients','target_mean','target_offset','target_scale','best_epoch'):
            if parameter in model:assert model[parameter]==refit['models'][key][parameter]


def test_models_coexist_splits_match_and_cancel_preserves_artifacts(tmp_path):
    synthetic(tmp_path)
    reports=[fit_design_models(tmp_path,method=method,epochs=10) for method in ('ols','ridge','neural')]
    assert all(r['test_indices']==reports[0]['test_indices'] and r['train_indices']==reports[0]['train_indices'] and r['validation_indices']==reports[0]['validation_indices'] for r in reports)
    before=(tmp_path/'models-neural.json').read_bytes()
    assert fit_design_models(tmp_path,method='neural',stop_requested=lambda:True) is None
    assert (tmp_path/'models-neural.json').read_bytes()==before
    assert all(read_design_models(tmp_path,method) for method in ('ols','ridge','neural'))
    (tmp_path/'dataset.json').write_text((tmp_path/'dataset.json').read_text()+' ')
    with pytest.raises(ValueError,match='match'):read_design_models(tmp_path,'ridge')


@pytest.mark.parametrize('options',[{'variants':True},{'variants':-1},{'amplitude':float('nan')},{'amplitude':1.1},{'seed':-1}])
def test_invalid_sampling_settings(options):
    with pytest.raises(ValueError):DesignSettings(**options)
