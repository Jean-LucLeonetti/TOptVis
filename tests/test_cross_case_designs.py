"""Auxiliary optimisation cases must never become the regression target physics."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from ml_opt.cross_case_designs import CrossCaseSettings,alternative_cases,generate_cross_designs
from ml_opt.design_regression import evaluate_design,fit_design_models,split_design_cases
from test_connectivity import config


def test_case_generator_varies_supports_and_loads_but_preserves_original():
    original=config(filter_radius=.1,projection_beta=2)
    before=asdict(original)
    settings=CrossCaseSettings(case_count=10,max_iterations=3)
    cases=alternative_cases(original,settings)
    assert len(cases)==10 and asdict(original)==before
    assert len({str(c.boundary_conditions) for _,c in cases})==10
    for _,case in cases:
        assert case.connection_anchors==original.connection_anchors and case.connection_targets==original.connection_targets
        assert case.non_design_regions==original.non_design_regions
        assert (case.nx,case.ny,case.penalisation,case.filter_radius,case.robust)==(12,4,3,.1,False)
    assert cases==alternative_cases(original,settings)
    assert cases!=alternative_cases(original,replace(settings,seed=1))


def test_labels_use_original_case_and_all_variants_stay_in_family(tmp_path):
    original=config(filter_radius=.08,projection_beta=2,robust=True)
    events=[]
    data=generate_cross_designs(original,tmp_path/'cross',CrossCaseSettings(case_count=4,max_iterations=4,variants=1),emit=events.append)
    assert len(data['cases'])==8 and data['status']=='complete'
    assert data['fixed_config']['boundary_conditions']==json.loads(json.dumps(asdict(original)))['boundary_conditions']
    assert data['fixed_config']['filter_radius']==data['fixed_config']['projection_beta']==0
    assert not data['fixed_config']['robust'] and data['source_robust']
    assert len(data['families'])==4
    assert sum(e['event']=='design_optimization' for e in events)==20
    for case in data['cases']:
        assert case['status']=='complete'
        checked=evaluate_design(tmp_path/'cross',case['density'])
        assert case['compliance']==pytest.approx(checked['compliance'])
        assert checked['connection_connected']==case['connection_connected']
    cases,train,val,test,groups=split_design_cases(data)
    assert all(set(groups[i] for i in a).isdisjoint(groups[i] for i in b) for a,b in ((train,val),(train,test),(val,test)))
    for method in ('ols','ridge','neural'):
        report=fit_design_models(tmp_path/'cross',method=method,epochs=10,degrees=[1,2])
        assert report['test_indices']==[cases[i]['index'] for i in test]


def test_cancellation_keeps_already_labelled_families(tmp_path):
    count=[0]
    def event(row):
        if row['event']=='design_case':count[0]+=1
    data=generate_cross_designs(config(),tmp_path/'partial',CrossCaseSettings(case_count=4,max_iterations=2,variants=2),emit=event,stop_requested=lambda:count[0]>=3)
    assert data['status']=='cancelled' and len(data['cases'])==3
    saved=json.loads((tmp_path/'partial'/'dataset.json').read_text())
    assert saved==data


@pytest.mark.parametrize('options',[{'case_count':2},{'case_count':True},{'max_iterations':0},{'max_iterations':1001}])
def test_settings_reject_invalid_batches(options):
    with pytest.raises(ValueError):CrossCaseSettings(**options)
