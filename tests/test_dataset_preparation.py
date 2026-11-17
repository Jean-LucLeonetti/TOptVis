"""Saved optimisation setups produce one target-labelled, reloadable dataset."""
from copy import deepcopy

from fastapi.testclient import TestClient
import pytest

from ml_opt.web.app import create_app
from test_connection_interface import request
from test_regression_interface import events


def preparation():
    model=request(filter_radius=.08, max_iterations=2)
    return {"name":"Combined designs", "target":{"name":"Target", "problem":model},
        "sources":[{"name":"Target", "problem":model},
            {"name":"Other load", "problem":{**model, "load_angle":-45}}],
        "volume_min":.3,"volume_max":.6,"volume_count":3,"variants":0}


def test_selected_setups_are_labelled_under_target_and_saved(tmp_path):
    root=tmp_path/'regression'
    payload=preparation()
    with TestClient(create_app(regression_root=root)) as client:
        result=events(client.post('/api/design-regression/prepare',json=payload))
        endpoint=f"/api/design-regression/{result[0]['dataset_id']}"
        data=client.get(endpoint).json()['dataset']
        assert data['name']=='Combined designs' and data['source_kind']=='saved_cases'
        assert len(data['families'])==6 and len(data['cases'])==6
        assert all(case['status']=='complete' for case in data['cases'])
        assert data['families'][3]['config']['load_angle']==-45
        assert data['fixed_config']['load_angle']==-90
        # Re-evaluation under the chosen target reproduces every label.
        for case in data['cases']:
            checked=events(client.post(endpoint+'/evaluate',json={'density':case['density']}))
            actual=next(e['evaluation']['compliance'] for e in checked if e['event']=='design_evaluation')
            assert actual==pytest.approx(case['compliance'],rel=1e-10)
        assert client.get(endpoint+'/artifacts/dataset').json()==data
        assert client.get('/api/design-regression/datasets').json()['datasets'][0]['name']=='Combined designs'
        trained=events(client.post(endpoint+'/fit',json={'method':'ridge','degrees':[1]}))
        assert trained[-1]['stop_reason']=='complete'
    with TestClient(create_app(regression_root=root)) as client:
        assert client.get(endpoint).json()['dataset']==data
        assert client.get(endpoint).json()['reports']['ridge']


@pytest.mark.parametrize("count",[1,3])
def test_one_saved_setup_is_supported(tmp_path,count):
    payload=preparation();payload['sources']=payload['sources'][:1]
    payload['volume_count']=count
    with TestClient(create_app(regression_root=tmp_path/'regression')) as client:
        result=events(client.post('/api/design-regression/prepare',json=payload))
        data=client.get(f"/api/design-regression/{result[0]['dataset_id']}").json()['dataset']
        assert data['family_count']==count and len(data['cases'])==count


def test_rejects_incompatible_setups_before_starting(tmp_path):
    with TestClient(create_app(regression_root=tmp_path/'regression')) as client:
        base=preparation()
        for update in ({'sources':[]},{'name':'  '},{'volume_min':.7,'volume_max':.6},
                       {'volume_count':True},{'volume_min':.3,'volume_max':.3}):
            assert client.post('/api/design-regression/prepare',json={**base,**update}).status_code==422
        for key,value in (('nx',14),('non_design_regions',[{'kind':'solid','selection':{'kind':'surface','start':[0,0],'end':[.1,1]}}])):
            payload=deepcopy(base);payload['sources'][0]['problem'][key]=value
            response=client.post('/api/design-regression/prepare',json=payload)
            assert response.status_code==422
        assert client.get('/api/design-regression/datasets').json()=={'datasets':[]}
