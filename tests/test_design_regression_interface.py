"""Exercise real shared-worker generation, all three fits and edited-density checks."""
import json

from fastapi.testclient import TestClient
import pytest

from ml_opt.web.app import create_app
from test_design_regression import source
from test_regression_interface import events


def test_generate_train_check_download_reload(tmp_path):
    root=tmp_path/'regression';identifier='a'*32;root.mkdir()
    source_dir=root/identifier;source(source_dir,count=6)
    original=(source_dir/'dataset.json').read_bytes()
    with TestClient(create_app(regression_root=root)) as client:
        assert client.get('/api/health').json()['design_regression_supported']
        assert client.get('/api/design-regression/datasets').json()=={'datasets':[]}
        assert client.get('/static/design-regression.js').status_code==200
        stream=events(client.post('/api/design-regression/generate',json={'source_id':identifier,'variants':1}))
        dataset_id=stream[0]['dataset_id'];endpoint=f'/api/design-regression/{dataset_id}'
        assert len([e for e in stream if e['event']=='design_case'])==12
        data=client.get(endpoint).json()['dataset']
        assert len(data['cases'])==12 and data['status']=='complete'
        reports={}
        for method in ('ols','ridge','neural'):
            fitted=events(client.post(endpoint+'/fit',json={'method':method,'epochs':20,'degrees':[1,2,12]}))
            if method=='neural':assert any(e['event']=='neural_training' for e in fitted)
            reports[method]=client.get(endpoint).json()['reports'][method]
            assert client.get(endpoint+f'/artifacts/models-{method}').json()==reports[method]
            assert not client.get('/api/health').json()['running']
        assert all(r['test_indices']==reports['ols']['test_indices'] for r in reports.values())
        assert client.get(endpoint+'/artifacts/dataset').json()==data
        assert (source_dir/'dataset.json').read_bytes()==original
        density=data['cases'][0]['density'].copy();density[3]=.05
        checked=events(client.post(endpoint+'/evaluate',json={'density':density}))
        evaluation=next(e['evaluation'] for e in checked if e['event']=='design_evaluation')
        assert evaluation['compliance']>0 and evaluation['residual']<1e-8
        assert evaluation['compliance']!=pytest.approx(data['cases'][0]['compliance'])
        assert client.get(endpoint).json()['dataset']==data
        # Invalid requests never replace existing models or start a worker job.
        for invalid in ({'method':'cnn'},{'degrees':[True]},{'degrees':[0]},{'ridge_alpha':-1},{'log_target':'yes'},{'validation_fraction':.05}):
            assert client.post(endpoint+'/fit',json=invalid).status_code==422
        density[0]=.5
        for invalid in (density,[.5],[-1]*12):
            assert client.post(endpoint+'/evaluate',json={'density':invalid}).status_code==422
        assert client.get(endpoint).json()['reports']==reports
        assert client.get('/api/design-regression/datasets').json()['datasets'][0]['case_count']==12
        assert client.get('/api/design-regression/no-such-dataset').status_code==404
    with TestClient(create_app(regression_root=root)) as client:
        assert client.get(endpoint).json()['reports']==reports
        assert client.get(endpoint).json()['dataset']==data


def test_rejects_invalid_sources_and_artifacts(tmp_path):
    client=TestClient(create_app(regression_root=tmp_path/'regression'))
    for payload in ({},{'source_id':'../x'},{'source_id':'a'*32,'variants':True},{'source_id':'a'*32,'amplitude':2}):
        assert client.post('/api/design-regression/generate',json=payload).status_code==422
    assert client.post('/api/design-regression/generate',json={'source_id':'a'*32}).status_code==404
    assert client.get('/api/design-regression/'+'b'*32+'/artifacts/dataset').status_code==404
