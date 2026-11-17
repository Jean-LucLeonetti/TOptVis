"""Stream and persist diffusion connection constraints through the actual worker."""
from dataclasses import asdict
import json

from fastapi.testclient import TestClient

from ml_opt.web.app import create_app
from ml_opt.config import CantileverConfig
from test_connectivity import config
from test_regression_interface import events


def request(**options):
    cfg=config(**options)
    allowed=client_fields()
    data={key:value for key,value in asdict(cfg).items() if key in allowed and value is not None}
    data.update(load_node_x=cfg.nx,load_node_y=cfg.ny//2)
    return json.loads(json.dumps(data))


def client_fields():
    from ml_opt.web.app import RunRequest
    return RunRequest.model_fields


def test_optimisation_stream_reports_temperature_paths_and_feasibility(tmp_path):
    with TestClient(create_app(training_root=tmp_path/'rom')) as client:
        assert client.get('/api/health').json()['connection_constraints_supported']
        data=request(max_iterations=3,filter_radius=.08)
        assert client.post('/api/project/validate',json=data).json()['model']['connection_targets']==data['connection_targets']
        stream=events(client.post('/api/run',json=data))
        iterations=[e for e in stream if e['event']=='iteration']
        assert len(iterations)==4
        for event in iterations:
            assert len(event['connection_temperatures']['target-1'])==48
            assert isinstance(event['connection_connected']['target-1'],bool)
            assert 'target-1' in event['history']['connection_constraints']
        final=stream[-1]
        assert final['connection_feasible']==(iterations[-1]['history']['connection_constraints']['target-1']<=1e-4)
        if not final['connection_feasible']:assert not final['feasible']
        original=data['connection_anchors'];data['connection_anchors']=[]
        assert client.post('/api/run',json=data).status_code==422
        data['connection_anchors']=original;data['connection_targets']=original
        assert client.post('/api/run',json=data).status_code==422


def test_regression_import_retains_connections_and_feasibility(tmp_path):
    with TestClient(create_app(regression_root=tmp_path/'regression')) as client:
        data=request(max_iterations=2)
        stream=events(client.post('/api/regression/generate',json={'problem':data,'case_count':6,'max_iterations':2}))
        identifier=stream[0]['dataset_id']
        dataset=client.get(f'/api/regression/{identifier}').json()['dataset']
        assert dataset['fixed_config']['connection_targets']==data['connection_targets']
        assert all('target-1' in c['constraints'] for c in dataset['cases'])
