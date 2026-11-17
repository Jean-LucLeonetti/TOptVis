from fastapi.testclient import TestClient
from ml_opt.web.app import create_app
from test_connection_interface import request
from test_regression_interface import events


def test_cross_case_generation_fit_and_original_check(tmp_path):
    with TestClient(create_app(regression_root=tmp_path/'regression')) as client:
        assert client.get('/api/health').json()['cross_case_designs_supported']
        payload={'problem':request(filter_radius=.08),'case_count':4,'max_iterations':3,'variants':1}
        result=events(client.post('/api/design-regression/cross-cases',json=payload))
        identifier=result[0]['dataset_id'];endpoint=f'/api/design-regression/{identifier}'
        data=client.get(endpoint).json()['dataset']
        assert data['source_kind']=='cross_cases' and len(data['cases'])==8
        assert len([e for e in result if e['event']=='design_optimization'])==16
        assert data['source_config']['connection_targets']==payload['problem']['connection_targets']
        assert data['families'][1]['config']['boundary_conditions']!=data['source_config']['boundary_conditions']
        events(client.post(endpoint+'/fit',json={'method':'ridge'}))
        checked=events(client.post(endpoint+'/evaluate',json={'density':data['cases'][2]['density']}))
        value=next(e['evaluation']['compliance'] for e in checked if e['event']=='design_evaluation')
        assert abs(value-data['cases'][2]['compliance'])<1e-8
        assert client.get(endpoint+'/artifacts/dataset').json()==data
        regenerated=events(client.post('/api/design-regression/cross-cases',json={'original_dataset_id':identifier,'case_count':3,'max_iterations':2,'variants':0}))
        saved=client.get(f"/api/design-regression/{regenerated[0]['dataset_id']}").json()['dataset']
        assert saved['source_config']==data['source_config']
        for invalid in ({},{'source_id':'not-an-id'},{**payload,'source_id':'a'*32},{**payload,'case_count':2}):
            assert client.post('/api/design-regression/cross-cases',json=invalid).status_code==422
