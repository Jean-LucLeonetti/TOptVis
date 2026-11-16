"""Non-design selections survive validation, streaming, projects and regression."""

import json

from fastapi.testclient import TestClient
import numpy as np

from ml_opt.web.app import create_app


REGIONS=[{'kind':'solid','selection':{'kind':'surface','start':[0,0],'end':[1/6,.5]}},
         {'kind':'void','selection':{'kind':'surface','start':[.5,0],'end':[2/3,.25]}}]


def test_fixed_regions_reach_streamed_optimisation_and_regression(tmp_path):
    with TestClient(create_app(regression_root=tmp_path)) as client:
        assert client.get('/api/health').json()['non_design_regions_supported']
        model={'nx':6,'ny':4,'load_node_x':6,'load_node_y':2,'max_iterations':2,
               'non_design_regions':REGIONS,'robust':True}
        response=client.post('/api/project/validate',json=model)
        assert response.status_code==200,response.text
        saved=response.json()['model']
        assert saved['non_design_regions']==REGIONS
        response=client.post('/api/run',json=saved)
        assert response.status_code==200,response.text
        events=[json.loads(line) for line in response.text.splitlines()]
        assert events[-1]['event']=='complete',events[-1]
        for event in events:
            if event['event']!='iteration':continue
            for density in event['field_densities'].values():
                np.testing.assert_array_equal(np.array(density)[[0,6,3]],[1,1,0])
        response=client.post('/api/regression/generate',json={'case_count':6,'problem':saved})
        assert response.status_code==200,response.text
        events=[json.loads(line) for line in response.text.splitlines()]
        assert events[-1]['event']=='complete',events[-1]
        dataset=client.get('/api/regression/'+events[0]['dataset_id']).json()['dataset']
        assert dataset['settings']['source_problem']['non_design_regions']==REGIONS
        for case in dataset['cases']:
            assert 'density' in case,case
            np.testing.assert_array_equal(np.array(case['density'])[[0,6,3]],[1,1,0])


def test_invalid_and_infeasible_regions_are_rejected_before_starting_worker(tmp_path):
    app=create_app(regression_root=tmp_path)
    client=TestClient(app)
    for regions in [[{'kind':'solid','selection':{'kind':'point','start':[.5,.5]}}],
                    [REGIONS[0],{**REGIONS[0],'kind':'void'}],
                    [{'kind':'solid','selection':{'kind':'surface','start':[0,0],'end':[1,1]}}]]:
        response=client.post('/api/project/validate',json={'non_design_regions':regions})
        assert response.status_code==422,response.text
    assert app.state.manager._process is None
    response=client.post('/api/regression/generate',json={'volume_max':.99,'problem':{
        'nx':6,'ny':4,'load_node_x':6,'load_node_y':2,'volume_constraint':'gt','non_design_regions':REGIONS}})
    assert response.status_code==422,response.text
    assert app.state.manager._process is None
