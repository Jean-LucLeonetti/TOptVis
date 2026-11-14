"""Train each architecture through the worker and compare/optimise real models."""

import hashlib
import json

from fastapi.testclient import TestClient

from ml_opt.model_lab.config import FAMILIES
from ml_opt.web.app import create_app


def events(response,terminal='complete'):
    assert response.status_code==200,response.text
    rows=[json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert rows[-1]['event']==terminal,rows[-1]
    return rows


def test_model_lab_trains_compares_and_runs_actual_corrected_optimisation(tmp_path):
    training=tmp_path/'data'; models=tmp_path/'models'
    with TestClient(create_app(training_root=training,model_root=models)) as client:
        assert client.get('/api/health').json()['model_lab_supported']
        catalog=client.get('/api/ml/catalog').json()
        assert {item['id'] for item in catalog['families']}==FAMILIES
        page=client.get('/').text
        assert 'id="tab-model-lab"' in page and 'id="ml-train-form"' in page
        assert client.get('/static/model-lab.js').status_code==200
        assert client.get('/static/model-lab.css').status_code==200
        generated=events(client.post('/api/rom/snapshots',json=dict(nx=6,ny=3,load_heights=[.25,.75],
            load_angles=[-90,-45],volume_fractions=[.4],iterations_per_case=2,snapshot_stride=1)))
        training_id=generated[-1]['training_id']
        before=hashlib.sha256((training/training_id/'snapshots.npz').read_bytes()).hexdigest()
        ids={}
        for family in sorted(FAMILIES):
            fitted=events(client.post('/api/ml/train',json=dict(training_id=training_id,
                options=dict(family=family,epochs=2,width=8,layers=1,max_modes=2,batch_size=4))))
            model=fitted[-1]['model'];ids[family]=model['id']
            assert model['status']=='trained' and model['checkpoint_ready']
            assert model['split']['train_case_indices'] and model['split']['validation_case_indices']
            assert any(row['event']=='ml_epoch' for row in fitted)
            checkpoint=client.get(f'/api/ml/models/{model["id"]}/artifacts/model')
            assert hashlib.sha256(checkpoint.content).hexdigest()==model['model_sha256']
        records=client.get('/api/ml/models').json()['models']
        assert len(records)==len(FAMILIES)
        validation=dict(load_heights=[.5],load_angles=[-65],volume_fractions=[.35],
            iterations_per_case=1,comparison_stride=1,modes=['surrogate','corrected'])
        comparison=events(client.post('/api/ml/compare',json=dict(model_ids=[ids['cnn'],ids['pod']],**validation)))
        reports=comparison[-1]['reports']
        assert len(reports)==2 and reports[0]['case_signature']==reports[1]['case_signature']
        assert all(report['summaries']['corrected']['meets_tolerances'] for report in reports)
        assert reports[0]['summaries']['surrogate']['solver_statistics']['fom_solves']==0
        report=client.get(f'/api/ml/models/{ids["cnn"]}/reports/validation').json()
        assert len(report['results'])==4
        experiment=events(client.post(f'/api/ml/models/{ids["cnn"]}/experiment',
            json=dict(mode='corrected',max_iterations=2)))
        result=experiment[-1]['experiment']
        assert result['final_fom_check']['passed'] and result['solver_statistics']['fom_solves']==3
        frames=[row for row in experiment if row['event']=='ml_iteration']
        assert [row['history']['iteration'] for row in frames]==[0,1,2]
        assert all(row['history']['state_solver']=='cnn/corrected' for row in frames)
        assert client.get(f'/api/ml/models/{ids["cnn"]}/artifacts/experiment_arrays').status_code==200
        assert hashlib.sha256((training/training_id/'snapshots.npz').read_bytes()).hexdigest()==before
        assert not client.get('/api/health').json()['running']
        for body in [dict(training_id=training_id,options={'family':'unknown'}),
            dict(training_id='../elsewhere'),dict(training_id=training_id,options={'epochs':0}),
            dict(training_id=training_id,options={'extra':True})]:
            assert client.post('/api/ml/train',json=body).status_code==422
        overlapping={**validation,'load_heights':[.25],'load_angles':[-90],'volume_fractions':[.4]}
        assert client.post(f'/api/ml/models/{ids["cnn"]}/validate',json=overlapping).status_code==422
        assert client.post('/api/ml/compare',json=dict(model_ids=[ids['cnn'],ids['cnn']],**validation)).status_code==422
        assert client.get('/api/ml/models/invalid').status_code==404
        assert client.post('/api/ml/models/'+'f'*32+'/experiment',json={}).status_code==404


def test_failed_training_does_not_poison_reused_worker(tmp_path):
    training=tmp_path/'data'; models=tmp_path/'models'
    with TestClient(create_app(training_root=training,model_root=models)) as client:
        generated=events(client.post('/api/rom/snapshots',json=dict(nx=6,ny=3,load_heights=[.25,.75],
            load_angles=[-90],volume_fractions=[.4],iterations_per_case=1,snapshot_stride=1)))
        training_id=generated[-1]['training_id']
        invalid=training/('f'*32);invalid.mkdir()
        metadata=json.loads((training/training_id/'metadata.json').read_text())
        metadata['id']='f'*32
        (invalid/'metadata.json').write_text(json.dumps(metadata))
        failed=events(client.post('/api/ml/train',json=dict(training_id='f'*32,options={'family':'pod'})),terminal='error')
        assert failed[-1]['message']
        fitted=events(client.post('/api/ml/train',json=dict(training_id=training_id,options={'family':'pod'})))
        assert fitted[-1]['model']['checkpoint_ready']
        assert not client.get('/api/health').json()['running']
