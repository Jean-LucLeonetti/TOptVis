"""Local Model Lab routes using the existing serialized, cancellable worker."""

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any,Literal

from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,ConfigDict,Field,model_validator
import torch

from ..config import CantileverConfig
from ..model_lab.config import CATALOG,POD_FAMILIES,TrainOptions
from ..model_lab.data import case_key
from ..model_lab.experiments import experiment_config,source_cases
from ..model_lab.training import device_for
from ..rom.training import read_metadata
from ..rom.validation import ValidationConfig
from .rom import ValidationRequest,training_directory
from .streams import stream_job


class TrainingRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    training_id:str=Field(pattern=r'^[0-9a-f]{32}$')
    options:dict[str,Any]=Field(default_factory=dict)

    @model_validator(mode='after')
    def valid_options(self):
        try: TrainOptions(**self.options)
        except (ValueError,TypeError) as error: raise ValueError(str(error)) from error
        return self


class ModelValidationRequest(ValidationRequest):
    modes:list[Literal['surrogate','corrected']]=Field(default_factory=lambda:['surrogate','corrected'],min_length=1,max_length=2)

    def configuration(self):
        return ValidationConfig(**self.model_dump(exclude={'modes','model_ids'}))

    @model_validator(mode='after')
    def distinct_modes(self):
        if len(set(self.modes))!=len(self.modes): raise ValueError('Evaluation modes must be distinct')
        return self


class CompareRequest(ModelValidationRequest):
    model_ids:list[str]=Field(min_length=1,max_length=16)

    @model_validator(mode='after')
    def distinct_models(self):
        if len(set(self.model_ids))!=len(self.model_ids): raise ValueError('Select each model only once')
        return self


class ExperimentRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    mode:Literal['surrogate','corrected']='corrected'
    load_height:float=Field(default=.4,ge=0,le=1)
    load_angle:float=Field(default=-75,ge=-180,le=180)
    volume_fraction:float=Field(default=.35,gt=0,lt=1)
    optimizer:Literal['mma','gcmma']='mma'
    max_iterations:int=Field(default=20,ge=1,le=300)


def model_directory(root,model_id):
    if not re.fullmatch(r'[0-9a-f]{32}',model_id): raise HTTPException(404,'Model not found')
    directory=Path(root)/model_id
    if directory.is_symlink() or directory.resolve().parent!=Path(root).resolve():
        raise HTTPException(404,'Model not found')
    return directory


def create_model_router(manager,root,training_root):
    router=APIRouter(prefix='/api/ml')
    root=Path(root).resolve(); training_root=Path(training_root).resolve()

    def record(model_id,require_ready=False):
        directory=model_directory(root,model_id)
        try: metadata=read_metadata(directory)
        except (OSError,ValueError): raise HTTPException(404,'Model not found') from None
        if require_ready and (not metadata.get('checkpoint_ready') or metadata.get('status') not in ('trained','cancelled')):
            raise HTTPException(409,'Train this model to save a checkpoint first')
        if require_ready and (not (directory/'model.pt').is_file() or (directory/'model.pt').is_symlink()):
            raise HTTPException(404,'Model checkpoint not found')
        return directory,metadata

    def start(task,payload,started):
        try: job=manager.start_task(task,dict(root=str(root),training_root=str(training_root),**payload))
        except RuntimeError as error: raise HTTPException(409,str(error)) from error
        return stream_job(manager,job,started(job))

    def validation_selection(ids,request):
        records=[]; signature=None
        for model_id in ids:
            _,metadata=record(model_id,True)
            source=training_directory(training_root,metadata['source_training_id'])
            try: cases=source_cases(source,metadata,request.configuration())
            except (OSError,ValueError) as error: raise HTTPException(422,str(error)) from error
            current=[asdict(c) for c in cases]
            if signature is not None and signature!=current:
                raise HTTPException(422,'Comparison models must use the same mesh, physics, filtering and validation cases')
            signature=current
            records.append(dict(model_id=model_id,model_sha256=metadata['model_sha256'],training_id=metadata['source_training_id']))
        return records

    @router.get('/catalog')
    async def catalog():
        return dict(families=[dict(id=item[0],name=item[1],group=item[2],description=item[3]) for item in CATALOG],
            defaults=asdict(TrainOptions()),cuda_available=torch.cuda.is_available(),devices=['cpu']+(['cuda'] if torch.cuda.is_available() else []),
            neural_node_limit=6000,operator_limit_mib=512,
            scope='Small experimental architectures on a fixed mesh with a clamped left edge. Transformer is a pooled-token reference, not a pretrained Transolver model.')

    @router.get('/models')
    async def models():
        records=[]
        if root.is_dir():
            for directory in root.iterdir():
                if re.fullmatch(r'[0-9a-f]{32}',directory.name) and directory.is_dir() and not directory.is_symlink():
                    try:
                        metadata=read_metadata(directory)
                        if metadata['status']=='training' and (manager.active is None or manager.active.id!=directory.name):
                            metadata['status']='interrupted'
                        metadata.pop('history',None)
                        records.append(metadata)
                    except (OSError,KeyError,ValueError): continue
        return {'models':sorted(records,key=lambda m:m['created_at'],reverse=True)}

    @router.get('/models/{model_id}')
    async def model(model_id:str):
        return record(model_id)[1]

    @router.post('/train')
    async def train(request:TrainingRequest):
        directory=training_directory(training_root,request.training_id)
        try:
            metadata=read_metadata(directory)
            if metadata.get('snapshot_count',0)<2 or metadata.get('status') not in ('snapshots_ready','basis_ready','cancelled'):
                raise ValueError('Generate a nonempty full-order snapshot dataset first')
            cases=[CantileverConfig(**c) for c in metadata['cases']]
            if len({case_key(c) for c in cases})<2:
                raise ValueError('At least two independent cases are needed for a case-level split')
            options=TrainOptions(**request.options)
            if options.family not in POD_FAMILIES:
                device_for(options.device)
                if (cases[0].nx+1)*(cases[0].ny+1)>6000:
                    raise ValueError('Neural training supports at most 6000 nodes; choose a smaller dataset')
        except OSError: raise HTTPException(404,'Snapshot dataset not found') from None
        except (ValueError,TypeError) as error: raise HTTPException(422,str(error)) from error
        return start('ml_train',dict(training_id=request.training_id,options=asdict(options)),
            lambda job:dict(kind='ml_train',model_id=job.id,family=options.family))

    @router.post('/models/{model_id}/validate')
    async def validate(model_id:str,request:ModelValidationRequest):
        records=validation_selection([model_id],request)
        return start('ml_validate',dict(models=records,options=asdict(request.configuration()),modes=request.modes),
            lambda job:dict(kind='ml_validate',model_id=model_id))

    @router.post('/compare')
    async def compare(request:CompareRequest):
        records=validation_selection(request.model_ids,request)
        return start('ml_validate',dict(models=records,options=asdict(request.configuration()),modes=request.modes),
            lambda job:dict(kind='ml_compare',model_ids=request.model_ids))

    @router.post('/models/{model_id}/experiment')
    async def experiment(model_id:str,request:ExperimentRequest):
        _,metadata=record(model_id,True)
        try: experiment_config(metadata,request.model_dump())
        except ValueError as error: raise HTTPException(422,str(error)) from error
        return start('ml_experiment',dict(model_id=model_id,model_sha256=metadata['model_sha256'],options=request.model_dump()),
            lambda job:dict(kind='ml_experiment',model_id=model_id))

    def report_path(directory,metadata,kind,filename):
        report_id=metadata.get(kind,{}).get('id','')
        if not re.fullmatch(r'[0-9a-f]{32}',report_id): raise HTTPException(404,'No saved report')
        parent=directory/('validations' if kind=='validation' else 'experiments')
        child=parent/report_id
        if parent.is_symlink() or child.is_symlink() or child.resolve().parent!=parent.resolve():
            raise HTTPException(404,'Report not found')
        path=child/filename
        if not path.is_file() or path.is_symlink(): raise HTTPException(404,'Report not found')
        return path

    @router.get('/models/{model_id}/reports/{kind}')
    async def report(model_id:str,kind:Literal['validation','experiment']):
        directory,metadata=record(model_id)
        return json.loads(report_path(directory,metadata,kind,'report.json').read_text(encoding='utf-8'))

    @router.get('/models/{model_id}/artifacts/{artifact}')
    async def artifact(model_id:str,artifact:Literal['model','metadata','validation','validation_arrays','experiment','experiment_arrays']):
        directory,metadata=record(model_id)
        if artifact in ('model','metadata'):
            path=directory/('model.pt' if artifact=='model' else 'metadata.json')
            if not path.is_file() or path.is_symlink(): raise HTTPException(404,'Artifact not found')
        else:
            kind='validation' if artifact.startswith('validation') else 'experiment'
            path=report_path(directory,metadata,kind,'arrays.npz' if artifact.endswith('_arrays') else 'report.json')
        return FileResponse(path,filename=f'{metadata["family"]}-{model_id[:8]}-{artifact}{path.suffix}')

    return router
