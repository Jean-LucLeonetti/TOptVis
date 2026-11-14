"""Independent FOM validation and saved, explicitly identified model trials."""

from contextlib import contextmanager
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from ..config import CantileverConfig
from ..problem import CantileverProblem
from ..rom.training import _save_arrays,read_metadata,save_metadata
from ..rom.validation import ValidationConfig,compare_evaluations,validation_cases
from ..runner import _optimise_problem
from .data import file_hash
from .inference import ModelProblem,Predictor


@contextmanager
def lab_threads():
    previous=torch.get_num_threads()
    torch.set_num_threads(min(4,previous))
    try: yield
    finally: torch.set_num_threads(previous)


def source_cases(source,model,options):
    metadata=read_metadata(source)
    case_hash=hashlib.sha256(json.dumps(metadata['cases'],sort_keys=True).encode()).hexdigest()
    if (metadata['id']!=model['source_training_id'] or file_hash(Path(source)/'snapshots.npz')!=model['source_sha256']
        or case_hash!=model['source_case_sha256']):
        raise ValueError('The source dataset changed since this model was trained')
    # Reuse the established unseen-case, snapped-node and memory checks.
    return validation_cases({**metadata,'status':'basis_ready','basis':{'rank':1}},options)


def validate_model(directory,source,options:ValidationConfig,*,modes=('surrogate','corrected'),
                   validation_id=None,expected_sha256=None,emit=lambda event:None,stop_requested=lambda:False):
    directory=Path(directory); metadata=read_metadata(directory)
    cases=source_cases(source,metadata,options)
    if not modes or len(set(modes))!=len(modes) or any(m not in ('surrogate','corrected') for m in modes):
        raise ValueError('Choose surrogate and/or corrected validation, without duplicates')
    base=cases[0]; free=2*base.nx*(base.ny+1)
    if 8*(2*free+4*base.nx*base.ny)*options.maximum_states*len(modes)>128*1024**2:
        raise ValueError('Comparison arrays exceed 128 MiB; reduce comparison states or modes')
    validation_id=validation_id or uuid.uuid4().hex
    report=dict(id=validation_id,model_id=metadata['id'],family=metadata['family'],
        model_sha256=expected_sha256 or metadata['model_sha256'],options=asdict(options),modes=list(modes),
        created_at=datetime.now(timezone.utc).isoformat(),cases=[asdict(c) for c in cases],results=[],
        case_signature=hashlib.sha256(json.dumps([asdict(c) for c in cases],sort_keys=True).encode()).hexdigest(),
        scope='Unseen FOM trajectories; validation does not train weights or enrich bases. Adaptive fallback and FEM correction counts are reported.',
        gradient='Raw neural gradients differentiate the network through material/filter/projection; Galerkin and corrected FEM use physical adjoints.')
    arrays={key:[] for key in ('designs','densities','full_displacements','model_displacements','full_gradients','model_gradients')}
    totals={mode:dict(predictions=0,fom_solves=0,correction_iterations=0,lu_fallbacks=0,basis_updates=0) for mode in modes}
    with lab_threads():
        for index,case in enumerate(cases):
            if stop_requested(): break
            emit(dict(event='ml_validation_case',model_id=metadata['id'],case_index=index,case_count=len(cases)))
            problems={}
            try:
                for mode in modes:
                    problems[mode]=ModelProblem(case,Predictor(directory,case,mode=mode,
                        expected_sha256=report['model_sha256'],stop_requested=stop_requested))
                with CantileverProblem(case) as full_problem:
                    last_seconds=0.; original=full_problem.evaluate
                    def timed(design,sensitivities=True):
                        nonlocal last_seconds
                        start=time.perf_counter(); result=original(design,sensitivities)
                        last_seconds=time.perf_counter()-start
                        return result
                    full_problem.evaluate=timed
                    def compare(row,design,full):
                        if stop_requested() or (row.iteration%options.comparison_stride and row.iteration!=case.max_iterations): return
                        full_seconds=last_seconds
                        for mode,problem in problems.items():
                            if stop_requested(): break
                            problem.projection_beta=full.projection_beta
                            start=time.perf_counter(); predicted=problem.evaluate(design)
                            elapsed=time.perf_counter()-start
                            result=dict(case_index=index,iteration=row.iteration,mode=mode,
                                model_seconds=elapsed,full_seconds=full_seconds,
                                **compare_evaluations(full,predicted,options))
                            report['results'].append(result)
                            for key,value in (('designs',design),('densities',full.density),
                                ('full_displacements',full.displacement[problem.mesh.free_dofs]),
                                ('model_displacements',predicted.displacement[problem.mesh.free_dofs]),
                                ('full_gradients',full.objective_gradient),('model_gradients',predicted.objective_gradient)):
                                arrays[key].append(value.copy())
                            emit(dict(event='ml_validation_state',model_id=metadata['id'],family=metadata['family'],
                                result=result,density=full.density.tolist(),config=asdict(case)))
                    _optimise_problem(full_problem,None,None,stop_requested,time.perf_counter(),evaluation_callback=compare)
            finally:
                for mode,problem in problems.items():
                    for key,value in problem.solver.stats().items():
                        if key in totals[mode]: totals[mode][key]+=value
                    problem.close()
    if not report['results']: return None
    report['status']='cancelled' if stop_requested() else 'complete'
    report['summaries']={}
    for mode in modes:
        rows=[r for r in report['results'] if r['mode']==mode]
        metrics={}
        for name in ('displacement','compliance','gradient'):
            values=[r[f'{name}_error'] for r in rows if r[f'{name}_error'] is not None]
            metrics[name]=dict(max=max(values) if values else None,mean=float(np.mean(values)) if values else None)
        report['summaries'][mode]=dict(metrics=metrics,state_count=len(rows),passed_states=sum(r['passed'] for r in rows),
            meets_tolerances=bool(rows) and report['status']=='complete' and all(r['passed'] for r in rows),
            model_evaluation_seconds=sum(r['model_seconds'] for r in rows),
            full_evaluation_seconds=sum(r['full_seconds'] for r in rows),solver_statistics=totals[mode])
    destination=directory/'validations'/validation_id; destination.mkdir(parents=True,exist_ok=False)
    _save_arrays(destination/'arrays.npz',**{key:np.column_stack(value) for key,value in arrays.items()},
        case_indices=np.array([r['case_index'] for r in report['results']]),
        iterations=np.array([r['iteration'] for r in report['results']]),
        modes=np.array([r['mode'] for r in report['results']]),metadata_json=np.array(json.dumps(report)))
    (destination/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    metadata['validation']={key:value for key,value in report.items() if key not in ('results','cases')}
    save_metadata(directory,metadata)
    return report


def experiment_config(metadata,options):
    base=CantileverConfig(**metadata['config'])
    height=options.get('load_height',.4)
    if not np.isfinite(height) or not 0<=height<=1: raise ValueError('Load height must be between 0 and 1')
    return replace(base,load_node_x=base.nx,load_node_y=int(np.floor(height*base.ny+.5)),
        load_angle=options.get('load_angle',-75.),volume_fraction=options.get('volume_fraction',.35),
        optimizer=options.get('optimizer','mma'),max_iterations=options.get('max_iterations',20),
        min_iterations=options.get('max_iterations',20),robust=False,max_solid_size=0,
        projection_schedule='fixed',state_solver='fom',rom_basis_path=None,rom_basis_sha256=None)


def run_experiment(directory,options,*,experiment_id=None,expected_sha256=None,emit=lambda event:None,stop_requested=lambda:False):
    started=time.perf_counter(); directory=Path(directory); metadata=read_metadata(directory)
    config=experiment_config(metadata,options); mode=options.get('mode','corrected')
    experiment_id=experiment_id or uuid.uuid4().hex
    destination=directory/'experiments'/experiment_id; destination.mkdir(parents=True,exist_ok=False)
    predictor=None
    with lab_threads():
        predictor=Predictor(directory,config,mode=mode,expected_sha256=expected_sha256,stop_requested=stop_requested)
        with ModelProblem(config,predictor) as problem:
            def iteration(entry):
                if mode=='surrogate' and entry.compliance<=0:
                    raise ValueError('This raw model predicts nonpositive compliance. Train further or choose FEM correction.')
                predictor.accept()
                entry.state_solver=f'{metadata["family"]}/{mode}'
                ranks=predictor.stats()['basis_ranks']
                entry.rom_basis_rank=max(ranks) if ranks else None
            def snapshot(entry,density):
                emit(dict(event='ml_iteration',history=asdict(entry),density=density.tolist(),
                    config=asdict(config),solver_statistics=predictor.stats()))
            result=_optimise_problem(problem,iteration,snapshot,stop_requested,started)
            statistics=predictor.stats()
            adaptive_basis=predictor.bases[0].copy() if metadata['family']=='adaptive_pod' else np.empty((0,0))
    elapsed=time.perf_counter()-started
    # Always check the physical final design, even when the model stopping test passes.
    start=time.perf_counter()
    with CantileverProblem(config) as reference:
        reference.projection_beta=result.evaluation.projection_beta
        full=reference.evaluate(result.design)
    check_seconds=time.perf_counter()-start
    report=dict(id=experiment_id,model_id=metadata['id'],model_sha256=expected_sha256 or metadata['model_sha256'],
        family=metadata['family'],mode=mode,config=asdict(config),status='cancelled' if result.stop_reason=='cancelled' else 'complete',
        stop_reason=result.stop_reason,iterations=result.history[-1].iteration,elapsed_seconds=elapsed,
        final_fom_check_seconds=check_seconds,history=[asdict(row) for row in result.history],
        solver_statistics=statistics,final_fom_check=compare_evaluations(full,result.evaluation,ValidationConfig()),
        full_volume=full.volume_fraction,full_volume_constraint=full.constraint,
        scope='Single-field trial. Raw predictions are experimental. FEM correction solves full equilibrium; its cost and any LU fallback are included. Final FOM check is timed separately.')
    _save_arrays(destination/'arrays.npz',design=result.design,density=result.evaluation.density,
        model_displacement=result.evaluation.displacement,full_displacement=full.displacement,
        model_gradient=result.evaluation.objective_gradient,full_gradient=full.objective_gradient,
        adaptive_basis=adaptive_basis,metadata_json=np.array(json.dumps(report)))
    (destination/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    metadata['experiment']={key:value for key,value in report.items() if key not in ('history','config')}
    save_metadata(directory,metadata)
    return report
