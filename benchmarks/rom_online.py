"""Train and benchmark exact affine ROM assembly on larger cantilever meshes.

Run serially with the checkout's src on PYTHONPATH and the project Python:
python benchmarks/rom_online.py --repeats 3

Defaults: 120x40 and 240x80, nine FOM training trajectories, 99 snapshots,
20 online iterations. Benchmark-only training uses CantileverConfig directly
so the interactive SnapshotConfig mesh limits remain unchanged. Core wall
times include per-run basis loading/setup/teardown, exclude app startup and
rendering. Training, POD/operator preparation, and disk writes are timed too.
The basis is never enriched from any benchmark/test trajectory.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import ModuleType
import uuid

import numpy as np

from benchmark import environment
from ml_opt.config import CantileverConfig
from ml_opt.fem.state_solver import create_state_solver
from ml_opt.problem import CantileverProblem
from ml_opt.rom.solver import load_basis
from ml_opt.rom.training import _save_arrays, build_basis, read_metadata, save_metadata
from ml_opt.rom.validation import ValidationConfig, compare_evaluations
from ml_opt.runner import optimise


def source_fingerprints():
    root=Path(__file__).resolve().parents[1]
    paths=['src/ml_opt/rom/solver.py','src/ml_opt/rom/operators.py',
        'src/ml_opt/rom/training.py','src/ml_opt/rom/validation.py',
        'src/ml_opt/fem/state_solver.py','benchmarks/rom_online.py']
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def summary(samples):
    return dict(median=statistics.median(samples), min=min(samples), max=max(samples), samples=samples)


def legacy_solver(revision):
    source = subprocess.check_output(['git', 'show', f'{revision}:src/ml_opt/rom/solver.py'], text=True)
    module = ModuleType('ml_opt.rom._benchmark_sparse_projection')
    module.__package__ = 'ml_opt.rom'
    sys.modules[module.__name__] = module
    exec(compile(source, f'{revision}/rom/solver.py', 'exec'), module.__dict__)
    return module.GalerkinSolver


def train(nx, ny, parent, iterations, modes, energy):
    directory = parent / uuid.uuid4().hex
    directory.mkdir(parents=True)
    cases = [CantileverConfig(nx=nx, ny=ny, load_node_x=nx,
        load_node_y=int(np.floor(height*ny+.5)), load_angle=-90,
        volume_fraction=volume, max_iterations=iterations, min_iterations=iterations,
        filter_radius=.1, projection_beta=2) for height,volume in product((.25,.5,.75),(.3,.4,.5))]
    columns, indices, steps, residuals, compliances = [], [], [], [], []
    start = time.perf_counter()
    # Capture only free displacement snapshots. Operators use the same build_basis
    # implementation as the training tab; this benchmark does not need density files.
    for index,config in enumerate(cases):
        free = create_state_solver(config)
        free_dofs = free.mesh.free_dofs.copy(); free.close()
        def capture(row, displacement, density):
            if row.iteration % 2 and row.iteration != iterations:
                return
            columns.append(displacement[free_dofs].copy())
            indices.append(index); steps.append(row.iteration)
            residuals.append(row.linear_residual); compliances.append(row.compliance)
        optimise(config, state_snapshot_callback=capture)
        print(f'{nx}x{ny} training {index+1}/{len(cases)}', flush=True)
    snapshot_seconds = time.perf_counter()-start
    metadata = dict(schema_version=1, id=directory.name,
        created_at=datetime.now(timezone.utc).isoformat(), status='snapshots_ready',
        settings=dict(nx=nx,ny=ny,length=3.,height=1.,benchmark_only=True),
        cases=[asdict(c) for c in cases], snapshot_count=len(columns),
        completed_cases=len(cases),case_count=len(cases),free_dof_count=len(free_dofs))
    save_metadata(directory,metadata)
    start=time.perf_counter()
    _save_arrays(directory/'snapshots.npz', displacements=np.column_stack(columns),
        free_dofs=free_dofs,case_indices=np.array(indices),iterations=np.array(steps),
        residuals=np.array(residuals),compliances=np.array(compliances))
    snapshot_write_seconds=time.perf_counter()-start
    del columns
    start=time.perf_counter()
    metadata=build_basis(directory,energy_target=energy,max_modes=modes)
    basis_seconds=time.perf_counter()-start
    costs=dict(snapshot_generation_seconds=snapshot_seconds,
        snapshot_write_seconds=snapshot_write_seconds,basis_operators_and_write_seconds=basis_seconds,
        total_seconds=snapshot_seconds+snapshot_write_seconds+basis_seconds)
    save_metadata(parent,dict(directory=str(directory),costs=costs))
    return directory,metadata,costs


def analyse_result(result, full_config):
    with CantileverProblem(full_config) as full:
        full.projection_beta=result.evaluation.projection_beta
        reference=full.evaluate(result.design)
    return {**compare_evaluations(reference,result.evaluation,ValidationConfig()),
        'full_volume':reference.volume_fraction,'reported_volume':result.evaluation.volume_fraction,
        'full_volume_constraint':reference.constraint}


def benchmark(args):
    reference_class=legacy_solver(args.baseline)
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        working_tree_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest(),
        source_files_sha256=source_fingerprints(),
        baseline=args.baseline,environment=environment(),options=vars(args)|{'output':str(args.output)},
        timing_scope='Warmed serial CPU process; online timings include model loading, setup, gradient evaluation and teardown; exclude offline training, browser/startup/output files.',
        quality_scope='Unseen load/volume combination; final ROM design independently evaluated with FOM. Equal iterations do not establish equal accuracy or convergence.',cases=[])
    args.output.mkdir(parents=True,exist_ok=True)
    for nx,ny in args.meshes:
        parent=args.output/f'{nx}x{ny}'; parent.mkdir(exist_ok=True)
        if args.reuse_training and (parent/'metadata.json').exists():
            record=read_metadata(parent); directory=Path(record['directory'])
            metadata=read_metadata(directory); costs=record['costs']
        else:
            directory,metadata,costs=train(nx,ny,parent,args.training_iterations,args.modes,args.energy)
        base=CantileverConfig(nx=nx,ny=ny,load_node_x=nx,load_node_y=int(np.floor(.4*ny+.5)),
            load_angle=-75,volume_fraction=.35,max_iterations=args.iterations,min_iterations=args.iterations,
            filter_radius=.1,projection_beta=2)
        configs={'fom':base,'rom':replace(base,state_solver='rom',rom_basis_path=str(directory/'basis.npz'))}
        case=dict(mesh=[nx,ny],free_dofs=2*nx*(ny+1),basis_rank=metadata['basis']['rank'],
            basis_target_met=metadata['basis']['target_met'],operator_bytes=metadata['basis']['operator_bytes'],
            offline=costs,training_directory=str(directory),config=asdict(base),optimisation={})
        print(f'{nx}x{ny}: {case["basis_rank"]} modes, offline {costs["total_seconds"]:.2f}s',flush=True)
        # The same physical moduli follow a FOM trajectory, independent of ROM.
        moduli=[]
        def capture(row,u,density):
            moduli.append(base.minimum_modulus+(base.young_modulus-base.minimum_modulus)*density**base.penalisation)
        optimise(base,state_snapshot_callback=capture)
        basis,info=load_basis(configs['rom'].rom_basis_path,base)
        case['basis_sha256']=info['basis_sha256']
        solvers={'fom':create_state_solver(base),'rom':create_state_solver(configs['rom']),
                 'previous_rom':reference_class(base,basis)}
        timings={name:[] for name in solvers}
        equivalence=[]
        try:
            for solver in solvers.values(): solver.solve(moduli[0])
            for repetition in range(args.repeats):
                names=list(solvers); names=names[repetition%3:]+names[:repetition%3]
                for name in names:
                    start=time.perf_counter()
                    for values in moduli: solvers[name].solve(values)
                    timings[name].append((time.perf_counter()-start)/len(moduli))
            for values in moduli:
                previous=solvers['previous_rom'].solve(values); current=solvers['rom'].solve(values)
                np.testing.assert_allclose(current.displacement,previous.displacement,rtol=3e-7,atol=1e-7)
                equivalence.append(abs(current.compliance-previous.compliance)/abs(previous.compliance))
        finally:
            for solver in solvers.values(): solver.close()
        del solvers
        case['state_solve_seconds']={n:summary(v) for n,v in timings.items()}
        case['max_previous_rom_compliance_difference']=max(equivalence)
        print('state ms: '+str({n:round(statistics.median(v)*1000,3) for n,v in timings.items()}),flush=True)
        for method in ['mma','gcmma']:
            settings={n:replace(c,optimizer=method) for n,c in configs.items()}
            for config in settings.values(): optimise(replace(config,max_iterations=2))
            trials={n:[] for n in settings}; last_results={}
            for repetition in range(args.repeats):
                for name in (['fom','rom'] if repetition%2==0 else ['rom','fom']):
                    start=time.perf_counter(); result=optimise(settings[name]); elapsed=time.perf_counter()-start
                    trials[name].append(dict(seconds=elapsed,iterations=result.history[-1].iteration,
                        state_solves=result.history[-1].state_solves,compliance=result.evaluation.compliance,
                        volume=result.evaluation.volume_fraction,stop_reason=result.stop_reason))
                    last_results[name]=result
            measured={n:dict(timing=summary([v['seconds'] for v in values]),trials=values)
                      for n,values in trials.items()}
            measured['rom_final_fom_check']=analyse_result(last_results['rom'],settings['fom'])
            measured['rom_design_full_compliance_vs_fom_design_ratio']=(
                measured['rom_final_fom_check']['full_compliance']/last_results['fom'].evaluation.compliance)
            for n,result in last_results.items():
                _save_arrays(parent/f'{method}-{n}-final.npz',design=result.design,density=result.evaluation.density,
                    displacement=result.evaluation.displacement,gradient=result.evaluation.objective_gradient)
            saved=measured['fom']['timing']['median']-measured['rom']['timing']['median']
            measured['speedup']=measured['fom']['timing']['median']/measured['rom']['timing']['median']
            measured['timing_only_break_even_runs']=math.ceil(costs['total_seconds']/saved) if saved>0 else None
            case['optimisation'][method]=measured
            print(f'{nx}x{ny} {method}: FOM {measured["fom"]["timing"]["median"]:.3f}s, ROM {measured["rom"]["timing"]["median"]:.3f}s, speedup {measured["speedup"]:.2f}x, final C error {measured["rom_final_fom_check"]["compliance_error"]:.2%}',flush=True)
        report['cases'].append(case)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meshes',nargs='+',default=['120x40','240x80'])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--training-iterations',type=int,default=20)
    parser.add_argument('--modes',type=int,default=64)
    parser.add_argument('--energy',type=float,default=1-1e-8)
    parser.add_argument('--baseline',default='c3eeee1')
    parser.add_argument('--reuse-training',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('outputs/benchmarks/rom-online'))
    options=parser.parse_args()
    if min(options.repeats,options.iterations,options.training_iterations)<1:
        parser.error('repeats and iteration counts must be positive')
    try: options.meshes=[tuple(map(int,value.lower().split('x'))) for value in options.meshes]
    except ValueError: parser.error('meshes must be NxM, for example 120x40')
    if any(len(mesh)!=2 or min(mesh)<2 for mesh in options.meshes):
        parser.error('meshes must have two dimensions of at least 2')
    benchmark(options)
