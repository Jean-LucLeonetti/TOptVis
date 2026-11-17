"""Optimise alternative boundary cases, then label under one original setup."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .boundaries import BoundaryCondition, MeshSelection, LoadCase
from .design_regression import DesignSettings, analysis_config
from .regression import _now, _save_json
from .regions import region_densities


@dataclass(frozen=True)
class CrossCaseSettings(DesignSettings):
    case_count: int = 12
    max_iterations: int = 150

    def __post_init__(self):
        super().__post_init__()
        if type(self.case_count) is not int or self.case_count<3:
            raise ValueError("At least three optimisation cases are required")
        if type(self.max_iterations) is not int or not 1<=self.max_iterations<=1000:
            raise ValueError("Optimisation iterations must lie between 1 and 1000")


def alternative_cases(original, settings):
    """All cases share mesh/material/design regions and connection selections."""
    original=replace(original,state_solver="fom",rom_basis_path=None,rom_basis_sha256=None,
        field_execution="sequential",max_iterations=settings.max_iterations)
    result=[("Original case",original)]
    def boundary(kind,start,end=None):
        return BoundaryCondition(kind,MeshSelection("line" if end is not None else "point",start,end))
    profiles=[("Clamped left",[boundary("fixed",(0,0),(0,1))]),
        ("Clamped right",[boundary("fixed",(1,0),(1,1))]),
        ("Clamped bottom",[boundary("fixed",(0,0),(1,0))]),
        ("Clamped top",[boundary("fixed",(0,1),(1,1))]),
        ("Pinned/roller bottom",[boundary("fixed",(0,0)),boundary("restrain_y",(1,0))]),
        ("Pinned/roller top",[boundary("fixed",(0,1)),boundary("restrain_y",(1,1))]),
        ("Pinned/roller left",[boundary("fixed",(0,0)),boundary("restrain_x",(0,1))]),
        ("Pinned/roller right",[boundary("fixed",(1,0)),boundary("restrain_x",(1,1))])]
    rng=np.random.default_rng(settings.seed)
    order=rng.permutation(len(profiles))
    perimeter=sorted({(x,y) for x in range(original.nx+1) for y in (0,original.ny)} |
        {(x,y) for x in (0,original.nx) for y in range(original.ny+1)})
    fixed=region_densities(original.non_design_regions,original.nx,original.ny)
    from .connectivity import connection_elements
    for i in range(1,settings.case_count):
        name,supports=profiles[int(order[(i-1)%len(order)])]
        for _ in range(200):
            x,y=perimeter[int(rng.integers(len(perimeter)))]
            point=MeshSelection("point",(x/original.nx,y/original.ny))
            if np.any(fixed[connection_elements(point,original.nx,original.ny)]==0):
                continue
            angle=float(rng.uniform(-180,180))
            force=BoundaryCondition("force",point,magnitude=original.load,angle=angle,case_id="case-1")
            try:
                candidate=replace(original,boundary_conditions=tuple(supports)+ (force,),
                    load_cases=(LoadCase("case-1",f"Auxiliary case {i}",1.),),load_aggregation="weighted",
                    load_node_x=None,load_node_y=None)
            except ValueError:
                continue
            result.append((name,candidate));break
        else:
            raise ValueError(f"Could not place a valid auxiliary force for {name}")
    return result


def generate_cross_designs(original, directory, settings, *, emit=lambda event:None,stop_requested=lambda:False):
    from .problem import CantileverProblem
    from .runner import optimise
    from .design_regression import perturbed_density

    configurations=alternative_cases(original,settings)
    target=analysis_config({"fixed_config":asdict(original)})
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    canonical=json.dumps(asdict(original),sort_keys=True)
    data={"format":"toptvis-design-regression-v1","id":directory.name,"created_at":_now(),
        "status":"generating","source_kind":"cross_cases","source_id":None,
        "source_sha256":hashlib.sha256(canonical.encode()).hexdigest(),"source_config":json.loads(canonical),
        "source_robust":original.robust,"fixed_config":json.loads(json.dumps(asdict(target))),
        "settings":asdict(settings),"objective_quantity":"elastic_work" if target.imposed_motion else "compliance",
        "field":"nominal","family_count":len(configurations),"expected_cases":len(configurations)*(settings.variants+1),
        "cases":[],"families":[],"generation_seconds":0.}
    started=perf_counter();generator=torch.Generator().manual_seed(settings.seed)
    def save():
        data['generation_seconds']=perf_counter()-started;_save_json(directory/'dataset.json',data)
    save()
    with CantileverProblem(target) as problem:
        for family,(name,configuration) in enumerate(configurations):
            if stop_requested():break
            def progress(entry):
                emit({"event":"design_optimization","family":family,"family_count":len(configurations),
                    "iteration":entry.iteration,"max_iterations":configuration.max_iterations,"compliance":entry.compliance})
            try:
                result=optimise(configuration,callback=progress,stop_requested=stop_requested)
                constraints=[result.evaluation.constraint,*((result.evaluation.connection_constraints or {}).values())]
                constraints.extend(value for value in (result.evaluation.max_size_constraint,
                    result.evaluation.max_void_constraint,result.evaluation.epigraph_constraint) if value is not None)
                metadata={"index":family,"name":name,"config":json.loads(json.dumps(asdict(configuration))),
                    "converged":result.converged,"stop_reason":result.stop_reason,
                    "feasible":all(value<=configuration.feasibility_tolerance for value in constraints),
                    "source_objective":result.evaluation.compliance,"iterations":result.history[-1].iteration}
                data['families'].append(metadata);save()
                if stop_requested():break
                base=result.evaluation.density
            except Exception as error:
                data['families'].append({"index":family,"name":name,"status":"failed","message":str(error)})
                save();emit({"event":"design_family_failed","family":family,"message":str(error)});continue
            for variant in range(settings.variants+1):
                if stop_requested():break
                density=base if variant==0 else perturbed_density(base,target,settings.amplitude,generator)
                case={"index":len(data['cases']),"family":str(family),"variant":variant,"density":density.tolist()}
                try:
                    evaluation=problem.evaluate(density,sensitivities=False)
                    if not np.isfinite(evaluation.compliance) or evaluation.compliance<=0:
                        raise ValueError("Original-case objective is not positive and finite")
                    case.update(status="complete",compliance=evaluation.compliance,physical_volume=evaluation.volume_fraction,
                        residual=evaluation.residual,connection_constraints=evaluation.connection_constraints,
                        connection_connected=evaluation.connection_connected)
                except Exception as error:
                    case.update(status="failed",message=str(error))
                data['cases'].append(case);save()
                emit({"event":"design_case","case":case,"completed_cases":len(data['cases']),"case_count":data['expected_cases']})
    data['status']='cancelled' if stop_requested() else 'complete'
    save();return data
