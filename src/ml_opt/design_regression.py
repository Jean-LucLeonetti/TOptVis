"""Physical-density to objective regression under a fixed structural setup."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .config import CantileverConfig
from .regression import _now, _save_json, _errors, read_dataset as read_volume_dataset, validate_degrees, validate_regression
from .neural_regression import NeuralSettings, train_network, predict_network
from .regions import region_densities


@dataclass(frozen=True)
class DesignSettings:
    variants: int = 4
    amplitude: float = .2
    seed: int = 42

    def __post_init__(self):
        if type(self.variants) is not int or self.variants < 0:
            raise ValueError("Variants per design must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32-1:
            raise ValueError("Sampling seed must be a nonnegative 32-bit integer")
        if (isinstance(self.amplitude, bool) or not isinstance(self.amplitude, (int, float)) or
                not math.isfinite(self.amplitude) or not 0 <= self.amplitude <= 1):
            raise ValueError("Perturbation size must be between 0 and 1")


def analysis_config(data):
    return replace(CantileverConfig(**data["fixed_config"]), filter_radius=0., projection_beta=0.,
        projection_schedule="fixed", robust=False, state_solver="fom", rom_basis_path=None,
        rom_basis_sha256=None, field_execution="sequential")


def validate_density(density, config):
    array = np.asarray(density, dtype=float)
    if array.shape != (config.nx*config.ny,) or not np.all(np.isfinite(array)) or np.any((array<0)|(array>1)):
        raise ValueError("Provide one finite physical density in [0, 1] per mesh element")
    fixed = region_densities(config.non_design_regions, config.nx, config.ny)
    mask = np.isfinite(fixed)
    if not np.allclose(array[mask], fixed[mask], atol=1e-12, rtol=0):
        raise ValueError("Preserve fixed solid and void regions")
    return np.where(mask, fixed, array)


def source_designs(data):
    config = analysis_config(data)
    bases, seen = [], set()
    for case in data["cases"]:
        if case.get("density") is None:
            continue
        density = validate_density(case["density"], config)
        digest = hashlib.sha256(density.tobytes()).hexdigest()
        if digest not in seen:
            seen.add(digest); bases.append((case["index"], density))
    if len(bases)<3:
        raise ValueError("At least three distinct base designs are needed for independent training, validation and test families")
    return config, bases


def generate_designs(source_directory, directory, settings, *, emit=lambda event: None, stop_requested=lambda: False):
    from .problem import CantileverProblem

    source_directory, directory = Path(source_directory), Path(directory)
    source = read_volume_dataset(source_directory)
    config, bases = source_designs(source)
    directory.mkdir(parents=True, exist_ok=False)
    total = len(bases)*(settings.variants+1)
    data = {"format":"toptvis-design-regression-v1", "id":directory.name, "created_at":_now(),
        "status":"generating", "source_id":source["id"],
        "source_sha256":hashlib.sha256((source_directory/"dataset.json").read_bytes()).hexdigest(),
        "source_robust":bool(source["fixed_config"].get("robust")),
        "fixed_config":json.loads(json.dumps(asdict(config))), "settings":asdict(settings),
        "objective_quantity":"elastic_work" if config.imposed_motion else "compliance",
        "field":"nominal", "family_count":len(bases), "expected_cases":total,
        "cases":[], "generation_seconds":0.}
    start = perf_counter()
    def save():
        data["generation_seconds"] = perf_counter()-start
        _save_json(directory/"dataset.json",data)
    save()
    fixed = region_densities(config.non_design_regions, config.nx, config.ny)
    mask = np.isfinite(fixed)
    generator = torch.Generator().manual_seed(settings.seed)
    with CantileverProblem(config) as problem:
        for family, base in bases:
            for variant in range(settings.variants+1):
                if stop_requested(): break
                density = base.copy()
                if variant:
                    coarse = torch.rand((1,1,min(6,config.ny),min(12,config.nx)),generator=generator,dtype=torch.float64)*2-1
                    noise = F.interpolate(coarse,size=(config.ny,config.nx),mode="bilinear",align_corners=False).numpy().ravel()
                    shift = float(torch.rand((),generator=generator))*.5-.25
                    density = np.clip(base+settings.amplitude*(noise-noise.mean()+shift),0,1)
                density[mask] = fixed[mask]
                index = len(data["cases"])
                case = {"index":index,"family":str(family),"variant":variant,"density":density.tolist()}
                try:
                    evaluation = problem.evaluate(density,sensitivities=False)
                    if not math.isfinite(evaluation.compliance) or evaluation.compliance<=0:
                        raise ValueError("Structural objective must be positive and finite")
                    case.update(compliance=evaluation.compliance,density=evaluation.density.tolist(),
                        physical_volume=evaluation.volume_fraction,residual=evaluation.residual,status="complete")
                except Exception as error:
                    case.update(status="failed",message=str(error))
                data["cases"].append(case); save()
                emit({"event":"design_case","case":case,"completed_cases":len(data["cases"]),"case_count":total})
            if stop_requested(): break
    data["status"] = "cancelled" if stop_requested() or len(data["cases"])<total else "complete"
    save()
    return data


def read_design_dataset(directory):
    data=json.loads((Path(directory)/"dataset.json").read_text())
    if data.get("format")!="toptvis-design-regression-v1":
        raise ValueError("Unsupported design regression dataset")
    return data


def split_design_cases(data, *, test_fraction=.25, validation_fraction=.2, seed=42):
    if not math.isfinite(test_fraction) or not .1<=test_fraction<=.5 or not math.isfinite(validation_fraction) or not .1<=validation_fraction<=.5:
        raise ValueError("Test and validation shares must be between 0.1 and 0.5")
    if type(seed) is not int or not 0<=seed<=2**32-1:
        raise ValueError("Split seed must be a nonnegative 32-bit integer")
    config=analysis_config(data)
    cases=[case for case in data["cases"] if case.get("status")=="complete"]
    parents={str(case["family"]):str(case["family"]) for case in cases}
    def root(group):
        while parents[group]!=group:
            parents[group]=parents[parents[group]];group=parents[group]
        return group
    seen={}
    for case in cases:
        density=validate_density(case["density"],config)
        if not math.isfinite(case["compliance"]) or case["compliance"]<=0:
            raise ValueError("Training objectives must be finite and positive")
        digest=hashlib.sha256(density.tobytes()).hexdigest()
        group=str(case["family"])
        if digest in seen: parents[root(group)]=root(seen[digest])
        else: seen[digest]=group
    assignments=[root(str(case["family"])) for case in cases]
    groups=sorted(set(assignments))
    if len(groups)<3:
        raise ValueError("At least three independent design families are required; duplicate densities stay in the same split")
    rng=np.random.default_rng(seed)
    test_groups=set(rng.choice(groups,min(max(1,math.ceil(len(groups)*test_fraction)),len(groups)-2),replace=False))
    remaining=sorted(set(groups)-test_groups)
    val_groups=set(rng.choice(remaining,min(max(1,math.ceil(len(remaining)*validation_fraction)),len(remaining)-1),replace=False))
    train=np.array([i for i,g in enumerate(assignments) if g not in test_groups|val_groups])
    validation=np.array([i for i,g in enumerate(assignments) if g in val_groups])
    test=np.array([i for i,g in enumerate(assignments) if g in test_groups])
    return cases,train,validation,test,assignments


def polynomial_kernel(left,right,degree):
    # Full weighted polynomial basis, including cross terms; no expanded matrix.
    dot=(left*2-1)@(right*2-1).T/left.shape[1]
    return torch.clamp((1+dot)/2,min=0,max=1).pow(degree)


def fit_design_polynomial(inputs,targets,degree,alpha):
    x=torch.as_tensor(inputs,dtype=torch.float64)
    y=torch.as_tensor(targets,dtype=torch.float64)
    kernel=polynomial_kernel(x,x,degree)
    column_mean=kernel.mean(dim=0);mean=kernel.mean()
    centred=kernel-column_mean[None,:]-column_mean[:,None]+mean
    eigenvalues,vectors=torch.linalg.eigh((centred+centred.T)/2)
    tolerance=torch.finfo(torch.float64).eps*len(inputs)*max(float(eigenvalues.max()),1e-15)
    retained=eigenvalues>tolerance
    inverse=torch.zeros_like(eigenvalues)
    if alpha>0: inverse=1/(eigenvalues.clamp_min(0)+alpha)
    else: inverse[retained]=1/eigenvalues[retained]
    target_mean=y.mean()
    dual=vectors@(inverse*(vectors.T@(y-target_mean)))
    if not torch.isfinite(dual).all():
        raise ValueError("Polynomial fit exceeded numerical precision; increase the ridge penalty")
    feature_count = math.comb(inputs.shape[1]+degree, degree)
    feature_description = str(feature_count) if feature_count.bit_length()<12000 else f"binomial({inputs.shape[1]+degree}, {degree})"
    return {"kind":"design_polynomial","degree":degree,"training_inputs":x.tolist(),
        "kernel_column_mean":column_mean.tolist(),"kernel_mean":float(mean),"target_mean":float(target_mean),
        "dual_coefficients":dual.tolist(),"training_rank":int(retained.sum())+1,
        "feature_count":feature_description}


def predict_design(model,densities):
    values=np.asarray(densities,dtype=float)
    single=values.ndim==1
    batch=values[None,:] if single else values
    if not np.all(np.isfinite(batch)) or np.any((batch<0)|(batch>1)):
        raise ValueError("Input densities must be finite and lie in [0, 1]")
    if model["kind"]=="neural_network":
        prediction=np.atleast_1d(predict_network(model,batch))
    else:
        inputs=torch.tensor(batch[:,model["input_indices"]],dtype=torch.float64)
        kernel=polynomial_kernel(inputs,torch.tensor(model["training_inputs"],dtype=torch.float64),model["degree"])
        centred=kernel-kernel.mean(dim=1,keepdim=True)-torch.tensor(model["kernel_column_mean"],dtype=torch.float64)+model["kernel_mean"]
        prediction=(centred@torch.tensor(model["dual_coefficients"],dtype=torch.float64)+model["target_mean"]).numpy()
    if model["target_transform"]=="log":
        with np.errstate(over="ignore",invalid="ignore"): prediction=np.exp(prediction)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Prediction exceeds numerical range")
    return float(prediction[0]) if single else prediction


def read_design_models(directory,method):
    if method not in ("ols","ridge","neural"):raise ValueError("Unknown design regression method")
    directory=Path(directory);path=directory/f"models-{method}.json"
    if not path.is_file():return None
    report=json.loads(path.read_text())
    if report.get("format")!="toptvis-design-models-v1" or report.get("dataset_sha256")!=hashlib.sha256((directory/"dataset.json").read_bytes()).hexdigest():
        raise ValueError("Saved design models do not match this dataset; train again")
    return report


def fit_design_models(directory, *, method="ridge", degrees=(1,2),ridge_alpha=.01,test_fraction=.25,
                      validation_fraction=.2,seed=42,log_target=True,neurons=8,epochs=2000,
                      learning_rate=.0001,weight_penalty=0.,network_seed=42,
                      emit=lambda event:None,stop_requested=lambda:False,allow_interrupted=False):
    directory=Path(directory);data=read_design_dataset(directory)
    if data["status"]=="generating" and not allow_interrupted:
        raise ValueError("Wait for design generation to finish before fitting")
    if type(log_target) is not bool:raise ValueError("Log target must be a boolean")
    if method!="neural":degrees=validate_degrees(degrees,method=method,ridge_alpha=ridge_alpha)
    settings=NeuralSettings(neurons,epochs,learning_rate,weight_penalty,network_seed,validation_fraction)
    cases,train,validation,test,groups=split_design_cases(data,test_fraction=test_fraction,validation_fraction=validation_fraction,seed=seed)
    config=analysis_config(data);fixed=region_densities(config.non_design_regions,config.nx,config.ny)
    indices=np.flatnonzero(np.isnan(fixed))
    densities=np.array([case["density"] for case in cases]);actual=np.array([case["compliance"] for case in cases])
    targets=np.log(actual) if log_target else actual
    report={"format":"toptvis-design-models-v1","dataset_id":data["id"],"method":method,"trained_at":_now(),
        "dataset_sha256":hashlib.sha256((directory/"dataset.json").read_bytes()).hexdigest(),"degrees":list(degrees) if method!="neural" else [],
        "ridge_alpha":validate_regression(method,ridge_alpha) if method!="neural" else 0.,
        "test_fraction":test_fraction,"validation_fraction":validation_fraction,"seed":seed,"log_target":log_target,
        "train_indices":[cases[i]["index"] for i in train],"validation_indices":[cases[i]["index"] for i in validation],
        "test_indices":[cases[i]["index"] for i in test],"family_assignments":{str(case["index"]):groups[i] for i,case in enumerate(cases)},
        "excluded_indices":[case["index"] for case in data["cases"] if case not in cases],"models":{},**vars(settings)}
    report["baseline"]={}
    # A volume-only predictor tests whether the design map learns more than mass.
    volume=densities.mean(axis=1);features=np.column_stack([np.ones(len(cases)),volume])
    volume_coefficients=np.linalg.lstsq(features[train],targets[train],rcond=None)[0]
    volume_pred=features@volume_coefficients
    if log_target:volume_pred=np.exp(volume_pred)
    for name,subset in (("train",train),("validation",validation),("test",test)):
        report["baseline"][name]=_errors(actual[subset],np.full(len(subset),actual[train].mean()))
    report["volume_baseline"]={name:_errors(actual[subset],volume_pred[subset]) for name,subset in (("train",train),("validation",validation),("test",test))}
    for degree in ([None] if method=="neural" else degrees):
        if stop_requested():return None
        start=perf_counter()
        if method=="neural":
            offset=float(targets[train].mean());scale=max(float(targets[train].std()),1e-12)
            x=torch.tensor((densities[:,indices]-.5)/.5,dtype=torch.float64)
            y=torch.tensor(((targets-offset)/scale)[:,None],dtype=torch.float64)
            model=train_network(x,y,train,validation,settings=settings,progress_target="design",emit=emit,stop_requested=stop_requested)
            if model is None:return None
            model.update(target="compliance",input_kind="density",input_offset=.5,input_scale=.5,target_offset=offset,target_scale=scale)
            key="neural_network"
        else:
            model=fit_design_polynomial(densities[train][:,indices],targets[train],degree,report["ridge_alpha"])
            model["training_seconds"]=perf_counter()-start;key=f"{method}_{degree}"
        model.update(input_indices=indices.tolist(),target_transform="log" if log_target else "identity",method=method)
        predicted=predict_design(model,densities)
        for name,subset in (("train",train),("validation",validation),("test",test)):
            model[name]=_errors(actual[subset],predicted[subset])
        model["cases"]=[{"index":case["index"],"predicted":float(predicted[i]),"reference":float(actual[i])} for i,case in enumerate(cases)]
        report["models"][key]=model
    if stop_requested():return None
    _save_json(directory/f"models-{method}.json",report)
    return report


def evaluate_design(directory,density,*,stop_requested=lambda:False):
    from .problem import CantileverProblem
    data=read_design_dataset(directory);config=analysis_config(data)
    density=validate_density(density,config)
    if stop_requested():return None
    with CantileverProblem(config) as problem:
        evaluation=problem.evaluate(density,sensitivities=False)
    if stop_requested():return None
    if not math.isfinite(evaluation.compliance) or evaluation.compliance<=0:
        raise ValueError("Structural objective must be positive and finite")
    return {"compliance":evaluation.compliance,"physical_volume":evaluation.volume_fraction,"residual":evaluation.residual,
            "field":"nominal","objective_quantity":data["objective_quantity"]}
