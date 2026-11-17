"""Fixed-setup physical design to compliance experiments."""

from pathlib import Path
import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..design_regression import (DesignSettings, read_design_dataset, read_design_models,
    source_designs, split_design_cases, validate_density, analysis_config)
from ..regression import read_dataset
from .regression import FitRequest
from .rom import training_directory
from .streams import stream_job


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    variants: int = Field(default=4, ge=0, strict=True)
    amplitude: float = Field(default=.2, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**32-1, strict=True)


class DesignFitRequest(FitRequest):
    learning_rate: float = Field(default=.0001, gt=0, strict=True)
    log_target: bool = Field(default=True, strict=True)


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    density: list[float] = Field(min_length=1)


class CrossCaseRequest(BaseModel):
    original_dataset_id: str | None = Field(default=None,pattern=r"^[0-9a-f]{32}$")
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
    source_id: str | None = Field(default=None,pattern=r"^[0-9a-f]{32}$")
    problem: dict | None = None
    case_count: int = Field(default=12,ge=3,strict=True)
    max_iterations: int = Field(default=150,ge=1,le=1000,strict=True)
    variants: int = Field(default=4,ge=0,strict=True)
    amplitude: float = Field(default=.2,ge=0,le=1)
    seed: int = Field(default=42,ge=0,le=2**32-1,strict=True)

    @model_validator(mode="after")
    def source(self):
        if sum(value is not None for value in (self.source_id,self.problem,self.original_dataset_id))!=1:
            raise ValueError("Select a saved original case or use the current optimisation setup")
        return self


def create_design_regression_router(manager, root: Path, source_root: Path, *, resolve_problem):
    router = APIRouter(prefix="/api/design-regression")
    root, source_root = root.resolve(), source_root.resolve()

    def read(identifier):
        directory = training_directory(root, identifier)
        try:
            return directory, read_design_dataset(directory)
        except (OSError, ValueError):
            raise HTTPException(404, "Design dataset not found") from None

    def start(task, payload, dataset_id=None):
        try:
            job = manager.start_task(task, {"root":str(root), **payload})
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        return stream_job(manager, job, {"dataset_id":dataset_id or job.id,"kind":task})

    @router.get("/datasets")
    async def datasets():
        records=[]
        if root.is_dir():
            for directory in root.iterdir():
                if not re.fullmatch(r"[0-9a-f]{32}",directory.name) or not directory.is_dir() or directory.is_symlink():
                    continue
                try:
                    data=read_design_dataset(directory)
                    record={key:data[key] for key in ("id","created_at","status","family_count","expected_cases")}
                    record["case_count"]=len(data["cases"])
                    if record["status"]=="generating" and (manager.active is None or manager.active.id!=record["id"]):
                        record["status"]="interrupted"
                    records.append(record)
                except (OSError, ValueError, KeyError):
                    continue
        return {"datasets":sorted(records,key=lambda row:row["created_at"],reverse=True)}

    @router.post("/generate")
    async def generate(request: GenerateRequest):
        source=training_directory(source_root,request.source_id)
        try:
            data=read_dataset(source)
        except (OSError, ValueError):
            raise HTTPException(404,"Source dataset not found") from None
        try:
            source_designs(data)
            settings=vars(DesignSettings(**request.model_dump(exclude={"source_id"})))
        except (ValueError, OSError, KeyError) as error:
            raise HTTPException(422,str(error)) from error
        if manager.active is not None and manager.active.id==request.source_id:
            raise HTTPException(409,"Wait for source design generation to finish")
        return start("design_generate",{"source_directory":str(source),"settings":settings})

    @router.post("/cross-cases")
    async def cross_cases(request: CrossCaseRequest):
        from ..config import CantileverConfig
        from ..cross_case_designs import CrossCaseSettings, alternative_cases
        from dataclasses import asdict
        try:
            if request.problem is not None:
                _,original=resolve_problem(request.problem)
            elif request.original_dataset_id is not None:
                _,saved=read(request.original_dataset_id)
                if "source_config" not in saved:
                    raise ValueError("This experiment has no saved original optimisation setup")
                original=CantileverConfig(**saved["source_config"])
            else:
                source=training_directory(source_root,request.source_id)
                original=CantileverConfig(**read_dataset(source)["fixed_config"])
            settings=CrossCaseSettings(**request.model_dump(exclude={"source_id","problem","original_dataset_id"}))
            alternative_cases(original,settings)
        except (ValueError,OSError) as error:
            raise HTTPException(422,str(error)) from error
        return start("design_cross_generate",{"configuration":asdict(original),"settings":asdict(settings)})

    @router.get("/{dataset_id}")
    async def dataset(dataset_id: str):
        directory,data=read(dataset_id)
        if data["status"]=="generating" and (manager.active is None or manager.active.id!=dataset_id):
            data["status"]="interrupted"
        reports,errors={},{}
        for method in ("ols","ridge","neural"):
            try:
                reports[method]=read_design_models(directory,method)
            except ValueError as error:
                errors[method]=str(error)
        return {"dataset":data,"reports":reports,"errors":errors}

    @router.post("/{dataset_id}/fit")
    async def fit(dataset_id: str, request: DesignFitRequest):
        _,data=read(dataset_id)
        if manager.active is not None and manager.active.id==dataset_id:
            raise HTTPException(409,"Wait for dataset generation to finish")
        try:
            split_design_cases(data,test_fraction=request.test_fraction,validation_fraction=request.validation_fraction,seed=request.seed)
        except ValueError as error:
            raise HTTPException(422,str(error)) from error
        options=request.model_dump(exclude={"include_unconverged"})
        return start("design_fit",{"dataset_id":dataset_id,"options":options},dataset_id)

    @router.post("/{dataset_id}/evaluate")
    async def evaluate(dataset_id: str, request: EvaluateRequest):
        _,data=read(dataset_id)
        try:
            density=validate_density(request.density,analysis_config(data)).tolist()
        except ValueError as error:
            raise HTTPException(422,str(error)) from error
        return start("design_evaluate",{"dataset_id":dataset_id,"density":density},dataset_id)

    @router.get("/{dataset_id}/artifacts/{kind}")
    async def artifact(dataset_id: str,kind: Literal["dataset","models-ols","models-ridge","models-neural"]):
        directory,_=read(dataset_id)
        path=directory/f"{kind}.json"
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404,"Design regression artifact not found")
        return FileResponse(path,media_type="application/json",filename=f"design-{dataset_id[:8]}-{kind}.json")

    return router
