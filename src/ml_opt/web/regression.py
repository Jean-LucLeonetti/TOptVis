"""Regression experiments using the existing cancellable solver worker."""

from dataclasses import asdict, replace
from pathlib import Path
import re
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..regression import (RegressionSettings, dataset_summary, read_dataset, read_models,
                          split_cases, validate_degrees)
from ..topology_regression import read_topology_models, topology_cases
from .rom import training_directory
from .streams import stream_job


class RegressionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    nx: int = Field(default=30, ge=1, strict=True)
    ny: int = Field(default=10, ge=1, strict=True)
    length: float = Field(default=3, gt=0)
    height: float = Field(default=1, gt=0)
    load_height: float = Field(default=0.5, ge=0, le=1)
    load_angle: float = Field(default=-90, ge=-180, le=180)
    volume_min: float = Field(default=0.25, gt=0, lt=1)
    volume_max: float = Field(default=0.65, gt=0, lt=1)
    case_count: int = Field(default=20, ge=6, strict=True)
    max_iterations: int = Field(default=200, ge=1, strict=True)
    filter_radius: float = Field(default=0.1, ge=0)
    projection_beta: float = Field(default=2, ge=0)
    projection_eta: float = Field(default=0.5, gt=0, lt=1)
    material_model: Literal["simp", "ramp"] = "simp"
    penalisation: float = Field(default=3, ge=0)
    problem: dict | None = None
    problem_name: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def valid_configuration(self):
        self.configuration()
        return self

    def configuration(self):
        # The shared run validator resolves imported models in the router.
        return RegressionSettings(**self.model_dump(exclude={"problem"}))


class FitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    test_fraction: float = Field(default=0.25, ge=0.1, le=0.5)
    seed: int = Field(default=42, ge=0, le=2**32-1, strict=True)
    include_unconverged: bool = Field(default=False, strict=True)
    degrees: list[Annotated[int, Field(strict=True, ge=1, le=10)]] = Field(
        default_factory=lambda: [1, 2], min_length=1, max_length=10)

    @model_validator(mode="after")
    def valid_degrees(self):
        validate_degrees(self.degrees, 11)
        return self


class TopologyPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    degree: int = Field(ge=1, le=10, strict=True)
    volume: float = Field(gt=0, lt=1)


def create_regression_router(manager, root: Path, *, resolve_problem):
    router = APIRouter(prefix="/api/regression")
    root = root.resolve()

    def read(dataset_id):
        directory = training_directory(root, dataset_id)
        try:
            data = read_dataset(directory)
        except (OSError, ValueError):
            raise HTTPException(404, "Regression dataset not found") from None
        return directory, data

    def start(task, payload, started):
        try:
            job = manager.start_task(task, {"root": str(root), **payload})
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        return stream_job(manager, job, started(job))

    @router.get("/datasets")
    async def datasets():
        records = []
        if root.is_dir():
            for directory in root.iterdir():
                if not re.fullmatch(r"[0-9a-f]{32}", directory.name) or not directory.is_dir() or directory.is_symlink():
                    continue
                try:
                    record = dataset_summary(read_dataset(directory))
                    if record["status"] == "generating" and (manager.active is None or manager.active.id != record["id"]):
                        record["status"] = "interrupted"
                    record["has_models"] = (directory / "models.json").is_file()
                    record["has_topology_models"] = (directory / "topology-models.json").is_file()
                    records.append(record)
                except (OSError, ValueError, KeyError):
                    continue
        return {"datasets": sorted(records, key=lambda record: record["created_at"], reverse=True)}

    @router.post("/generate")
    async def generate(request: RegressionRequest):
        configuration = request.configuration()
        if request.problem is not None:
            try:
                source, problem = resolve_problem(request.problem)
                configuration = replace(configuration, problem=asdict(problem), source_problem=source)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
        settings = asdict(configuration)
        return start("regression_generate", {"settings": settings},
                     lambda job: {"dataset_id": job.id, "kind": "regression_generate", "settings": settings})

    @router.get("/{dataset_id}")
    async def dataset(dataset_id: str):
        directory, data = read(dataset_id)
        try:
            models = read_models(directory)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        if data["status"] == "generating" and (manager.active is None or manager.active.id != dataset_id):
            data["status"] = "interrupted"
        try:
            topology_models, topology_error = read_topology_models(directory), None
        except ValueError as error:
            topology_models, topology_error = None, str(error)
        return {"dataset": data, "models": models, "topology_models": topology_models,
                "topology_model_error": topology_error}

    @router.post("/{dataset_id}/fit")
    async def fit(dataset_id: str, request: FitRequest):
        _, data = read(dataset_id)
        if manager.active is not None and manager.active.id == dataset_id:
            raise HTTPException(409, "Wait for dataset generation to finish before training")
        try:
            _, train, _ = split_cases(data, **request.model_dump(exclude={"degrees"}))
            validate_degrees(request.degrees, len(train))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return start("regression_fit", {"dataset_id": dataset_id, "options": request.model_dump()},
                     lambda job: {"dataset_id": dataset_id, "kind": "regression_fit"})

    @router.post("/{dataset_id}/topology/fit")
    async def fit_topology(dataset_id: str, request: FitRequest):
        _, data = read(dataset_id)
        if manager.active is not None and manager.active.id == dataset_id:
            raise HTTPException(409, "Wait for dataset generation to finish before training")
        try:
            _, train, _, _, _ = topology_cases(data, **request.model_dump(exclude={"degrees"}))
            validate_degrees(request.degrees, len(train))
        except (ValueError, KeyError) as error:
            raise HTTPException(422, str(error)) from error
        return start("regression_topology_fit", {"dataset_id": dataset_id, "options": request.model_dump()},
                     lambda job: {"dataset_id": dataset_id, "kind": "regression_topology_fit"})

    @router.post("/{dataset_id}/topology/evaluate")
    async def evaluate_topology(dataset_id: str, request: TopologyPredictionRequest):
        directory, _ = read(dataset_id)
        try:
            report = read_topology_models(directory)
            if report is None or str(request.degree) not in report["models"]:
                raise ValueError("Train the selected topology model first")
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return start("regression_topology_evaluate", {"dataset_id": dataset_id, "options": request.model_dump()},
                     lambda job: {"dataset_id": dataset_id, "kind": "regression_topology_evaluate"})

    @router.get("/{dataset_id}/artifacts/{kind}")
    async def artifact(dataset_id: str, kind: Literal["dataset", "models", "topology-models"]):
        directory, _ = read(dataset_id)
        path = directory / f"{kind}.json"
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "Regression artifact not found")
        return FileResponse(path, media_type="application/json", filename=f"regression-{dataset_id[:8]}-{kind}.json")

    return router
