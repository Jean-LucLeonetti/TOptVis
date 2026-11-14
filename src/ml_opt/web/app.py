"""Same-origin static frontend and an iteration-by-iteration JSON stream."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import CantileverConfig
from .jobs import RunManager
from .rom import create_rom_router, selected_basis_path
from ..rom.solver import load_basis
from .streams import stream_job
from .runtime import source_revision
from .model_lab import create_model_router


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    nx: int = Field(default=60, ge=2, le=160)
    ny: int = Field(default=20, ge=2, le=100)
    length: float = Field(default=3.0, gt=0, le=1000)
    height: float = Field(default=1.0, gt=0, le=1000)
    load: float = Field(default=1.0, gt=0, le=1e6)
    load_angle: float = Field(default=-90.0, ge=-180, le=180)
    load_node_x: int = Field(default=60, ge=1)
    load_node_y: int = Field(default=10, ge=0)
    volume_fraction: float = Field(default=0.4, gt=0, lt=1)
    max_solid_size: float = Field(default=0, ge=0, le=1000)
    max_size_beta: float = Field(default=16, gt=0, le=64)
    max_size_tolerance: float = Field(default=.001, gt=0, lt=1)
    max_void_size: float = Field(default=0, ge=0, le=1000)
    max_void_beta: float = Field(default=16, gt=0, le=64)
    max_void_tolerance: float = Field(default=.001, gt=0, lt=1)
    material_model: Literal["simp", "ramp"] = "simp"
    penalisation: float = Field(default=3.0, ge=0)
    filter_radius: float = Field(default=0.1, ge=0, le=1000)
    projection_beta: float = Field(default=2.0, ge=0, le=64)
    projection_eta: float = Field(default=0.5, gt=0, lt=1)
    projection_schedule: Literal["fixed", "geometric"] = "fixed"
    projection_beta_interval: int = Field(default=25, ge=1)
    projection_beta_multiplier: float = Field(default=2.0, gt=1)
    projection_beta_max: float = Field(default=16.0, gt=0, le=64)
    robust: bool = False
    robust_eta_offset: float = Field(default=0.1, gt=0, lt=0.5)
    field_execution: Literal["sequential", "parallel"] = "sequential"
    optimizer: Literal["mma", "gcmma"] = "mma"
    max_iterations: int = Field(default=100, ge=1, le=1000)
    state_solver: Literal["fom", "rom"] = "fom"
    linear_solver: Literal["lu", "cholesky", "cg-icc", "cg-gamg"] = "lu"
    rom_training_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")

    @model_validator(mode="after")
    def valid_node(self):
        if self.load_node_x > self.nx or self.load_node_y > self.ny:
            raise ValueError("The selected load node must be inside the mesh")
        if self.state_solver == "rom" and self.rom_training_id is None:
            raise ValueError("Choose a saved ROM basis for ROM optimisation")
        if self.state_solver == "fom" and self.rom_training_id is not None:
            raise ValueError("A saved ROM selection requires state_solver='rom'")
        # File selection is resolved against this app's training root in /api/run.
        CantileverConfig(**{**self.model_dump(exclude={"rom_training_id"}), "state_solver": "fom"})
        return self

    def configuration(self, *, rom_basis_path=None) -> CantileverConfig:
        return CantileverConfig(**self.model_dump(exclude={"rom_training_id"}), rom_basis_path=rom_basis_path)


def create_app(*, training_root: Path | None = None, model_root: Path | None = None) -> FastAPI:
    training_root = (training_root or Path(__file__).resolve().parents[3] / "outputs" / "rom-training").resolve()
    manager = RunManager()
    # Capture at startup; recomputing in /health would misidentify old imported code.
    revision = source_revision()
    @asynccontextmanager
    async def lifespan(app):
        manager.warmup()
        try:
            yield
        finally:
            await asyncio.to_thread(manager.shutdown)

    app = FastAPI(title="TOptVis", lifespan=lifespan)
    app.state.manager = manager
    app.include_router(create_rom_router(manager, training_root))
    app.include_router(create_model_router(manager,model_root or training_root.parent/'model-lab',training_root))
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    async def health():
        return {"status": "ready", "server_revision": revision, "running": manager.active is not None,
                "worker_ready": manager.worker_ready, "projection_supported": True,
                "robust_supported": True,
                "material_schedule_supported": True,
                "max_size_supported": True,
                "max_void_supported": True,
                "rom_training_supported": True,
                "rom_validation_supported": True,
                "rom_optimisation_supported": True,
                "model_lab_supported": True,
                "linear_solver_supported": True,
                "active_kind": manager.active.kind if manager.active else None}

    @app.post("/api/run")
    async def run(request: RunRequest):
        try:
            if request.state_solver == "rom":
                path = selected_basis_path(training_root, request.rom_training_id)
                config = request.configuration(rom_basis_path=str(path))
                _, info = load_basis(path, config)
                if info["training_id"] != request.rom_training_id:
                    raise ValueError("The ROM artifact does not belong to the selected training dataset")
                config = replace(config, rom_basis_sha256=info["basis_sha256"])
            else:
                config = request.configuration()
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        try:
            job = manager.start(config)
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

        return stream_job(manager, job, {"config": asdict(config)})

    @app.post("/api/run/{run_id}/stop")
    async def stop(run_id: str):
        if not manager.stop(run_id):
            raise HTTPException(404, "That run is no longer active")
        return {"status": "stopping"}

    return app


app = create_app()
