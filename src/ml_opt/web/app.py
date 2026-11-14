"""Same-origin static frontend and an iteration-by-iteration JSON stream."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
from pathlib import Path
import queue
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import CantileverConfig
from .jobs import RunManager


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
    optimizer: Literal["mma", "gcmma"] = "mma"
    max_iterations: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def valid_node(self):
        if self.load_node_x > self.nx or self.load_node_y > self.ny:
            raise ValueError("The selected load node must be inside the mesh")
        return self

    def configuration(self) -> CantileverConfig:
        return CantileverConfig(**self.model_dump())


def create_app() -> FastAPI:
    manager = RunManager()
    @asynccontextmanager
    async def lifespan(app):
        manager.warmup()
        try:
            yield
        finally:
            await asyncio.to_thread(manager.shutdown)

    app = FastAPI(title="TOptVis", lifespan=lifespan)
    app.state.manager = manager
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    async def health():
        return {"status": "ready", "running": manager.active is not None,
                "worker_ready": manager.worker_ready}

    @app.post("/api/run")
    async def run(request: RunRequest):
        config = request.configuration()
        try:
            job = manager.start(config)
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

        async def events():
            try:
                yield json.dumps({"event": "started", "run_id": job.id,
                                  "config": asdict(config)}) + "\n"
                while True:
                    try:
                        event = await asyncio.to_thread(job.messages.get, True, 0.25)
                    except queue.Empty:
                        if not job.process.is_alive():
                            yield json.dumps({"event": "error", "message": "The solver process exited unexpectedly."}) + "\n"
                            break
                        continue
                    yield json.dumps(event, allow_nan=False) + "\n"
                    if event["event"] in ("complete", "error"):
                        break
            finally:
                job.cancel.set()
                await asyncio.shield(asyncio.to_thread(manager.finish, job))

        return StreamingResponse(events(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/run/{run_id}/stop")
    async def stop(run_id: str):
        if not manager.stop(run_id):
            raise HTTPException(404, "That run is no longer active")
        return {"status": "stopping"}

    return app


app = create_app()
