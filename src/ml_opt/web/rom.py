"""Local offline training API; only server-assigned dataset IDs address files."""

from dataclasses import asdict
from pathlib import Path
import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..rom.training import SnapshotConfig, read_metadata
from .streams import stream_job


class SnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    nx: int = Field(default=30, ge=2, le=160)
    ny: int = Field(default=10, ge=2, le=100)
    length: float = Field(default=3, gt=0, le=1000)
    height: float = Field(default=1, gt=0, le=1000)
    load_heights: list[float] = Field(default_factory=lambda: [.25, .5, .75], min_length=1, max_length=24)
    load_angles: list[float] = Field(default_factory=lambda: [-90], min_length=1, max_length=24)
    volume_fractions: list[float] = Field(default_factory=lambda: [.3, .4, .5], min_length=1, max_length=24)
    iterations_per_case: int = Field(default=20, ge=1, le=300)
    snapshot_stride: int = Field(default=2, ge=1, le=300)
    filter_radius: float = Field(default=.1, ge=0, le=1000)
    projection_beta: float = Field(default=2, ge=0, le=64)
    projection_eta: float = Field(default=.5, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_training_size(self):
        self.configuration()
        return self

    def configuration(self):
        return SnapshotConfig(**self.model_dump())


class BasisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    energy_target: float = Field(default=.999, gt=0, le=1)
    max_modes: int = Field(default=32, ge=1, le=256)
    normalize: bool = True


def create_rom_router(manager, root: Path):
    router = APIRouter(prefix="/api/rom")
    root = root.resolve()

    def directory_for(training_id):
        if not re.fullmatch(r"[0-9a-f]{32}", training_id):
            raise HTTPException(404, "Training dataset not found")
        directory = root / training_id
        if directory.resolve().parent != root:
            raise HTTPException(404, "Training dataset not found")
        return directory

    def metadata_for(training_id):
        try:
            return read_metadata(directory_for(training_id))
        except (OSError, ValueError):
            raise HTTPException(404, "Training dataset not found") from None

    def start_task(task, payload, started):
        try:
            job = manager.start_task(task, {"root": str(root), **payload})
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        return stream_job(manager, job, started(job))

    @router.get("/trainings")
    async def list_trainings():
        records = []
        if root.is_dir():
            for directory in root.iterdir():
                if re.fullmatch(r"[0-9a-f]{32}", directory.name) and directory.is_dir() and not directory.is_symlink():
                    try:
                        record = read_metadata(directory)
                        # A killed worker can leave a manifest, never a usable basis.
                        if record["status"] == "generating" and (manager.active is None or manager.active.id != directory.name):
                            record["status"] = "interrupted"
                        records.append(record)
                    except (OSError, ValueError, KeyError):
                        continue
        return {"trainings": sorted(records, key=lambda x: x["created_at"], reverse=True)}

    @router.post("/snapshots")
    async def snapshots(request: SnapshotRequest):
        settings = asdict(request.configuration())
        return start_task("rom_snapshots", {"settings": settings},
                          lambda job: {"training_id": job.id, "kind": "snapshots", "config": settings})

    @router.post("/{training_id}/basis")
    async def basis(training_id: str, request: BasisRequest):
        metadata = metadata_for(training_id)
        if not metadata.get("snapshot_count") or metadata["status"] not in ("snapshots_ready", "cancelled", "basis_ready"):
            raise HTTPException(409, "Generate a nonempty snapshot dataset before building a basis")
        return start_task("rom_basis", {"training_id": training_id, "options": request.model_dump()},
                          lambda job: {"training_id": training_id, "kind": "basis"})

    @router.get("/{training_id}/artifacts/{artifact}")
    async def download(training_id: str, artifact: Literal["snapshots", "basis", "metadata"]):
        metadata = metadata_for(training_id)
        filename = {"snapshots": "snapshots.npz", "basis": "basis.npz", "metadata": "metadata.json"}[artifact]
        path = directory_for(training_id) / filename
        if not path.is_file() or (artifact == "basis" and "basis" not in metadata):
            raise HTTPException(404, "That training artifact is not available")
        return FileResponse(path, filename=f"rom-{training_id[:8]}-{filename}",
                            media_type="application/json" if artifact == "metadata" else "application/octet-stream")

    return router
