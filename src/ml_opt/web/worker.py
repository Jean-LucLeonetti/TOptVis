"""Run PETSc in a dedicated process and emit actual evaluated iterations."""

from dataclasses import asdict
import multiprocessing as mp
import queue
import traceback
from pathlib import Path

from ..config import CantileverConfig
from ..runner import optimise
from ..fem.worker_pool import FieldWorkerPool
from ..optimizers._mmapy import warmup_optimizer


class _RunMessages:
    def __init__(self, messages, run_id):
        self.messages, self.run_id = messages, run_id

    def put(self, event):
        self.messages.put((self.run_id, event))


def worker_loop(commands, messages, cancel, ready) -> None:
    field_pool = FieldWorkerPool()
    try:
        warmup_optimizer()
        ready.set()
        while not _parent_gone():
            try:
                command = commands.get(timeout=0.25)
            except queue.Empty:
                field_pool.evict_idle()
                continue
            if command is None:
                return
            run_id, configuration = command
            # Each command constructs and closes a fresh problem and optimiser.
            if configuration.get("task") in ("regression_generate", "regression_fit"):
                field_pool.clear()
                run_regression(configuration, _RunMessages(messages, run_id), cancel)
            elif configuration.get("task") in ("rom_snapshots", "rom_basis", "rom_validation"):
                field_pool.clear()
                run_training(configuration, _RunMessages(messages, run_id), cancel)
            else:
                run_optimization(configuration, _RunMessages(messages, run_id), cancel, field_pool=field_pool)
    finally:
        field_pool.close()
        if _parent_gone():
            # No reader remains to drain the feeder thread after a forced parent exit.
            messages.cancel_join_thread()


def _parent_gone() -> bool:
    parent = mp.parent_process()
    return parent is not None and not parent.is_alive()


def run_regression(configuration, messages, cancel):
    from ..regression import RegressionSettings, generate_dataset, fit_models
    try:
        stopped = lambda: cancel.is_set() or _parent_gone()
        root = Path(configuration["root"])
        dataset_id = messages.run_id if configuration["task"] == "regression_generate" else configuration["dataset_id"]
        directory = root / dataset_id
        if configuration["task"] == "regression_generate":
            result = generate_dataset(RegressionSettings(**configuration["settings"]), directory,
                                      emit=messages.put, stop_requested=stopped)
            cancelled = result["status"] == "cancelled"
        else:
            # This task owns the worker lock, so an old generating manifest is
            # interrupted, not a concurrently changing dataset.
            result = fit_models(directory, **configuration["options"], stop_requested=stopped,
                                allow_interrupted=True)
            cancelled = result is None
        messages.put({"event": "complete", "dataset_id": dataset_id,
                      "stop_reason": "cancelled" if cancelled else "complete"})
    except Exception as error:
        traceback.print_exc()
        messages.put({"event": "error", "message": str(error) or type(error).__name__})


def run_training(configuration, messages, cancel):
    from ..rom.training import SnapshotConfig, build_basis, generate_snapshots
    from ..rom.validation import ValidationConfig, validate_basis
    try:
        stopped = lambda: cancel.is_set() or _parent_gone()
        root = Path(configuration["root"])
        if configuration["task"] == "rom_snapshots":
            directory = root / messages.run_id
            metadata = generate_snapshots(SnapshotConfig(**configuration["settings"]), directory,
                                          emit=messages.put, stop_requested=stopped)
        elif configuration["task"] == "rom_validation":
            directory = root / configuration["training_id"]
            metadata = validate_basis(directory, ValidationConfig(**configuration["options"]),
                                       validation_id=messages.run_id, emit=messages.put, stop_requested=stopped)
        else:
            directory = root / configuration["training_id"]
            metadata = build_basis(directory, **configuration["options"],
                                   emit=messages.put, stop_requested=stopped)
        cancelled = metadata is None or metadata["status"] == "cancelled" or (
            configuration["task"] == "rom_validation" and metadata["validation"]["status"] == "cancelled")
        messages.put({"event": "complete", "stop_reason": "cancelled" if cancelled else "complete",
                      "training_id": directory.name, "training": metadata})
    except Exception as error:
        traceback.print_exc()
        messages.put({"event": "error", "message": str(error) or type(error).__name__})


def run_optimization(configuration: dict, messages, cancel, *, field_pool=None) -> None:
    try:
        config = CantileverConfig(**configuration)
        if field_pool is not None and not (config.robust and config.field_execution == "parallel"):
            field_pool.clear()
        def snapshot(entry, fields):
            event = {"event": "iteration", "history": asdict(entry), "density": fields["nominal"].tolist()}
            if config.robust:
                event["field_densities"] = {name: density.tolist() for name, density in fields.items()}
            messages.put(event)
        result = optimise(config, field_snapshot_callback=snapshot,
                          stop_requested=lambda: cancel.is_set() or _parent_gone(), field_pool=field_pool)
        volume_feasible = result.evaluation.constraint <= config.feasibility_tolerance
        epigraph_feasible = (not config.robust or
            result.evaluation.epigraph_constraint <= config.feasibility_tolerance)
        max_size_feasible = (result.evaluation.max_size_constraint is None or
                            result.evaluation.max_size_constraint <= config.feasibility_tolerance)
        max_void_feasible = (result.evaluation.max_void_constraint is None or
                            result.evaluation.max_void_constraint <= config.feasibility_tolerance)
        messages.put({"event": "complete", "stop_reason": result.stop_reason,
                      "converged": result.converged, "volume_feasible": volume_feasible,
                      "epigraph_feasible": epigraph_feasible,
                      "max_size_feasible": max_size_feasible,
                      "max_void_feasible": max_void_feasible,
                      "state_solver": result.config.state_solver, "rom_basis": result.rom_basis,
                      "feasible": volume_feasible and epigraph_feasible and max_size_feasible and max_void_feasible})
    except Exception as error:
        traceback.print_exc()
        messages.put({"event": "error", "message": str(error) or type(error).__name__})
