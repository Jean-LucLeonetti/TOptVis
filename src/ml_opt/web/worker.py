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
            if configuration.get("task") in ("regression_generate", "regression_fit",
                                               "regression_topology_fit", "regression_topology_evaluate"):
                field_pool.clear()
                run_regression(configuration, _RunMessages(messages, run_id), cancel)
            elif configuration.get("task") in ("design_generate", "design_cross_generate", "design_fit", "design_evaluate"):
                field_pool.clear()
                run_design_regression(configuration, _RunMessages(messages, run_id), cancel)
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
    from ..topology_regression import fit_topology_models, evaluate_topology_prediction
    from ..neural_regression import fit_neural_models, NeuralSettings
    try:
        stopped = lambda: cancel.is_set() or _parent_gone()
        root = Path(configuration["root"])
        dataset_id = messages.run_id if configuration["task"] == "regression_generate" else configuration["dataset_id"]
        directory = root / dataset_id
        if configuration["task"] == "regression_generate":
            result = generate_dataset(RegressionSettings(**configuration["settings"]), directory,
                                      emit=messages.put, stop_requested=stopped)
            cancelled = result["status"] == "cancelled"
        elif configuration["task"] == "regression_topology_evaluate":
            result = evaluate_topology_prediction(directory, **configuration["options"], stop_requested=stopped)
            cancelled = result is None
            if result is not None:
                messages.put({"event": "topology_evaluation", "evaluation": result})
        else:
            # This task owns the worker lock, so an old generating manifest is
            # interrupted, not a concurrently changing dataset.
            options = dict(configuration["options"])
            if options.get("method") == "neural":
                for key in ("method", "ridge_alpha", "degrees"):
                    options.pop(key, None)
                target = "topology" if configuration["task"] == "regression_topology_fit" else "compliance"
                result = fit_neural_models(directory, target=target, **options, emit=messages.put,
                                           stop_requested=stopped, allow_interrupted=True)
            else:
                for key in NeuralSettings.__dataclass_fields__:
                    options.pop(key, None)
                fit = fit_topology_models if configuration["task"] == "regression_topology_fit" else fit_models
                result = fit(directory, **options, stop_requested=stopped, allow_interrupted=True)
            cancelled = result is None
        messages.put({"event": "complete", "dataset_id": dataset_id,
                      "stop_reason": "cancelled" if cancelled else "complete"})
    except Exception as error:
        traceback.print_exc()
        messages.put({"event": "error", "message": str(error) or type(error).__name__})


def run_design_regression(configuration, messages, cancel):
    from ..design_regression import DesignSettings, generate_designs, fit_design_models, evaluate_design
    try:
        stopped = lambda: cancel.is_set() or _parent_gone()
        task = configuration["task"]
        identifier = messages.run_id if task in ("design_generate","design_cross_generate") else configuration["dataset_id"]
        directory = Path(configuration["root"]) / identifier
        if task == "design_cross_generate":
            from ..cross_case_designs import CrossCaseSettings,generate_cross_designs
            result=generate_cross_designs(CantileverConfig(**configuration["configuration"]),directory,
                CrossCaseSettings(**configuration["settings"]),emit=messages.put,stop_requested=stopped)
            cancelled=result["status"]=="cancelled"
        elif task == "design_generate":
            result = generate_designs(configuration["source_directory"], directory,
                DesignSettings(**configuration["settings"]), emit=messages.put, stop_requested=stopped)
            cancelled = result["status"] == "cancelled"
        elif task == "design_fit":
            result = fit_design_models(directory, **configuration["options"], emit=messages.put,
                stop_requested=stopped, allow_interrupted=True)
            cancelled = result is None
        else:
            result = evaluate_design(directory, configuration["density"], stop_requested=stopped)
            cancelled = result is None
            if result is not None:
                messages.put({"event":"design_evaluation", "evaluation":result})
        messages.put({"event":"complete", "dataset_id":identifier,
                      "stop_reason":"cancelled" if cancelled else "complete"})
    except Exception as error:
        traceback.print_exc()
        messages.put({"event":"error", "message":str(error) or type(error).__name__})


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
        def snapshot(entry, design, evaluation):
            fields = evaluation.field_densities or {"nominal": evaluation.density}
            event = {"event": "iteration", "history": asdict(entry), "density": fields["nominal"].tolist()}
            if config.robust:
                event["field_densities"] = {name: density.tolist() for name, density in fields.items()}
            if evaluation.connection_temperatures is not None:
                event["connection_temperatures"] = {key:values.tolist() for key,values in evaluation.connection_temperatures.items()}
                event["connection_connected"] = evaluation.connection_connected
            messages.put(event)
        result = optimise(config, evaluation_callback=snapshot,
                          stop_requested=lambda: cancel.is_set() or _parent_gone(), field_pool=field_pool)
        volume_feasible = result.evaluation.constraint <= config.feasibility_tolerance
        epigraph_feasible = (not config.robust or
            result.evaluation.epigraph_constraint <= config.feasibility_tolerance)
        max_size_feasible = (result.evaluation.max_size_constraint is None or
                            result.evaluation.max_size_constraint <= config.feasibility_tolerance)
        max_void_feasible = (result.evaluation.max_void_constraint is None or
                            result.evaluation.max_void_constraint <= config.feasibility_tolerance)
        connection_feasible = all(value <= config.feasibility_tolerance for value in (result.evaluation.connection_constraints or {}).values())
        messages.put({"event": "complete", "stop_reason": result.stop_reason,
                      "converged": result.converged, "volume_feasible": volume_feasible,
                      "epigraph_feasible": epigraph_feasible,
                      "max_size_feasible": max_size_feasible,
                      "max_void_feasible": max_void_feasible, "connection_feasible": connection_feasible,
                      "state_solver": result.config.state_solver, "rom_basis": result.rom_basis,
                      "feasible": volume_feasible and epigraph_feasible and max_size_feasible and max_void_feasible and connection_feasible})
    except Exception as error:
        traceback.print_exc()
        messages.put({"event": "error", "message": str(error) or type(error).__name__})
