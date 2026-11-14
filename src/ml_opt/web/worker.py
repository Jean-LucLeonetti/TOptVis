"""Run PETSc in a dedicated process and emit actual evaluated iterations."""

from dataclasses import asdict
import multiprocessing as mp
import queue
import traceback
from pathlib import Path

from ..config import CantileverConfig
from ..runner import optimise


class _RunMessages:
    def __init__(self, messages, run_id):
        self.messages, self.run_id = messages, run_id

    def put(self, event):
        self.messages.put((self.run_id, event))


def worker_loop(commands, messages, cancel, ready) -> None:
    ready.set()
    try:
        while not _parent_gone():
            try:
                command = commands.get(timeout=0.25)
            except queue.Empty:
                continue
            if command is None:
                return
            run_id, configuration = command
            # Each command constructs and closes a fresh problem and optimiser.
            task=configuration.get('task','')
            target = run_model_lab if task.startswith('ml_') else run_training if task in ("rom_snapshots", "rom_basis", "rom_validation") else run_optimization
            target(configuration, _RunMessages(messages, run_id), cancel)
    finally:
        if _parent_gone():
            # No reader remains to drain the feeder thread after a forced parent exit.
            messages.cancel_join_thread()


def _parent_gone() -> bool:
    parent = mp.parent_process()
    return parent is not None and not parent.is_alive()


def run_model_lab(configuration,messages,cancel):
    from ..model_lab.config import TrainOptions
    from ..model_lab.training import train_model
    from ..model_lab.experiments import validate_model,run_experiment
    from ..rom.validation import ValidationConfig
    try:
        stopped=lambda:cancel.is_set() or _parent_gone()
        root=Path(configuration['root']); training_root=Path(configuration['training_root'])
        task=configuration['task']
        if task=='ml_train':
            metadata=train_model(training_root/configuration['training_id'],root/messages.run_id,
                TrainOptions(**configuration['options']),emit=messages.put,stop_requested=stopped)
            result=dict(model=metadata,model_id=messages.run_id)
            cancelled=metadata['status']=='cancelled'
        elif task=='ml_validate':
            reports=[]
            for selection in configuration['models']:
                if stopped(): break
                report=validate_model(root/selection['model_id'],training_root/selection['training_id'],
                    ValidationConfig(**configuration['options']),modes=configuration['modes'],
                    validation_id=messages.run_id,expected_sha256=selection['model_sha256'],
                    emit=messages.put,stop_requested=stopped)
                if report is not None:
                    reports.append({key:value for key,value in report.items() if key not in ('results','cases')})
                    messages.put(dict(event='ml_model_validated',report=reports[-1]))
            result=dict(reports=reports); cancelled=stopped()
        else:
            report=run_experiment(root/configuration['model_id'],configuration['options'],
                experiment_id=messages.run_id,expected_sha256=configuration['model_sha256'],
                emit=messages.put,stop_requested=stopped)
            result=dict(experiment=report,model_id=configuration['model_id']); cancelled=report['status']=='cancelled'
        messages.put(dict(event='complete',stop_reason='cancelled' if cancelled else 'complete',**result))
    except Exception as error:
        traceback.print_exc()
        messages.put(dict(event='error',message=str(error) or type(error).__name__))


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


def run_optimization(configuration: dict, messages, cancel) -> None:
    try:
        config = CantileverConfig(**configuration)
        def snapshot(entry, fields):
            event = {"event": "iteration", "history": asdict(entry), "density": fields["nominal"].tolist()}
            if config.robust:
                event["field_densities"] = {name: density.tolist() for name, density in fields.items()}
            messages.put(event)
        result = optimise(config, field_snapshot_callback=snapshot,
                          stop_requested=lambda: cancel.is_set() or _parent_gone())
        volume_feasible = result.evaluation.constraint <= config.feasibility_tolerance
        epigraph_feasible = (not config.robust or
            result.evaluation.epigraph_constraint <= config.feasibility_tolerance)
        max_size_feasible = (result.evaluation.max_size_constraint is None or
                            result.evaluation.max_size_constraint <= config.feasibility_tolerance)
        messages.put({"event": "complete", "stop_reason": result.stop_reason,
                      "converged": result.converged, "volume_feasible": volume_feasible,
                      "epigraph_feasible": epigraph_feasible,
                      "max_size_feasible": max_size_feasible,
                      "state_solver": result.config.state_solver, "rom_basis": result.rom_basis,
                      "feasible": volume_feasible and epigraph_feasible and max_size_feasible})
    except Exception as error:
        traceback.print_exc()
        messages.put({"event": "error", "message": str(error) or type(error).__name__})
