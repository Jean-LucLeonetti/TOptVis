"""Run PETSc in a dedicated process and emit actual evaluated iterations."""

from dataclasses import asdict
import multiprocessing as mp
import queue
import traceback

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
            run_optimization(configuration, _RunMessages(messages, run_id), cancel)
    finally:
        if _parent_gone():
            # No reader remains to drain the feeder thread after a forced parent exit.
            messages.cancel_join_thread()


def _parent_gone() -> bool:
    parent = mp.parent_process()
    return parent is not None and not parent.is_alive()


def run_optimization(configuration: dict, messages, cancel) -> None:
    try:
        config = CantileverConfig(**configuration)
        def snapshot(entry, density):
            messages.put({"event": "iteration", "history": asdict(entry), "density": density.tolist()})
        result = optimise(config, snapshot_callback=snapshot,
                          stop_requested=lambda: cancel.is_set() or _parent_gone())
        messages.put({"event": "complete", "stop_reason": result.stop_reason,
                      "converged": result.converged,
                      "feasible": result.evaluation.constraint <= config.feasibility_tolerance})
    except Exception as error:
        traceback.print_exc()
        messages.put({"event": "error", "message": str(error) or type(error).__name__})
