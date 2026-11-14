"""Run PETSc in a dedicated process and emit actual evaluated iterations."""

from dataclasses import asdict
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
    while (command := commands.get()) is not None:
        run_id, configuration = command
        # Each command constructs and closes a fresh problem and optimiser.
        run_optimization(configuration, _RunMessages(messages, run_id), cancel)


def run_optimization(configuration: dict, messages, cancel) -> None:
    try:
        config = CantileverConfig(**configuration)
        def snapshot(entry, density):
            messages.put({"event": "iteration", "history": asdict(entry), "density": density.tolist()})
        result = optimise(config, snapshot_callback=snapshot, stop_requested=cancel.is_set)
        messages.put({"event": "complete", "stop_reason": result.stop_reason,
                      "converged": result.converged,
                      "feasible": result.evaluation.constraint <= config.feasibility_tolerance})
    except Exception as error:
        traceback.print_exc()
        messages.put({"event": "error", "message": str(error) or type(error).__name__})
