"""Run PETSc in a dedicated process and emit actual evaluated iterations."""

from dataclasses import asdict
import traceback

from ..config import CantileverConfig
from ..runner import optimise


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
