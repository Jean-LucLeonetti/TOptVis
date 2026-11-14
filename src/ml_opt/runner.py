"""Outer optimisation loop; every history row describes an evaluated design."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .config import CantileverConfig
from .optimizers import MovingAsymptotes
from .problem import CantileverProblem, Evaluation


@dataclass
class HistoryEntry:
    iteration: int
    compliance: float
    volume_fraction: float
    constraint: float
    design_change: float
    relative_objective_change: float
    inner_iterations: int
    linear_residual: float
    linear_iterations: int
    state_solves: int
    elapsed_seconds: float = 0.0


@dataclass
class OptimizationResult:
    config: CantileverConfig
    design: np.ndarray
    evaluation: Evaluation
    history: list[HistoryEntry]
    stop_reason: str

    @property
    def converged(self) -> bool:
        return self.stop_reason == "design_and_objective_stable"


def optimise(config: CantileverConfig,
             callback: Callable[[HistoryEntry], None] | None = None, *,
             snapshot_callback: Callable[[HistoryEntry, np.ndarray], None] | None = None,
             stop_requested: Callable[[], bool] | None = None) -> OptimizationResult:
    started = perf_counter()
    with CantileverProblem(config) as problem:
        return _optimise_problem(problem, callback, snapshot_callback, stop_requested, started)


def _optimise_problem(problem, callback, snapshot_callback, stop_requested, started) -> OptimizationResult:
    config = problem.config
    design = problem.initial_design()
    current = problem.evaluate(design)
    optimizer = MovingAsymptotes(design, config.optimizer, config.move_limit,
                                 config.max_inner_iterations)
    history = [HistoryEntry(0, current.compliance, current.volume_fraction, current.constraint,
                            0.0, 0.0, 0, current.residual, current.linear_iterations,
                            problem.solver.solve_count, elapsed_seconds=perf_counter()-started)]
    if callback:
        callback(history[-1])
    if snapshot_callback:
        snapshot_callback(history[-1], current.density.copy())
    stable_iterations = 0
    stop_reason = "maximum_iterations"
    for iteration in range(1, config.max_iterations + 1):
        if stop_requested and stop_requested():
            stop_reason = "cancelled"
            break
        step = optimizer.step(design, current, problem.evaluate)
        # PERF B4: GCMMA just solved this accepted trial without sensitivities;
        # this evaluation repeats its state solve. See benchmarks/RESULTS.md.
        updated = problem.evaluate(step.design)
        change = float(np.max(np.abs(step.design - design)))
        relative_change = abs(updated.compliance - current.compliance) / max(abs(current.compliance), 1e-30)
        history.append(HistoryEntry(iteration, updated.compliance, updated.volume_fraction,
                                     updated.constraint, change, relative_change,
                                     step.inner_iterations, updated.residual, updated.linear_iterations,
                                     problem.solver.solve_count, elapsed_seconds=perf_counter()-started))
        if callback:
            callback(history[-1])
        if snapshot_callback:
            snapshot_callback(history[-1], updated.density.copy())
        design, current = step.design, updated
        stable = (iteration >= config.min_iterations and change <= config.design_tolerance and
                  relative_change <= config.objective_tolerance and
                  current.constraint <= config.feasibility_tolerance)
        stable_iterations = stable_iterations + 1 if stable else 0
        if stable_iterations >= config.convergence_patience:
            stop_reason = "design_and_objective_stable"
            break
    return OptimizationResult(config, design.copy(), current, history, stop_reason)
