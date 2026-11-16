"""Outer optimisation loop; every history row describes an evaluated design."""

from collections.abc import Callable
from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter

import numpy as np

from .config import CantileverConfig
from .continuation import next_beta
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
    field_compliances: dict[str, float] | None = None
    field_volumes: dict[str, float] | None = None
    worst_field: str | None = None
    epigraph_compliance: float | None = None
    epigraph_constraint: float | None = None
    state_solver: str = "fom"
    rom_basis_rank: int | None = None
    projection_beta: float = 0.0
    max_size_constraint: float | None = None
    max_size_eroded_volume: float | None = None
    max_void_constraint: float | None = None
    max_void_remaining_volume: float | None = None
    volume_error: float | None = None
    load_case_compliances: dict[str, float] | None = None
    active_load_case: str | None = None


@dataclass
class OptimizationResult:
    config: CantileverConfig
    design: np.ndarray
    evaluation: Evaluation
    history: list[HistoryEntry]
    stop_reason: str
    rom_basis: dict | None = None

    @property
    def converged(self) -> bool:
        return self.stop_reason == "design_and_objective_stable"


def optimise(config: CantileverConfig,
             callback: Callable[[HistoryEntry], None] | None = None, *,
             snapshot_callback: Callable[[HistoryEntry, np.ndarray], None] | None = None,
             field_snapshot_callback: Callable[[HistoryEntry, dict[str, np.ndarray]], None] | None = None,
             stop_requested: Callable[[], bool] | None = None,
             state_snapshot_callback: Callable[[HistoryEntry, np.ndarray, np.ndarray], None] | None = None,
             evaluation_callback: Callable[[HistoryEntry, np.ndarray, Evaluation], None] | None = None,
             field_pool=None) -> OptimizationResult:
    """State snapshots copy equilibrium pairs; evaluation callbacks copy design and state."""
    started = perf_counter()
    problem = CantileverProblem(config) if field_pool is None else CantileverProblem(config, field_pool=field_pool)
    with problem:
        return _optimise_problem(problem, callback, snapshot_callback, stop_requested, started,
                                 state_snapshot_callback, field_snapshot_callback, evaluation_callback)


def _optimise_problem(problem, callback, snapshot_callback, stop_requested, started,
                      state_snapshot_callback=None, field_snapshot_callback=None, evaluation_callback=None) -> OptimizationResult:
    config = problem.config
    rom_rank = problem.rom_info["rank"] if problem.rom_info else None
    design = problem.initial_design()
    current = problem.evaluate(design)
    if config.robust:
        current.epigraph_compliance = current.compliance
        current.epigraph_constraint = 0.0
    optimizer = MovingAsymptotes(design, config.optimizer, config.move_limit,
                                 config.max_inner_iterations, robust=config.robust,
                                 max_size=config.max_solid_size > 0, max_void=config.max_void_size > 0,
                                 volume_mode=config.volume_constraint, volume_tolerance=config.feasibility_tolerance,
                                 design_mask=problem.design_mask if config.non_design_regions else None)
    history = [HistoryEntry(0, current.compliance, current.volume_fraction, current.constraint,
                            0.0, 0.0, 0, current.residual, current.linear_iterations,
                            problem.solve_count, elapsed_seconds=perf_counter()-started,
                            field_compliances=current.field_compliances,
                            field_volumes=current.field_volumes, worst_field=current.worst_field,
                            epigraph_compliance=current.epigraph_compliance,
                            epigraph_constraint=current.epigraph_constraint,
                            state_solver=config.state_solver, rom_basis_rank=rom_rank,
                            projection_beta=current.projection_beta,
                            max_size_constraint=current.max_size_constraint,
                            max_size_eroded_volume=current.max_size_eroded_volume,
                            max_void_constraint=current.max_void_constraint,
                            max_void_remaining_volume=current.max_void_remaining_volume,
                            volume_error=current.volume_error,
                            load_case_compliances=current.load_case_compliances, active_load_case=current.active_load_case)]
    if callback:
        callback(history[-1])
    if snapshot_callback:
        snapshot_callback(history[-1], current.density.copy())
    if field_snapshot_callback:
        field_snapshot_callback(history[-1], {name: density.copy() for name, density in
            (current.field_densities or {"nominal": current.density}).items()})
    if state_snapshot_callback:
        state_density = current.field_densities[current.worst_field] if config.robust else current.density
        state_snapshot_callback(history[-1], current.displacement.copy(), state_density.copy())
    if evaluation_callback:
        evaluation_callback(deepcopy(history[-1]), design.copy(), deepcopy(current))
    stable_iterations = 0
    stop_reason = "maximum_iterations"
    for iteration in range(1, config.max_iterations + 1):
        if stop_requested and stop_requested():
            stop_reason = "cancelled"
            break
        beta = next_beta(config, problem.projection_beta, iteration - 1)
        if beta != problem.projection_beta:
            problem.projection_beta = beta
            # A new projection changes values and sensitivities at the same raw
            # design. Re-evaluate before stepping and restart local approximations.
            current = problem.evaluate(design)
            if config.robust:
                current.epigraph_compliance = current.compliance
                current.epigraph_constraint = 0.0
            optimizer = MovingAsymptotes(design, config.optimizer, config.move_limit,
                                         config.max_inner_iterations, robust=config.robust,
                                         max_size=config.max_solid_size > 0, max_void=config.max_void_size > 0,
                                         volume_mode=config.volume_constraint, volume_tolerance=config.feasibility_tolerance,
                                         design_mask=problem.design_mask if config.non_design_regions else None)
            stable_iterations = 0
        step = optimizer.step(design, current, problem.evaluate)
        # Rebuild the PyTorch graph for derivatives. Exact-modulus state caches
        # reuse GCMMA's accepted trial state and energies without another solve.
        updated = problem.evaluate(step.design)
        if config.robust:
            updated.epigraph_compliance = step.epigraph_compliance
            updated.epigraph_constraint = (updated.compliance - step.epigraph_compliance) / optimizer.scale
        change = float(np.max(np.abs(step.design - design)))
        old_objective = current.epigraph_compliance if config.robust else current.compliance
        new_objective = updated.epigraph_compliance if config.robust else updated.compliance
        relative_change = abs(new_objective - old_objective) / max(abs(old_objective), 1e-30)
        history.append(HistoryEntry(iteration, updated.compliance, updated.volume_fraction,
                                     updated.constraint, change, relative_change,
                                     step.inner_iterations, updated.residual, updated.linear_iterations,
                                     problem.solve_count, elapsed_seconds=perf_counter()-started,
                                     field_compliances=updated.field_compliances,
                                     field_volumes=updated.field_volumes, worst_field=updated.worst_field,
                                     epigraph_compliance=updated.epigraph_compliance,
                                     epigraph_constraint=updated.epigraph_constraint,
                                     state_solver=config.state_solver, rom_basis_rank=rom_rank,
                                     projection_beta=updated.projection_beta,
                                     max_size_constraint=updated.max_size_constraint,
                                     max_size_eroded_volume=updated.max_size_eroded_volume,
                                     max_void_constraint=updated.max_void_constraint,
                                     max_void_remaining_volume=updated.max_void_remaining_volume,
                                     volume_error=updated.volume_error,
                                     load_case_compliances=updated.load_case_compliances, active_load_case=updated.active_load_case))
        if callback:
            callback(history[-1])
        if snapshot_callback:
            snapshot_callback(history[-1], updated.density.copy())
        if field_snapshot_callback:
            field_snapshot_callback(history[-1], {name: density.copy() for name, density in
                (updated.field_densities or {"nominal": updated.density}).items()})
        if state_snapshot_callback:
            state_density = updated.field_densities[updated.worst_field] if config.robust else updated.density
            state_snapshot_callback(history[-1], updated.displacement.copy(), state_density.copy())
        if evaluation_callback:
            evaluation_callback(deepcopy(history[-1]), step.design.copy(), deepcopy(updated))
        design, current = step.design, updated
        final_beta = config.projection_schedule == "fixed" or problem.projection_beta >= config.projection_beta_max
        stable = (final_beta and iteration >= config.min_iterations and change <= config.design_tolerance and
                  relative_change <= config.objective_tolerance and
                  current.constraint <= config.feasibility_tolerance and
                  (current.max_size_constraint is None or current.max_size_constraint <= config.feasibility_tolerance) and
                  (current.max_void_constraint is None or current.max_void_constraint <= config.feasibility_tolerance) and
                  (current.epigraph_constraint is None or current.epigraph_constraint <= config.feasibility_tolerance))
        stable_iterations = stable_iterations + 1 if stable else 0
        if stable_iterations >= config.convergence_patience:
            stop_reason = "design_and_objective_stable"
            break
    return OptimizationResult(config, design.copy(), current, history, stop_reason, problem.rom_info)
