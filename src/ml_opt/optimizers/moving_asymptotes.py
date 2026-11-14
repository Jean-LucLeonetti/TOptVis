"""Stateful moving-asymptote methods for one volume inequality.

The numerical subproblem routines come from mmapy (GPL-3.0-or-later).
GCMMA accepts a trial only after checking its conservative approximations.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from mmapy import asymp, concheck, raaupdate

from ._mmapy import gcmmasub, mmasub

from ..problem import Evaluation


@dataclass
class OptimizerStep:
    design: np.ndarray
    inner_iterations: int = 0


class MovingAsymptotes:
    def __init__(self, initial_design: np.ndarray, method: str = "mma",
                 move_limit: float = 0.2, max_inner_iterations: int = 30):
        if method not in ("mma", "gcmma"):
            raise ValueError("Unknown moving-asymptote method")
        if not 0 < move_limit <= 1 or max_inner_iterations < 1:
            raise ValueError("Invalid move limit or inner iteration limit")
        self.method = method
        self.move = move_limit
        self.max_inner_iterations = max_inner_iterations
        self.n = len(initial_design)
        self.xold1 = np.asarray(initial_design, dtype=float).reshape(-1, 1).copy()
        self.xold2 = self.xold1.copy()
        self.xmin, self.xmax = np.zeros_like(self.xold1), np.ones_like(self.xold1)
        self.low, self.upp = self.xmin.copy(), self.xmax.copy()
        self.iteration = 0
        self.scale: float | None = None
        self.raa0 = 0.01
        self.raa = np.full((1, 1), 0.01)
        self.raa_floor = np.full((1, 1), 1e-6)
        self.epsilon = 1e-7
        # No artificial z coupling; a large slack penalty keeps the volume constraint active.
        self.a = np.zeros((1, 1))
        self.c = np.full((1, 1), 1000.0)
        self.d = np.ones((1, 1))

    def step(self, design: np.ndarray, current: Evaluation,
             evaluate: Callable[[np.ndarray, bool], Evaluation]) -> OptimizerStep:
        if current.objective_gradient is None or current.constraint_gradient is None:
            raise ValueError("Moving asymptotes requires objective and constraint gradients")
        x = np.asarray(design, dtype=float).reshape(-1, 1)
        if self.scale is None:
            self.scale = current.compliance
            if not np.isfinite(self.scale) or self.scale <= 0:
                raise ValueError("Initial compliance must be finite and positive")
        self.iteration += 1
        objective = current.compliance / self.scale
        objective_gradient = current.objective_gradient.reshape(-1, 1) / self.scale
        constraint = np.array([[current.constraint]])
        constraint_gradient = current.constraint_gradient.reshape(1, -1)
        inner_iterations = 0
        # PERF B2: retain mmapy's subproblem algorithm, with diagonal row
        # scaling performed by broadcasting instead of allocating sparse matrices.
        if self.method == "mma":
            result = mmasub(1, self.n, self.iteration, x, self.xmin, self.xmax,
                            self.xold1, self.xold2, objective, objective_gradient,
                            constraint, constraint_gradient, self.low, self.upp,
                            1.0, self.a, self.c, self.d, move=self.move)
            candidate, self.low, self.upp = result[0], result[-2], result[-1]
        else:
            self.low, self.upp, self.raa0, self.raa = asymp(
                self.iteration, self.n, x, self.xold1, self.xold2, self.xmin,
                self.xmax, self.low, self.upp, self.raa0, self.raa,
                1e-6, self.raa_floor, objective_gradient, constraint_gradient)
            # Apply move limits to each subproblem, consistently in its curvature update.
            lower, upper = np.maximum(self.xmin, x - self.move), np.minimum(self.xmax, x + self.move)
            for inner_iterations in range(self.max_inner_iterations + 1):
                result = gcmmasub(
                    1, self.n, self.iteration, self.epsilon, x, lower, upper,
                    self.low, self.upp, self.raa0, self.raa, objective,
                    objective_gradient, constraint, constraint_gradient,
                    1.0, self.a, self.c, self.d)
                candidate, approximation, constraint_approximation = result[0], result[-2], result[-1]
                trial = evaluate(candidate.ravel(), False)
                trial_objective = np.array([[trial.compliance / self.scale]])
                trial_constraint = np.array([[trial.constraint]])
                if concheck(1, self.epsilon, approximation, trial_objective,
                            constraint_approximation, trial_constraint):
                    break
                if inner_iterations == self.max_inner_iterations:
                    raise RuntimeError("GCMMA failed to find a conservative trial within the inner iteration limit")
                self.raa0, self.raa = raaupdate(
                    candidate, x, lower, upper, self.low, self.upp,
                    trial_objective, trial_constraint, approximation,
                    constraint_approximation, self.raa0, self.raa,
                    1e-6, self.raa_floor, self.epsilon)
        candidate = candidate.ravel()
        if not np.all(np.isfinite(candidate)) or np.any(candidate < 0) or np.any(candidate > 1):
            raise RuntimeError("Optimizer returned invalid design densities")
        self.xold2, self.xold1 = self.xold1.copy(), x.copy()
        return OptimizerStep(candidate.copy(), inner_iterations)
