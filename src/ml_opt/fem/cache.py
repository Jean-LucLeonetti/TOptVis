"""One-entry exact-modulus state caching, independent of the autograd graph."""

import numpy as np

from .elasticity import ElasticityState


def copy_state(state):
    return ElasticityState(state.displacement.copy(), state.compliance, state.element_energy.copy(),
                           state.relative_residual, state.iterations,
                           None if state.case_compliances is None else dict(state.case_compliances),
                           state.active_case, state.state_solves)


class StateCache:
    def __init__(self):
        self.clear()
        self.hits = 0

    def clear(self):
        self._moduli = self._states = None

    def get(self, moduli):
        if self._moduli is None or not np.array_equal(self._moduli, moduli):
            return None
        self.hits += 1
        return [copy_state(state) for state in self._states]

    def put(self, moduli, states):
        self._moduli = np.array(moduli, copy=True)
        self._states = [copy_state(state) for state in states]


class CachedStateSolver:
    """Per-problem adapter; the underlying solver retains its physical solve count."""

    def __init__(self, solver):
        self.solver = solver
        self.cache = StateCache()
        self.last_state = None

    def solve(self, moduli):
        if getattr(self.solver, "_closed", False):
            raise RuntimeError("The state solver is closed")
        states = self.cache.get(moduli)
        if states is None:
            self.last_state = self.solver.solve(moduli)
            self.cache.put(moduli, [self.last_state])
        else:
            self.last_state = states[0]
        return self.last_state
