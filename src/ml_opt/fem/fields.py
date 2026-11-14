"""Independent state solves and a batched first-order PyTorch adjoint."""

import numpy as np
import torch

from .cache import StateCache
from .worker_pool import FieldWorkerPool


class FieldSolves:
    def __init__(self, config, sequential_solver, *, worker_pool=None):
        self.solver = sequential_solver
        self.solve_count = 0
        self.last_states = []
        self.cache = StateCache()
        self._closed = False
        self._pool = self._group = None
        self._owns_pool = False
        if config.field_execution == "parallel":
            self._owns_pool = worker_pool is None
            self._pool = worker_pool if worker_pool is not None else FieldWorkerPool()
            try:
                self._group = self._pool.acquire(config)
            except BaseException:
                if self._owns_pool:
                    self._pool.close()
                raise

    @property
    def _workers(self):
        return self._group.workers if self._group is not None else []

    def solve(self, moduli):
        if self._closed:
            raise RuntimeError("The field solvers are closed")
        moduli = np.asarray(moduli, dtype=np.float64)
        if moduli.shape != (3, self.solver.mesh.element_count):
            raise ValueError("Expected three fields with one modulus per element")
        states = self.cache.get(moduli)
        if states is not None:
            self.last_states = states
            return states
        try:
            states = (self._group.solve(moduli) if self._group is not None else
                      [self.solver.solve(field) for field in moduli])
        except BaseException:
            self.close(healthy=False)
            raise
        self.solve_count += len(states)
        self.cache.put(moduli, states)
        self.last_states = states
        return states

    def close(self, *, healthy=True):
        if self._closed:
            return
        self._closed = True
        self.cache.clear()
        if self._group is not None:
            self._pool.release(self._group, healthy=healthy)
        if self._owns_pool:
            self._pool.close()


class _FieldCompliance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, moduli, fields):
        if moduli.device.type != "cpu" or moduli.dtype != torch.float64:
            raise ValueError("The PETSc bridge expects CPU float64 tensors")
        states = fields.solve(moduli.detach().numpy())
        ctx.save_for_backward(torch.from_numpy(np.stack([s.element_energy for s in states])))
        return moduli.new_tensor([s.compliance for s in states])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        (energies,) = ctx.saved_tensors
        return -gradient[:, None] * energies, None


def field_compliance(moduli, fields):
    return _FieldCompliance.apply(moduli, fields)
