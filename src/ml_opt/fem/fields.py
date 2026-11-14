"""Independent PETSc field solves with a batched first-order PyTorch adjoint.

Only NumPy data crosses process boundaries. Each spawned worker owns its PETSc
objects and retains the matrix graph and symbolic factors between evaluations.
"""

import multiprocessing as mp
import queue

import numpy as np
import torch

from .elasticity import ElasticitySolver
from .mesh import StructuredMesh


def _field_worker(config, index, commands, results):
    solver = None
    parent = mp.parent_process()
    try:
        torch.set_num_threads(1)
        solver = ElasticitySolver(StructuredMesh(config.nx, config.ny, config.length, config.height),
            config.poisson_ratio, config.thickness, config.load, config.linear_tolerance,
            load_angle=config.load_angle, load_node=None if config.load_node_x is None else
            (config.load_node_x, config.load_node_y))
        while parent is None or parent.is_alive():
            try:
                moduli = commands.get(timeout=0.25)
            except queue.Empty:
                continue
            if moduli is None:
                return
            results.put((index, solver.solve(moduli), None))
    except Exception as error:
        results.put((index, None, f"{type(error).__name__}: {error}"))
    finally:
        if solver is not None:
            solver.close()
        # Flush useful error diagnostics unless the parent can no longer read.
        if parent is not None and not parent.is_alive():
            results.cancel_join_thread()
        results.close()
        results.join_thread()


class FieldSolves:
    def __init__(self, config, sequential_solver):
        self.solver = sequential_solver
        self.solve_count = 0
        self.last_states = []
        self._closed = False
        self._workers = []
        self._commands = []
        self._results = None
        if config.field_execution == "parallel":
            context = mp.get_context("spawn")
            self._results = context.Queue()
            try:
                for index in range(3):
                    commands = context.Queue()
                    self._commands.append(commands)
                    worker = context.Process(target=_field_worker,
                        args=(config, index, commands, self._results), daemon=True)
                    worker.start()
                    self._workers.append(worker)
            except BaseException:
                self.close()
                raise

    def solve(self, moduli):
        if self._closed:
            raise RuntimeError("The field solvers are closed")
        moduli = np.asarray(moduli, dtype=np.float64)
        if moduli.shape != (3, self.solver.mesh.element_count):
            raise ValueError("Expected three fields with one modulus per element")
        if not self._workers:
            states = [self.solver.solve(field) for field in moduli]
        else:
            for commands, field in zip(self._commands, moduli):
                commands.put(field)
            states = [None] * 3
            pending = set(range(3))
            try:
                while pending:
                    try:
                        index, state, error = self._results.get(timeout=0.25)
                    except queue.Empty:
                        if any(not worker.is_alive() for worker in self._workers):
                            raise RuntimeError("A robust field solver exited unexpectedly")
                        continue
                    if error:
                        raise RuntimeError(f"Robust field {index} failed: {error}")
                    states[index] = state
                    pending.remove(index)
            except BaseException:
                self.close()
                raise
        self.solve_count += len(states)
        self.last_states = states
        return states

    def close(self):
        if self._closed:
            return
        self._closed = True
        for commands in self._commands:
            commands.put(None)
        for worker in self._workers:
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
            worker.close()
        for channel in [*self._commands, self._results]:
            if channel is not None:
                channel.close()
                channel.cancel_join_thread()


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
