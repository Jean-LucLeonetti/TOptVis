"""A bounded, exclusively leased group of three compatible PETSc workers."""

import multiprocessing as mp
import math
import queue
import time

import torch

from .state_solver import create_state_solver


def _field_worker(config, index, commands, results):
    solver = None
    parent = mp.parent_process()
    try:
        torch.set_num_threads(1)
        solver = create_state_solver(config)
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
        if parent is not None and not parent.is_alive():
            results.cancel_join_thread()
        results.close()
        results.join_thread()


class _WorkerGroup:
    def __init__(self, config):
        context = mp.get_context("spawn")
        self.workers, self.commands = [], []
        self.results = context.Queue()
        self.closed = False
        try:
            for index in range(3):
                commands = context.Queue()
                self.commands.append(commands)
                worker = context.Process(target=_field_worker, args=(config, index, commands, self.results), daemon=True)
                worker.start()
                self.workers.append(worker)
        except BaseException:
            self.close()
            raise

    def alive(self):
        return not self.closed and all(worker.is_alive() for worker in self.workers)

    def solve(self, moduli):
        if not self.alive():
            raise RuntimeError("A robust field solver exited unexpectedly")
        for commands, field in zip(self.commands, moduli):
            commands.put(field)
        states, pending = [None] * 3, set(range(3))
        while pending:
            try:
                index, state, error = self.results.get(timeout=0.25)
            except queue.Empty:
                if not self.alive():
                    raise RuntimeError("A robust field solver exited unexpectedly")
                continue
            if error:
                raise RuntimeError(f"Robust field {index} failed: {error}")
            states[index] = state
            pending.remove(index)
        return states

    def close(self):
        if self.closed:
            return
        self.closed = True
        for commands in self.commands:
            commands.put(None)
        for worker in self.workers:
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
            worker.close()
        for channel in [*self.commands, self.results]:
            channel.close()
            channel.cancel_join_thread()


class FieldWorkerPool:
    """Keep at most one group and expire idle workers after 60 seconds.

    Intended for serial run submission, as enforced by the web run manager.
    Identity covers every state-solver input. Projection and material laws
    stay in the parent and send new moduli. State caches and counts belong
    to each problem, even when its workers persist across runs.
    """

    def __init__(self, idle_timeout=60.0):
        if not math.isfinite(idle_timeout) or idle_timeout < 0:
            raise ValueError("idle_timeout must be finite and nonnegative")
        self.idle_timeout = idle_timeout
        self._group = self._key = None
        self._leased = self._closed = False
        self._last_used = time.monotonic()

    @staticmethod
    def configuration_key(config):
        names = ("nx", "ny", "length", "height", "poisson_ratio", "thickness", "load", "load_angle",
                 "load_node_x", "load_node_y", "linear_tolerance", "linear_solver", "state_solver",
                 "rom_basis_path", "rom_basis_sha256", "boundary_conditions", "load_cases", "load_aggregation")
        return tuple(getattr(config, name) for name in names)

    def acquire(self, config):
        if self._closed:
            raise RuntimeError("The field worker pool is closed")
        if self._leased:
            raise RuntimeError("The field worker pool is already in use")
        self.evict_idle()
        key = self.configuration_key(config)
        if self._group is not None and (key != self._key or not self._group.alive()):
            self.clear()
        if self._group is None:
            self._group = _WorkerGroup(config)
            self._key = key
        self._leased = True
        return self._group

    def release(self, group, *, healthy=True):
        if self._group is not group:
            return
        self._leased = False
        self._last_used = time.monotonic()
        if not healthy:
            self.clear()

    def evict_idle(self):
        if not self._leased and time.monotonic() - self._last_used >= self.idle_timeout:
            self.clear()

    def clear(self):
        if self._leased:
            raise RuntimeError("Cannot retire field workers while a problem is using them")
        if self._group is not None:
            self._group.close()
        self._group = self._key = None

    def close(self):
        if self._closed:
            return
        if self._group is not None:
            self._group.close()
        self._group = self._key = None
        self._leased = False
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
