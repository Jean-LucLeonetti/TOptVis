"""Differentiable cell-centred pseudo-heat diffusion using PETSc.

Every target supplies unit total heat independently. Anchor cells are held at
zero temperature; the remaining exterior faces are insulated. Harmonic face
conductivity suppresses spurious conduction across low-density cuts.
"""

import numpy as np
from petsc4py import PETSc
import torch

from ..connectivity import connection_regions, connected_targets


class ConnectionDiffusion:
    def __init__(self, config):
        self.config = config
        self.anchors, self.targets = connection_regions(config.connection_anchors, config.connection_targets, config.nx, config.ny)
        cells = np.arange(config.nx*config.ny).reshape(config.ny, config.nx)
        self.left = np.concatenate([cells[:, :-1].ravel(), cells[:-1, :].ravel()])
        self.right = np.concatenate([cells[:, 1:].ravel(), cells[1:, :].ravel()])
        dx, dy = config.length/config.nx, config.height/config.ny
        self.geometry = torch.tensor(np.concatenate([np.full(config.ny*(config.nx-1), dy/dx),
            np.full((config.ny-1)*config.nx, dx/dy)]), dtype=torch.float64)
        self.loads = np.zeros((len(self.targets), cells.size))
        for row, source in zip(self.loads, self.targets):
            row[source] = 1/len(source)
        self.matrix = self.solver = self.rhs = self.solution = None
        self.last_conductance = None
        self.solve_count = 0
        self.last_temperatures = None
        self._closed = False
        try:
            self.matrix = PETSc.Mat().create(comm=PETSc.COMM_SELF)
            self.matrix.setSizes((cells.size, cells.size))
            self.matrix.setType("aij")
            row = np.concatenate([self.left, self.right, self.left, self.right, np.arange(cells.size)])
            col = np.concatenate([self.left, self.right, self.right, self.left, np.arange(cells.size)])
            self.matrix.setPreallocationCOO(row.astype(PETSc.IntType), col.astype(PETSc.IntType))
            self.matrix.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
            self.matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            self.matrix.setOption(PETSc.Mat.Option.SPD, True)
            self.solution, self.rhs = self.matrix.createVecs()
            self.solver = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            self.solver.setType("preonly")
            self.solver.getPC().setType("lu")
            self.solver.setErrorIfNotConverged(True)
            self.solver.setOperators(self.matrix)
            solid_conductance = self.geometry.numpy()
            reference = self.solve(solid_conductance, self.loads)
            self.reference = np.array([np.linalg.norm(t[source], ord=8) for t, source in zip(reference, self.targets)])
            if not np.all(np.isfinite(self.reference)) or np.any(self.reference <= 0):
                raise ValueError("Connection regions do not define a positive diffusion reference")
        except BaseException:
            self.close()
            raise

    def solve(self, conductance, loads):
        if self._closed:
            raise RuntimeError("The connection diffusion solver is closed")
        if self.last_conductance is None or not np.array_equal(conductance, self.last_conductance):
            values = np.concatenate([conductance, conductance, -conductance, -conductance,
                np.zeros(self.config.nx*self.config.ny)])
            self.matrix.setValuesCOO(values, PETSc.InsertMode.INSERT_VALUES)
            self.matrix.zeroRowsColumns(self.anchors.astype(PETSc.IntType), diag=1.)
            self.last_conductance = np.array(conductance, copy=True)
        temperatures = np.zeros_like(loads)
        for result, load in zip(temperatures, loads):
            if not np.any(load):
                continue
            self.rhs.array[:] = load
            self.rhs.array[self.anchors] = 0.
            self.solver.solve(self.rhs, self.solution)
            result[:] = self.solution.array
            if self.solver.getConvergedReason() <= 0 or not np.all(np.isfinite(result)):
                raise RuntimeError("PETSc failed to solve connection diffusion")
            self.solve_count += 1
        return temperatures

    def apply(self, density):
        k = self.config.connection_min_conductivity + (1-self.config.connection_min_conductivity)*density.pow(self.config.connection_penalisation)
        a, b = k[self.left], k[self.right]
        conductance = self.geometry * (2*a*(b/(a+b)))
        temperature = _Temperature.apply(conductance, self)
        self.last_temperatures = temperature.detach().numpy().copy()
        # Normalising before the p-norm also keeps large disconnected responses
        # representable. The p-norm bounds every target-cell temperature.
        ratios = torch.stack([torch.linalg.vector_norm(t[source]/reference, ord=8)
            for t, source, reference in zip(temperature, self.targets, self.reference)])
        return ratios/self.config.connection_max_temperature - 1

    def diagnostics(self, density):
        return connected_targets(density, self.config.nx, self.config.ny, self.anchors, self.targets)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for name in ("solver", "rhs", "solution", "matrix"):
            obj = getattr(self, name)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)


class _Temperature(torch.autograd.Function):
    @staticmethod
    def forward(ctx, conductance, solver):
        temperature = solver.solve(conductance.detach().numpy(), solver.loads)
        ctx.solver = solver
        ctx.save_for_backward(conductance, torch.from_numpy(temperature))
        return torch.from_numpy(temperature.copy())

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        conductance, temperature = ctx.saved_tensors
        solver = ctx.solver
        adjoint = solver.solve(conductance.detach().numpy(), gradient.detach().numpy())
        jump_t = temperature.numpy()[:, solver.left] - temperature.numpy()[:, solver.right]
        jump_a = adjoint[:, solver.left] - adjoint[:, solver.right]
        return torch.from_numpy(-np.sum(jump_t*jump_a, axis=0)), None
