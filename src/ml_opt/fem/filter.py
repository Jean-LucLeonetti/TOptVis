"""Cell-centred Helmholtz density filter with homogeneous Neumann boundaries.

Solve (I - radius**2 * Laplacian_h) rho = design on the rectangular mesh.
The finite-volume operator is symmetric, preserves the mean, and has a
nonnegative inverse. Missing boundary faces impose zero normal flux.
"""

import math

import numpy as np
from petsc4py import PETSc
import torch

from .mesh import StructuredMesh


class HelmholtzFilter:
    def __init__(self, mesh: StructuredMesh, radius: float):
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("filter_radius must be finite and nonnegative")
        if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
            raise RuntimeError("The Helmholtz filter requires real double-precision PETSc")
        self.mesh, self.radius = mesh, radius
        self._closed = False
        self._matrix = self._solver = self._rhs = self._solution = None
        self.solve_count = 0
        if radius == 0:
            return
        try:
            cells = np.arange(mesh.element_count, dtype=PETSc.IntType).reshape(mesh.ny, mesh.nx)
            rows, columns, values = [cells.ravel()], [cells.ravel()], [np.ones(mesh.element_count)]
            for left, right, spacing in (
                (cells[:, :-1].ravel(), cells[:, 1:].ravel(), mesh.length / mesh.nx),
                (cells[:-1, :].ravel(), cells[1:, :].ravel(), mesh.height / mesh.ny),
            ):
                ratio = radius / spacing
                weight = ratio * ratio
                if not math.isfinite(weight):
                    raise ValueError("Filter size is too large relative to the mesh spacing")
                rows.extend((left, right, left, right))
                columns.extend((left, right, right, left))
                values.extend((np.full(left.size, weight), np.full(left.size, weight),
                               np.full(left.size, -weight), np.full(left.size, -weight)))
            self._matrix = PETSc.Mat().create(comm=PETSc.COMM_SELF)
            self._matrix.setSizes((mesh.element_count, mesh.element_count))
            self._matrix.setType("aij")
            self._matrix.setPreallocationCOO(np.concatenate(rows), np.concatenate(columns))
            self._matrix.setValuesCOO(np.concatenate(values), PETSc.InsertMode.INSERT_VALUES)
            self._matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            self._matrix.setOption(PETSc.Mat.Option.SPD, True)
            self._solution, self._rhs = self._matrix.createVecs()
            self._solver = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            self._solver.setOperators(self._matrix)
            self._solver.setType("preonly")
            self._solver.getPC().setType("lu")
            self._solver.setErrorIfNotConverged(True)
            # Matrix values never change within a run: retain numeric factors too.
        except BaseException:
            self.close()
            raise

    def solve(self, values: np.ndarray) -> np.ndarray:
        """Apply the filter (or its identical transpose) to any finite RHS."""
        if self._closed:
            raise RuntimeError("The Helmholtz filter is closed")
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.mesh.element_count,) or not np.all(np.isfinite(array)):
            raise ValueError("Expected one finite filter value per element")
        if self.radius == 0:
            return array.copy()
        self._rhs.array[:] = array
        self._solver.solve(self._rhs, self._solution)
        result = self._solution.array.copy()
        if self._solver.getConvergedReason() <= 0 or not np.all(np.isfinite(result)):
            raise RuntimeError("PETSc failed to solve the Helmholtz filter")
        self.solve_count += 1
        return result

    def apply(self, design: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("The Helmholtz filter is closed")
        if design.device.type != "cpu" or design.dtype != torch.float64:
            raise ValueError("The Helmholtz bridge expects CPU float64 tensors")
        if self.radius == 0:
            return design
        return _FilteredDensity.apply(design, self)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name in ("_solver", "_rhs", "_solution", "_matrix"):
            obj = getattr(self, name)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_closed"):
            try:
                self.close()
            except Exception:
                pass


class _FilteredDensity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, design, density_filter):
        ctx.density_filter = density_filter
        return torch.from_numpy(density_filter.solve(design.detach().numpy()))

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        # A = A^T, so dJ/dx = A^{-T} dJ/drho uses the same retained factors.
        return torch.from_numpy(ctx.density_filter.solve(gradient.detach().numpy())), None
