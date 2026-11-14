"""Sparse PETSc state solves and a self-adjoint compliance autograd operator."""

from dataclasses import dataclass

import numpy as np
from petsc4py import PETSc
import torch

from .element import q4_stiffness
from .mesh import StructuredMesh


@dataclass
class ElasticityState:
    displacement: np.ndarray
    compliance: float
    element_energy: np.ndarray
    relative_residual: float
    iterations: int


class ElasticitySolver:
    def __init__(self, mesh: StructuredMesh, poisson: float = 0.3,
                 thickness: float = 1.0, load: float = 1.0,
                 tolerance: float = 1e-10, *, load_angle: float = -90.0,
                 load_node: tuple[int, int] | None = None, linear_solver: str = "lu"):
        if linear_solver not in ("lu", "cholesky", "cg-icc", "cg-gamg"):
            raise ValueError("Unknown elasticity linear solver")
        if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
            raise RuntimeError("This example requires a real double-precision PETSc build")
        self.mesh = mesh
        self.tolerance = tolerance
        self.linear_solver = linear_solver
        self.element_stiffness = q4_stiffness(mesh.length / mesh.nx, mesh.height / mesh.ny,
                                              poisson, thickness)
        self.force = mesh.cantilever_force(load, load_angle, load_node)
        mapping = np.full(mesh.dof_count, -1, dtype=np.int64)
        mapping[mesh.free_dofs] = np.arange(len(mesh.free_dofs))
        reduced = mapping[mesh.element_dofs]
        rows = np.repeat(reduced, 8, axis=1).ravel()
        columns = np.tile(reduced, (1, 8)).ravel()
        self.mask = (rows >= 0) & (columns >= 0)
        self.rows, self.columns = rows[self.mask], columns[self.mask]
        self.solve_count = 0
        self.last_state: ElasticityState | None = None
        self._closed = False
        self._matrix = self._solution = self._rhs = self._residual = self._solver = None
        self._near_nullspace = None
        try:
            count = len(mesh.free_dofs)
            self._matrix = PETSc.Mat().create(comm=PETSc.COMM_SELF)
            self._matrix.setSizes((count, count))
            self._matrix.setType("aij")
            self._matrix.setBlockSize(2)
            # PETSc remembers the element-entry reduction, including shared-node sums.
            self._matrix.setPreallocationCOO(self.rows.astype(PETSc.IntType),
                                              self.columns.astype(PETSc.IntType))
            self._matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            self._matrix.setOption(PETSc.Mat.Option.SPD, True)
            self._solution, self._rhs = self._matrix.createVecs()
            self._residual = self._rhs.duplicate()
            self._rhs.array[:] = self.force[mesh.free_dofs]
            self._rhs_norm = max(self._rhs.norm(), np.finfo(float).tiny)
            self._solver = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            self._solver.setOperators(self._matrix)
            if linear_solver in ("lu", "cholesky"):
                self._solver.setType("preonly")
                self._solver.getPC().setType(linear_solver)
            else:
                self._solver.setType("cg")
                self._solver.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
                self._solver.getPC().setType("icc" if linear_solver == "cg-icc" else "gamg")
                if linear_solver == "cg-gamg":
                    # Vertex-major x/y DOFs and all three rigid-body modes for
                    # elasticity smoothed aggregation (PETSc GAMG guidance).
                    nodes = mesh.free_dofs[::2] // 2
                    coordinates = PETSc.Vec().createWithArray(mesh.coordinates[nodes].ravel().copy(),
                                                              bsize=2, comm=PETSc.COMM_SELF)
                    try:
                        self._near_nullspace = PETSc.NullSpace().createRigidBody(coordinates)
                        self._matrix.setNearNullSpace(self._near_nullspace)
                    finally:
                        coordinates.destroy()
            self._solver.setTolerances(rtol=tolerance, atol=1e-14, max_it=10000)
            self._solver.setErrorIfNotConverged(True)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release the per-problem native objects; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        for name in ("_solver", "_solution", "_rhs", "_residual", "_matrix", "_near_nullspace"):
            obj = getattr(self, name)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)

    def __del__(self):
        # Explicit close is preferred; guard partially constructed objects/interpreter exit.
        if hasattr(self, "_closed"):
            try:
                self.close()
            except Exception:
                pass

    def solve(self, moduli: np.ndarray) -> ElasticityState:
        if self._closed:
            raise RuntimeError("The elasticity solver is closed")
        moduli = np.asarray(moduli, dtype=np.float64)
        if moduli.shape != (self.mesh.element_count,) or not np.all(np.isfinite(moduli)) or np.any(moduli <= 0):
            raise ValueError("Expected one finite positive modulus per element")
        entries = (moduli[:, None, None] * self.element_stiffness).ravel()[self.mask]
        # PERF B3: retain the graph/KSP/LU symbolic factorisation. Updating values
        # invalidates numeric factors, so every solve still uses the current design.
        # Do not enable KSP/PC reuse-preconditioner: that would retain stale factors.
        self._matrix.setValuesCOO(entries, PETSc.InsertMode.INSERT_VALUES)
        self._solver.solve(self._rhs, self._solution)
        if self._solver.getConvergedReason() <= 0:
            raise RuntimeError(f"PETSc failed to converge: {self._solver.getConvergedReason()}")
        self._matrix.mult(self._solution, self._residual)
        self._residual.axpy(-1.0, self._rhs)
        residual = self._residual.norm() / self._rhs_norm
        if not np.isfinite(residual) or residual > max(1e-7, 50 * self.tolerance):
            raise RuntimeError(f"PETSc equilibrium residual is too large: {residual:.3e}")
        displacement = np.zeros(self.mesh.dof_count)
        displacement[self.mesh.free_dofs] = self._solution.array
        element_u = displacement[self.mesh.element_dofs]
        energy = np.einsum("ei,ij,ej->e", element_u, self.element_stiffness, element_u)
        state = ElasticityState(displacement, float(self.force @ displacement), energy,
                                residual, self._solver.getIterationNumber())
        self.solve_count += 1
        self.last_state = state
        return state


class _Compliance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, moduli: torch.Tensor, solver: ElasticitySolver) -> torch.Tensor:
        if moduli.device.type != "cpu" or moduli.dtype != torch.float64:
            raise ValueError("The PETSc bridge expects CPU float64 tensors")
        state = solver.solve(moduli.detach().numpy())
        ctx.save_for_backward(torch.from_numpy(state.element_energy.copy()))
        return moduli.new_tensor(state.compliance)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, output_gradient):
        # K u = f, C = f^T u: dC/dE_e = -u_e^T K_e(E=1) u_e.
        (energy,) = ctx.saved_tensors
        return -output_gradient * energy, None


def compliance(moduli: torch.Tensor, solver: ElasticitySolver) -> torch.Tensor:
    """First-order implicit differentiation; the backward pass needs no extra solve."""
    return _Compliance.apply(moduli, solver)
