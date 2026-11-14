"""Sparse PETSc state solves and a self-adjoint compliance autograd operator."""

from dataclasses import dataclass

import numpy as np
from petsc4py import PETSc
from scipy.sparse import coo_matrix
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
                 load_node: tuple[int, int] | None = None):
        if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
            raise RuntimeError("This example requires a real double-precision PETSc build")
        self.mesh = mesh
        self.tolerance = tolerance
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

    def solve(self, moduli: np.ndarray) -> ElasticityState:
        moduli = np.asarray(moduli, dtype=np.float64)
        if moduli.shape != (self.mesh.element_count,) or not np.all(np.isfinite(moduli)) or np.any(moduli <= 0):
            raise ValueError("Expected one finite positive modulus per element")
        count = len(self.mesh.free_dofs)
        entries = (moduli[:, None, None] * self.element_stiffness).ravel()[self.mask]
        stiffness = coo_matrix((entries, (self.rows, self.columns)), shape=(count, count)).tocsr()
        # PERF B3: recreate matrix/KSP and redo symbolic + numeric LU per evaluation.
        # The sparsity graph is fixed; numeric factors change with the design.
        matrix = PETSc.Mat().createAIJ(
            size=(count, count), comm=PETSc.COMM_SELF,
            csr=(stiffness.indptr.astype(PETSc.IntType),
                 stiffness.indices.astype(PETSc.IntType), stiffness.data))
        matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        solution, rhs = matrix.createVecs()
        solver = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        try:
            rhs.array[:] = self.force[self.mesh.free_dofs]
            solver.setOperators(matrix)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setTolerances(rtol=self.tolerance, atol=1e-14, max_it=10000)
            solver.setErrorIfNotConverged(True)
            solver.solve(rhs, solution)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError(f"PETSc failed to converge: {solver.getConvergedReason()}")
            free_u = solution.array.copy()
            residual = float(np.linalg.norm(stiffness @ free_u - rhs.array) /
                             max(np.linalg.norm(rhs.array), np.finfo(float).tiny))
            if not np.isfinite(residual) or residual > max(1e-7, 50 * self.tolerance):
                raise RuntimeError(f"PETSc equilibrium residual is too large: {residual:.3e}")
            displacement = np.zeros(self.mesh.dof_count)
            displacement[self.mesh.free_dofs] = free_u
            element_u = displacement[self.mesh.element_dofs]
            energy = np.einsum("ei,ij,ej->e", element_u, self.element_stiffness, element_u)
            state = ElasticityState(displacement, float(self.force @ displacement), energy,
                                    residual, solver.getIterationNumber())
            self.solve_count += 1
            self.last_state = state
            return state
        finally:
            solver.destroy()
            solution.destroy()
            rhs.destroy()
            matrix.destroy()


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
