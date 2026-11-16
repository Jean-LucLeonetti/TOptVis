"""PETSc equilibrium/adjoint solves and implicit PyTorch sensitivities."""

from dataclasses import dataclass

import numpy as np
from petsc4py import PETSc
import torch

from .element import q4_stiffness
from .mesh import StructuredMesh
from ..boundaries import normalise_load_cases, resolve_boundary_data


@dataclass
class ElasticityState:
    displacement: np.ndarray
    compliance: float
    element_energy: np.ndarray  # Negative objective derivative w.r.t. element modulus.
    relative_residual: float
    iterations: int
    case_compliances: dict[str, float] | None = None
    active_case: str | None = None
    state_solves: int = 1


class ElasticitySolver:
    def __init__(self, mesh: StructuredMesh, poisson: float = 0.3,
                 thickness: float = 1.0, load: float = 1.0,
                 tolerance: float = 1e-10, *, load_angle: float = -90.0,
                 load_node: tuple[int, int] | None = None, linear_solver: str = "lu",
                 load_cases=None, load_aggregation="weighted"):
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
        self.load_cases = normalise_load_cases(load_cases)
        self.load_aggregation = load_aggregation
        if load_aggregation not in ("weighted", "worst"):
            raise ValueError("Unknown load aggregation")
        self.forces = [self.force] if mesh.boundary_conditions is None else [
            resolve_boundary_data(mesh.boundary_conditions, mesh.nx, mesh.ny, case_id=case.id,
                default_case_id=self.load_cases[0].id, length=mesh.length, height=mesh.height).force
            for case in self.load_cases]
        self.prescribed = mesh.prescribed_values
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
            paired = (count % 2 == 0 and np.all(mesh.free_dofs[::2] % 2 == 0) and
                      np.array_equal(mesh.free_dofs[1::2], mesh.free_dofs[::2] + 1))
            self._matrix.setBlockSize(2 if paired else 1)
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
                    if paired:
                        nodes = mesh.free_dofs[::2] // 2
                        coordinates = PETSc.Vec().createWithArray(mesh.coordinates[nodes].ravel().copy(),
                                                                  bsize=2, comm=PETSc.COMM_SELF)
                        try:
                            self._near_nullspace = PETSc.NullSpace().createRigidBody(coordinates)
                        finally:
                            coordinates.destroy()
                    else:
                        modes = np.zeros((mesh.dof_count, 3))
                        modes[0::2, 0], modes[1::2, 1] = 1, 1
                        modes[0::2, 2], modes[1::2, 2] = -mesh.coordinates[:, 1], mesh.coordinates[:, 0]
                        basis, _ = np.linalg.qr(modes[mesh.free_dofs])
                        vectors = [PETSc.Vec().createWithArray(column.copy(), comm=PETSc.COMM_SELF) for column in basis.T]
                        try:
                            self._near_nullspace = PETSc.NullSpace().create(vectors=vectors, comm=PETSc.COMM_SELF)
                        finally:
                            for vector in vectors:
                                vector.destroy()
                    self._matrix.setNearNullSpace(self._near_nullspace)
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
        lift = np.zeros(self.mesh.dof_count)
        imposed = np.any(self.prescribed)
        if imposed:
            lift = np.bincount(self.mesh.element_dofs.ravel(), minlength=self.mesh.dof_count,
                weights=(moduli[:, None] * (self.prescribed[self.mesh.element_dofs] @ self.element_stiffness)).ravel())
        states = []
        for force in self.forces:
            free_u, residual, iterations = self._solve_rhs(force[self.mesh.free_dofs] - lift[self.mesh.free_dofs])
            displacement = self.prescribed.copy()
            displacement[self.mesh.free_dofs] = free_u
            element_u = displacement[self.mesh.element_dofs]
            unit_energy = np.einsum("ei,ij,ej->e", element_u, self.element_stiffness, element_u)
            energy = unit_energy
            count = 1
            if imposed:
                adjoint = np.zeros(self.mesh.dof_count)
                if np.any(force[self.mesh.free_dofs]):
                    adjoint[self.mesh.free_dofs], adj_residual, adj_iterations = self._solve_rhs(force[self.mesh.free_dofs])
                    residual = max(residual, adj_residual); iterations += adj_iterations; count += 1
                # For elastic work u.T K u with fixed prescribed motion:
                # dC/dE_e = u_e.T K_e u_e - 2 lambda_e.T K_e u_e.
                energy = 2 * np.einsum("ei,ij,ej->e", adjoint[self.mesh.element_dofs], self.element_stiffness, element_u) - unit_energy
            value = float(moduli @ unit_energy) if imposed else float(force @ displacement)
            states.append(ElasticityState(displacement, value, energy, residual, iterations, state_solves=count))
        values = np.array([state.compliance for state in states])
        active = int(np.argmax(values))
        weights = np.array([case.weight for case in self.load_cases], dtype=float)
        weights /= weights.sum()
        if self.load_aggregation == "worst":
            weights[:] = 0; weights[active] = 1
        state = ElasticityState(states[active].displacement.copy(), float(weights @ values),
            sum(weight * item.element_energy for weight, item in zip(weights, states)),
            max(item.relative_residual for item in states), sum(item.iterations for item in states),
            {case.id: item.compliance for case, item in zip(self.load_cases, states)}, self.load_cases[active].id,
            sum(item.state_solves for item in states))
        self.last_state = state
        return state

    def _solve_rhs(self, rhs):
        self._rhs.array[:] = rhs
        self._rhs_norm = max(self._rhs.norm(), np.finfo(float).tiny)
        self._solver.solve(self._rhs, self._solution)
        if self._solver.getConvergedReason() <= 0:
            raise RuntimeError(f"PETSc failed to converge: {self._solver.getConvergedReason()}")
        self._matrix.mult(self._solution, self._residual)
        self._residual.axpy(-1.0, self._rhs)
        residual = self._residual.norm() / self._rhs_norm
        if not np.isfinite(residual) or residual > max(1e-7, 50 * self.tolerance):
            raise RuntimeError(f"PETSc equilibrium residual is too large: {residual:.3e}")
        self.solve_count += 1
        return self._solution.array.copy(), residual, self._solver.getIterationNumber()


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
        # The solver supplies -dC/dE, including prescribed-motion adjoints.
        (energy,) = ctx.saved_tensors
        return -output_gradient * energy, None


def compliance(moduli: torch.Tensor, solver: ElasticitySolver) -> torch.Tensor:
    """First-order implicit differentiation; the backward pass needs no extra solve."""
    return _Compliance.apply(moduli, solver)
