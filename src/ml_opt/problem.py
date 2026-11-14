"""Element-density compliance objective with a physical-volume inequality."""

from dataclasses import dataclass, replace

import numpy as np
import torch

from .config import CantileverConfig
from .fem import StructuredMesh
from .fem.state_solver import create_state_solver
from .fem.elasticity import compliance
from .fem.filter import HelmholtzFilter
from .fem.fields import FieldSolves, field_compliance
from .projection import heaviside_projection
from .material import elastic_modulus


@dataclass
class Evaluation:
    compliance: float
    volume_fraction: float
    constraint: float
    density: np.ndarray
    displacement: np.ndarray
    residual: float
    linear_iterations: int
    objective_gradient: np.ndarray | None = None
    constraint_gradient: np.ndarray | None = None
    field_densities: dict[str, np.ndarray] | None = None
    field_compliances: dict[str, float] | None = None
    field_volumes: dict[str, float] | None = None
    field_displacements: dict[str, np.ndarray] | None = None
    worst_field: str | None = None
    field_gradients: dict[str, np.ndarray] | None = None
    epigraph_compliance: float | None = None
    epigraph_constraint: float | None = None
    projection_beta: float = 0.0


class CantileverProblem:
    def __init__(self, config: CantileverConfig, *, state_solver=None):
        if state_solver is not None and config.robust:
            raise ValueError("A supplied state solver is supported only for single-field problems")
        self.config = config
        self.projection_beta = config.projection_beta
        self.mesh = StructuredMesh(config.nx, config.ny, config.length, config.height)
        self.solver = state_solver if state_solver is not None else create_state_solver(config)
        self.rom_info = getattr(self.solver, "rom_info", None)
        if self.rom_info is not None:
            config = self.config = replace(config, rom_basis_sha256=self.rom_info["basis_sha256"])
        self.fields = None
        self.density_filter = None
        try:
            self.density_filter = HelmholtzFilter(self.mesh, config.filter_radius)
            if config.robust:
                self.fields = FieldSolves(config, self.solver)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            if self.fields is not None:
                self.fields.close()
        finally:
            try:
                if self.density_filter is not None:
                    self.density_filter.close()
            finally:
                self.solver.close()

    @property
    def solve_count(self) -> int:
        return self.fields.solve_count if self.fields is not None else self.solver.solve_count

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def initial_design(self) -> np.ndarray:
        return np.full(self.mesh.element_count, self.config.volume_fraction)

    def evaluate(self, design: np.ndarray, sensitivities: bool = True) -> Evaluation:
        array = np.asarray(design, dtype=np.float64)
        if array.shape != (self.mesh.element_count,) or not np.all(np.isfinite(array)):
            raise ValueError("Expected one finite design value per element")
        if np.any(array < 0) or np.any(array > 1):
            raise ValueError("Design densities must lie in [0, 1]")
        x = torch.tensor(array, dtype=torch.float64, requires_grad=sensitivities)
        names = ("eroded", "nominal", "dilated")
        field_densities = field_compliances = field_volumes = field_displacements = None
        worst_field = None
        field_gradients = None
        with torch.set_grad_enabled(sensitivities):
            filtered = self.density_filter.apply(x)
            if self.config.robust:
                eta, offset = self.config.projection_eta, self.config.robust_eta_offset
                densities = torch.stack([heaviside_projection(filtered, self.projection_beta, threshold)
                                         for threshold in (eta + offset, eta, eta - offset)])
                density = densities[1]
            else:
                densities = density = heaviside_projection(filtered, self.projection_beta,
                                                           self.config.projection_eta)
            moduli = elastic_modulus(densities, self.config)
            if self.config.robust:
                compliances = field_compliance(moduli, self.fields)
                # The maximum is a reporting metric. MMA uses all three smooth
                # compliance constraints and its native epigraph variable z.
                objective, active = torch.max(compliances, dim=0)
                worst_field = names[int(active)]
                volumes = densities.mean(dim=1)
                volume = volumes[2]
                states = self.fields.last_states
                state = states[int(active)]
                field_densities = dict(zip(names, densities.detach().numpy().copy()))
                field_compliances = dict(zip(names, compliances.detach().tolist()))
                field_volumes = dict(zip(names, volumes.detach().tolist()))
                field_displacements = {name: s.displacement.copy() for name, s in zip(names, states)}
                residual = max(s.relative_residual for s in states)
                iterations = sum(s.iterations for s in states)
            else:
                objective = compliance(moduli, self.solver)
                volume = density.mean()
                state = self.solver.last_state
                residual, iterations = state.relative_residual, state.iterations
            constraint = volume / self.config.volume_fraction - 1.0
            objective_gradient = constraint_gradient = None
            if sensitivities:
                if self.config.robust:
                    field_gradients = {name: torch.autograd.grad(compliances[index], x, retain_graph=True)[0].numpy().copy()
                                       for index, name in enumerate(names)}
                    objective_gradient = field_gradients[worst_field].copy()
                else:
                    objective_gradient = torch.autograd.grad(objective, x, retain_graph=True)[0].numpy().copy()
                constraint_gradient = torch.autograd.grad(constraint, x)[0].numpy().copy()
        return Evaluation(float(objective.detach()), float(volume.detach()), float(constraint.detach()),
                          density.detach().numpy().copy(), state.displacement.copy(),
                          residual, iterations, objective_gradient, constraint_gradient,
                          field_densities, field_compliances, field_volumes, field_displacements, worst_field,
                          field_gradients, projection_beta=self.projection_beta)
