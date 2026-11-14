"""Element-density SIMP compliance objective with a volume inequality."""

from dataclasses import dataclass

import numpy as np
import torch

from .config import CantileverConfig
from .fem import ElasticitySolver, StructuredMesh
from .fem.elasticity import compliance
from .fem.filter import HelmholtzFilter
from .projection import heaviside_projection


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


class CantileverProblem:
    def __init__(self, config: CantileverConfig, *, state_solver=None):
        self.config = config
        self.mesh = StructuredMesh(config.nx, config.ny, config.length, config.height)
        # Validation can supply a frozen Galerkin solver, retaining the identical
        # material/filter/projection/autograd path. Normal optimisation uses PETSc.
        self.solver = state_solver if state_solver is not None else ElasticitySolver(self.mesh, config.poisson_ratio, config.thickness,
                                       config.load, config.linear_tolerance,
                                       load_angle=config.load_angle,
                                       load_node=None if config.load_node_x is None else
                                       (config.load_node_x, config.load_node_y))
        try:
            self.density_filter = HelmholtzFilter(self.mesh, config.filter_radius)
        except BaseException:
            self.solver.close()
            raise

    def close(self) -> None:
        try:
            self.density_filter.close()
        finally:
            self.solver.close()

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
        with torch.set_grad_enabled(sensitivities):
            filtered = self.density_filter.apply(x)
            density = heaviside_projection(filtered, self.config.projection_beta,
                                           self.config.projection_eta)
            moduli = self.config.minimum_modulus + (self.config.young_modulus -
                      self.config.minimum_modulus) * density.pow(self.config.penalisation)
            objective = compliance(moduli, self.solver)
            volume = density.mean()
            constraint = volume / self.config.volume_fraction - 1.0
            objective_gradient = constraint_gradient = None
            if sensitivities:
                objective_gradient = torch.autograd.grad(objective, x, retain_graph=True)[0].numpy().copy()
                constraint_gradient = torch.autograd.grad(constraint, x)[0].numpy().copy()
        state = self.solver.last_state
        assert state is not None
        return Evaluation(float(objective.detach()), float(volume.detach()), float(constraint.detach()),
                          density.detach().numpy().copy(), state.displacement.copy(),
                          state.relative_residual, state.iterations,
                          objective_gradient, constraint_gradient)
