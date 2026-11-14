"""Configuration for the nondimensional, plane-stress cantilever benchmark."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CantileverConfig:
    nx: int = 60
    ny: int = 20
    length: float = 3.0
    height: float = 1.0
    young_modulus: float = 1.0
    minimum_modulus: float = 1e-6
    poisson_ratio: float = 0.3
    thickness: float = 1.0
    load: float = 1.0
    load_angle: float = -90.0
    load_node_x: int | None = None
    load_node_y: int | None = None
    volume_fraction: float = 0.4
    penalisation: float = 3.0
    filter_radius: float = 0.0
    projection_beta: float = 0.0
    projection_eta: float = 0.5
    robust: bool = False
    robust_eta_offset: float = 0.1
    field_execution: str = "sequential"
    optimizer: str = "mma"
    move_limit: float = 0.2
    max_iterations: int = 100
    min_iterations: int = 20
    design_tolerance: float = 5e-3
    objective_tolerance: float = 1e-4
    feasibility_tolerance: float = 1e-4
    convergence_patience: int = 5
    max_inner_iterations: int = 30
    linear_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "max_iterations", "min_iterations",
                     "convergence_patience", "max_inner_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in vars(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("length", "height", "young_modulus", "minimum_modulus",
                     "thickness", "load", "design_tolerance",
                     "objective_tolerance", "feasibility_tolerance", "linear_tolerance"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.minimum_modulus < self.young_modulus:
            raise ValueError("minimum_modulus must be below young_modulus")
        if not -1 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie between -1 and 0.5")
        if not 0 < self.volume_fraction < 1:
            raise ValueError("volume_fraction must lie between 0 and 1")
        if self.penalisation < 1:
            raise ValueError("penalisation must be at least 1")
        if not math.isfinite(self.filter_radius) or self.filter_radius < 0:
            raise ValueError("filter_radius must be finite and nonnegative")
        if not math.isfinite(self.projection_beta) or self.projection_beta < 0:
            raise ValueError("projection_beta must be finite and nonnegative")
        if not math.isfinite(self.projection_eta) or not 0 < self.projection_eta < 1:
            raise ValueError("projection_eta must be finite and lie between 0 and 1")
        if type(self.robust) is not bool:
            raise ValueError("robust must be a boolean")
        if not math.isfinite(self.robust_eta_offset) or not 0 < self.robust_eta_offset < 0.5:
            raise ValueError("robust_eta_offset must be finite and lie between 0 and 0.5")
        if self.field_execution not in ("sequential", "parallel"):
            raise ValueError("field_execution must be sequential or parallel")
        if self.robust:
            if self.projection_beta <= 0:
                raise ValueError("robust formulation requires projection_beta > 0")
            if not self.robust_eta_offset < self.projection_eta < 1 - self.robust_eta_offset:
                raise ValueError("robust projection thresholds eta +/- offset must lie inside (0, 1)")
        if not 0 < self.move_limit <= 1:
            raise ValueError("move_limit must lie in (0, 1]")
        if self.optimizer not in ("mma", "gcmma"):
            raise ValueError("optimizer must be mma or gcmma")
        if not math.isfinite(self.load_angle):
            raise ValueError("load_angle must be finite")
        if (self.load_node_x is None) != (self.load_node_y is None):
            raise ValueError("Specify both load_node_x and load_node_y")
        if self.load_node_x is not None:
            if (type(self.load_node_x) is not int or type(self.load_node_y) is not int or
                    not 1 <= self.load_node_x <= self.nx or not 0 <= self.load_node_y <= self.ny):
                raise ValueError("The load must be on a free mesh node (column 1..nx, row 0..ny)")
