"""Configuration for the nondimensional, plane-stress cantilever benchmark."""

from dataclasses import dataclass
import math
import re

from .boundaries import (BoundaryCondition, LoadCase, normalise_conditions,
                         normalise_load_cases, resolve_boundary_data)
from .regions import NonDesignRegion, normalise_regions, region_densities


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
    volume_constraint: str = "lt"
    max_solid_size: float = 0.0
    max_size_beta: float = 16.0
    max_size_tolerance: float = 1e-3
    max_void_size: float = 0.0
    max_void_beta: float = 16.0
    max_void_tolerance: float = 1e-3
    material_model: str = "simp"
    penalisation: float = 3.0
    filter_radius: float = 0.0
    projection_beta: float = 0.0
    projection_eta: float = 0.5
    projection_schedule: str = "fixed"
    projection_beta_interval: int = 25
    projection_beta_multiplier: float = 2.0
    projection_beta_max: float = 16.0
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
    linear_solver: str = "lu"
    state_solver: str = "fom"
    rom_basis_path: str | None = None
    rom_basis_sha256: str | None = None
    boundary_conditions: tuple[BoundaryCondition, ...] | None = None
    load_cases: tuple[LoadCase, ...] | None = None
    load_aggregation: str = "weighted"
    non_design_regions: tuple[NonDesignRegion, ...] = ()

    @property
    def extended_boundaries(self):
        return len(self.load_cases) > 1 or any(condition.kind not in ("fixed", "force")
                                            for condition in (self.boundary_conditions or ()))

    @property
    def imposed_motion(self):
        return any(condition.kind == "displacement" and
                   (("x" in condition.components and condition.ux != 0) or
                    ("y" in condition.components and condition.uy != 0))
                   for condition in (self.boundary_conditions or ()))

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_conditions", normalise_conditions(self.boundary_conditions))
        object.__setattr__(self, "load_cases", normalise_load_cases(self.load_cases))
        object.__setattr__(self, "non_design_regions", normalise_regions(self.non_design_regions))
        if self.load_aggregation not in ("weighted", "worst"):
            raise ValueError("Load aggregation must be weighted or worst")
        if self.state_solver == "rom" and self.extended_boundaries:
            raise ValueError("Directional supports, prescribed motion, and independent load cases require the full model")
        if len(self.load_cases) > 1 and self.boundary_conditions is None:
            raise ValueError("Assign forces to each independent load case")
        if self.linear_solver not in ("lu", "cholesky", "cg-icc", "cg-gamg"):
            raise ValueError("linear_solver must be lu, cholesky, cg-icc, or cg-gamg")
        if self.state_solver not in ("fom", "rom"):
            raise ValueError("state_solver must be fom or rom")
        if self.state_solver == "rom" and (not isinstance(self.rom_basis_path, str) or not self.rom_basis_path):
            raise ValueError("ROM optimisation requires a rom_basis_path")
        if self.state_solver == "fom" and (self.rom_basis_path is not None or self.rom_basis_sha256 is not None):
            raise ValueError("ROM basis options require state_solver='rom'")
        if self.rom_basis_sha256 is not None and (not isinstance(self.rom_basis_sha256, str) or
                                                  not re.fullmatch(r"[0-9a-f]{64}", self.rom_basis_sha256)):
            raise ValueError("rom_basis_sha256 must be a lowercase SHA-256 digest")
        for name in ("nx", "ny", "max_iterations", "min_iterations",
                     "convergence_patience", "max_inner_iterations", "projection_beta_interval"):
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
        if self.volume_constraint not in ("lt", "gt", "eq"):
            raise ValueError("volume_constraint must be lt, gt, or eq")
        if self.non_design_regions:
            fixed = region_densities(self.non_design_regions, self.nx, self.ny)
            minimum = (fixed == 1).sum() / fixed.size
            maximum = 1 - (fixed == 0).sum() / fixed.size
            if self.volume_constraint in ("lt", "eq") and minimum > self.volume_fraction * (1 + self.feasibility_tolerance):
                raise ValueError("Fixed solid regions exceed the permitted material volume")
            if self.volume_constraint in ("gt", "eq") and maximum < self.volume_fraction * (1 - self.feasibility_tolerance):
                raise ValueError("Fixed void regions prevent the required material volume")
        for size_name, beta_name, tolerance_name in (
                ("max_solid_size", "max_size_beta", "max_size_tolerance"),
                ("max_void_size", "max_void_beta", "max_void_tolerance")):
            size = getattr(self, size_name)
            if size < 0:
                raise ValueError(f"{size_name} must be nonnegative; 0 disables the constraint")
            if not 0 < getattr(self, beta_name) <= 64:
                raise ValueError(f"{beta_name} must lie in (0, 64]")
            if not 0 < getattr(self, tolerance_name) < 1:
                raise ValueError(f"{tolerance_name} must lie in (0, 1)")
            if size > 0:
                dx, dy = self.length / self.nx, self.height / self.ny
                if size <= 2 * max(dx, dy):
                    raise ValueError(f"{size_name} must exceed two element spacings in both directions; refine the mesh")
                sx, sy = math.ceil(size / (2*dx)), math.ceil(size / (2*dy))
                if (2*sx+1) * (2*sy+1) > 1_000_000:
                    raise ValueError(f"Maximum-size neighbourhood exceeds one million cells; reduce {size_name}")
        if self.material_model not in ("simp", "ramp"):
            raise ValueError("material_model must be simp or ramp")
        minimum_penalty = 1 if self.material_model == "simp" else 0
        if self.penalisation < minimum_penalty:
            raise ValueError(f"{self.material_model.upper()} penalisation must be at least {minimum_penalty}")
        if not math.isfinite(self.filter_radius) or self.filter_radius < 0:
            raise ValueError("filter_radius must be finite and nonnegative")
        if not math.isfinite(self.projection_beta) or self.projection_beta < 0:
            raise ValueError("projection_beta must be finite and nonnegative")
        if not math.isfinite(self.projection_eta) or not 0 < self.projection_eta < 1:
            raise ValueError("projection_eta must be finite and lie between 0 and 1")
        if self.projection_schedule not in ("fixed", "geometric"):
            raise ValueError("projection_schedule must be fixed or geometric")
        if self.projection_beta_multiplier <= 1:
            raise ValueError("projection_beta_multiplier must be greater than 1")
        if self.projection_beta_max <= 0:
            raise ValueError("projection_beta_max must be positive")
        if self.projection_schedule == "geometric":
            if self.projection_beta <= 0:
                raise ValueError("Scheduled projection requires projection_beta > 0")
            if self.projection_beta_max < self.projection_beta:
                raise ValueError("projection_beta_max must be at least the starting projection_beta")
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
        if self.boundary_conditions is not None:
            identifiers = {case.id for case in self.load_cases}
            if any(condition.case_id is not None and condition.case_id not in identifiers for condition in self.boundary_conditions):
                raise ValueError("A force refers to an unknown load case")
            for case in self.load_cases:
                try:
                    resolve_boundary_data(self.boundary_conditions, self.nx, self.ny, case_id=case.id,
                                          default_case_id=self.load_cases[0].id, length=self.length, height=self.height)
                except ValueError as error:
                    raise ValueError(f"{case.name}: {error}") from error
        elif self.load_node_x is not None:
            if (type(self.load_node_x) is not int or type(self.load_node_y) is not int or
                    not 1 <= self.load_node_x <= self.nx or not 0 <= self.load_node_y <= self.ny):
                raise ValueError("The load must be on a free mesh node (column 1..nx, row 0..ny)")
