"""Mesh selections and boundary data, independent of the state-solver backend."""

from dataclasses import dataclass
import math
import re

import numpy as np


@dataclass(frozen=True)
class MeshSelection:
    kind: str
    start: tuple[float, float]
    end: tuple[float, float] | None = None

    def __post_init__(self):
        if self.kind not in ("point", "line", "surface"):
            raise ValueError("Selection must be point, line, or surface")
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is None and name == "end":
                continue
            if (not isinstance(value, (tuple, list)) or len(value) != 2 or
                    any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                        not math.isfinite(x) or not 0 <= x <= 1 for x in value)):
                raise ValueError("Selection coordinates must be pairs of fractions between 0 and 1")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        if self.kind == "point":
            if self.end is not None and self.end != self.start:
                raise ValueError("A point selection has only one location")
            object.__setattr__(self, "end", None)
        elif self.end is None:
            raise ValueError("Line and surface selections need two endpoints")
        elif self.kind == "line" and (self.start[0] != self.end[0] and self.start[1] != self.end[1]):
            raise ValueError("Select a horizontal or vertical mesh line")
        if self.end is not None:
            start, end = self.start, self.end
            object.__setattr__(self, "start", tuple(min(a, b) for a, b in zip(start, end)))
            object.__setattr__(self, "end", tuple(max(a, b) for a, b in zip(start, end)))

    def nodes_and_weights(self, nx, ny):
        """Snap to nodes; consistent Q1 weights preserve each load's resultant."""
        x0, y0 = (int(math.floor(a * n + .5)) for a, n in zip(self.start, (nx, ny)))
        x1, y1 = (int(math.floor(a * n + .5)) for a, n in zip(self.end or self.start, (nx, ny)))
        if self.kind == "line" and x0 == x1 and y0 == y1:
            raise ValueError("The selected line is smaller than one mesh edge; refine the mesh")
        if self.kind == "surface" and (x0 == x1 or y0 == y1):
            raise ValueError("The selected surface is smaller than one element; refine the mesh")
        xs, ys = np.arange(x0, x1 + 1), np.arange(y0, y1 + 1)
        wx, wy = np.ones(len(xs)), np.ones(len(ys))
        if len(xs) > 1:
            wx[[0, -1]] = .5
        if len(ys) > 1:
            wy[[0, -1]] = .5
        nodes = (ys[:, None] * (nx + 1) + xs[None, :]).ravel()
        weights = (wy[:, None] * wx[None, :]).ravel()
        return nodes, weights / weights.sum()


@dataclass(frozen=True)
class BoundaryCondition:
    kind: str
    selection: MeshSelection
    magnitude: float = 1.0
    angle: float = -90.0
    components: str = "xy"
    ux: float = 0.0
    uy: float = 0.0
    case_id: str | None = None

    def __post_init__(self):
        if self.kind not in ("fixed", "restrain_x", "restrain_y", "displacement", "force"):
            raise ValueError("Unknown boundary condition type")
        if self.components not in ("x", "y", "xy"):
            raise ValueError("Displacement components must be x, y, or xy")
        for value in (self.ux, self.uy):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Prescribed displacements must be finite")
        if self.case_id is not None and (self.kind != "force" or not isinstance(self.case_id, str) or
                                        not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.case_id)):
            raise ValueError("Only forces can belong to a named load case")
        if isinstance(self.selection, dict):
            try:
                object.__setattr__(self, "selection", MeshSelection(**self.selection))
            except TypeError as error:
                raise ValueError("Invalid mesh selection") from error
        if not isinstance(self.selection, MeshSelection):
            raise ValueError("A boundary condition needs a mesh selection")
        if (isinstance(self.magnitude, bool) or not isinstance(self.magnitude, (int, float)) or
                not math.isfinite(self.magnitude) or not 0 < self.magnitude <= 1e6):
            raise ValueError("Force magnitude must lie in (0, 1000000]")
        if (isinstance(self.angle, bool) or not isinstance(self.angle, (int, float)) or
                not math.isfinite(self.angle) or not -180 <= self.angle <= 180):
            raise ValueError("Force direction must lie between -180 and 180 degrees")


def normalise_conditions(values):
    if values is None:
        return None
    if not isinstance(values, (tuple, list)) or len(values) > 128:
        raise ValueError("Provide at most 128 boundary conditions")
    try:
        return tuple(value if isinstance(value, BoundaryCondition) else BoundaryCondition(**value)
                     for value in values)
    except TypeError as error:
        raise ValueError("Invalid boundary condition") from error


@dataclass(frozen=True)
class LoadCase:
    id: str = "case-1"
    name: str = "Load case 1"
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.id):
            raise ValueError("Invalid load case identifier")
        if not isinstance(self.name, str) or not 1 <= len(self.name.strip()) <= 80:
            raise ValueError("Load case names need between 1 and 80 characters")
        if (isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)) or
                not math.isfinite(self.weight) or not 0 < self.weight <= 1e6):
            raise ValueError("Load case weights must be positive and finite")


def normalise_load_cases(values):
    if values is None:
        return (LoadCase(),)
    if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= 16:
        raise ValueError("Provide between 1 and 16 load cases")
    try:
        cases = tuple(value if isinstance(value, LoadCase) else LoadCase(**value) for value in values)
    except TypeError as error:
        raise ValueError("Invalid load case") from error
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Load case identifiers must be distinct")
    return cases


@dataclass
class BoundaryData:
    fixed_dofs: np.ndarray
    values: np.ndarray
    force: np.ndarray


def resolve_boundary_data(conditions, nx, ny, *, case_id="case-1", default_case_id="case-1", length=1, height=1):
    """Resolve shared supports and one independent load case, rejecting mechanisms."""
    fixed = {}
    force = np.zeros(2 * (nx + 1) * (ny + 1))
    for condition in conditions:
        nodes, weights = condition.selection.nodes_and_weights(nx, ny)
        if condition.kind != "force":
            components = {"fixed": "xy", "restrain_x": "x", "restrain_y": "y"}.get(condition.kind, condition.components)
            for component, letter in enumerate("xy"):
                if letter not in components:
                    continue
                value = (condition.ux, condition.uy)[component] if condition.kind == "displacement" else 0.0
                for node in nodes:
                    dof = 2 * int(node) + component
                    if dof in fixed and not math.isclose(fixed[dof], value, rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError("Conflicting prescribed displacements on the same node and direction")
                    fixed[dof] = value
        elif (condition.case_id or default_case_id) == case_id:
            radians = math.radians(condition.angle)
            direction = np.array([math.cos(radians), math.sin(radians)])
            direction[np.abs(direction) < 1e-14] = 0
            force.reshape(-1, 2)[nodes] += condition.magnitude * weights[:, None] * direction
    fixed_dofs = np.array(sorted(fixed), dtype=np.int64)
    modes = []
    scale = max(length, height)
    for dof in fixed_dofs:
        node, component = divmod(int(dof), 2)
        x, y = node % (nx+1) / nx * length / scale, node // (nx+1) / ny * height / scale
        modes.append((1, 0, -y) if component == 0 else (0, 1, x))
    if not modes or np.linalg.matrix_rank(np.array(modes)) < 3:
        raise ValueError("Restrain rigid motion using at least two distinct nodes and suitable directions")
    if len(fixed_dofs) == len(force):
        raise ValueError("Leave at least one node unconstrained")
    free_force = force.copy()
    free_force[fixed_dofs] = 0
    values = np.zeros_like(force)
    values[fixed_dofs] = [fixed[int(dof)] for dof in fixed_dofs]
    if not np.any(free_force) and not np.any(values):
        raise ValueError("Apply a nonzero resultant force to at least one unconstrained node")
    return BoundaryData(fixed_dofs, values, force)


def resolve_conditions(conditions, nx, ny, case_id="case-1"):
    """Compatibility view for existing zero-displacement state solvers."""
    data = resolve_boundary_data(conditions, nx, ny, case_id=case_id, default_case_id=case_id)
    return data.fixed_dofs, data.force
