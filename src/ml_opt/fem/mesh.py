"""A rectangular mesh with counterclockwise Q4 element connectivity."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..boundaries import normalise_conditions, resolve_boundary_data


@dataclass(frozen=True)
class StructuredMesh:
    nx: int
    ny: int
    length: float
    height: float
    boundary_conditions: tuple | None = None
    load_case_id: str = "case-1"

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1 or self.length <= 0 or self.height <= 0:
            raise ValueError("Mesh dimensions and element counts must be positive")
        object.__setattr__(self, "boundary_conditions", normalise_conditions(self.boundary_conditions))

    @property
    def element_count(self) -> int:
        return self.nx * self.ny

    @property
    def dof_count(self) -> int:
        return 2 * (self.nx + 1) * (self.ny + 1)

    @cached_property
    def coordinates(self) -> np.ndarray:
        x, y = np.meshgrid(np.linspace(0, self.length, self.nx + 1),
                           np.linspace(0, self.height, self.ny + 1))
        return np.column_stack((x.ravel(), y.ravel()))

    @cached_property
    def elements(self) -> np.ndarray:
        nodes = np.arange((self.nx + 1) * (self.ny + 1)).reshape(self.ny + 1, self.nx + 1)
        return np.stack((nodes[:-1, :-1], nodes[:-1, 1:], nodes[1:, 1:],
                         nodes[1:, :-1]), axis=-1).reshape(-1, 4)

    @cached_property
    def element_dofs(self) -> np.ndarray:
        return (2 * self.elements[:, :, None] + np.arange(2)).reshape(-1, 8)

    @cached_property
    def fixed_dofs(self) -> np.ndarray:
        if self.boundary_conditions is not None:
            return self.boundary_data.fixed_dofs
        nodes = np.arange(0, (self.nx + 1) * (self.ny + 1), self.nx + 1)
        return (2 * nodes[:, None] + np.arange(2)).ravel()

    @cached_property
    def boundary_data(self):
        return resolve_boundary_data(self.boundary_conditions, self.nx, self.ny,
            case_id=self.load_case_id, default_case_id=self.load_case_id, length=self.length, height=self.height)

    @cached_property
    def prescribed_values(self):
        return np.zeros(self.dof_count) if self.boundary_conditions is None else self.boundary_data.values

    @cached_property
    def free_dofs(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.dof_count), self.fixed_dofs)

    def cantilever_force(self, magnitude: float, angle: float = -90.0,
                         node: tuple[int, int] | None = None) -> np.ndarray:
        """Resolve manual assignments, or apply the legacy cantilever load."""
        if self.boundary_conditions is not None:
            return self.boundary_data.force.copy()
        force = np.zeros(self.dof_count)
        direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        direction[np.abs(direction) < 1e-14] = 0
        if node is not None:
            ix, iy = node
            if not 1 <= ix <= self.nx or not 0 <= iy <= self.ny:
                raise ValueError("Choose a mesh node outside the clamped left edge")
            index = iy * (self.nx + 1) + ix
            force[2 * index:2 * index + 2] = magnitude * direction
            return force
        lower = self.ny // 2
        fraction = self.ny / 2 - lower
        index = lower * (self.nx + 1) + self.nx
        force[2 * index:2 * index + 2] = magnitude * (1 - fraction) * direction
        if fraction:
            upper = index + self.nx + 1
            force[2 * upper:2 * upper + 2] = magnitude * fraction * direction
        return force
