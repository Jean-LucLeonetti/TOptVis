"""A rectangular mesh with counterclockwise Q4 element connectivity."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class StructuredMesh:
    nx: int
    ny: int
    length: float
    height: float

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1 or self.length <= 0 or self.height <= 0:
            raise ValueError("Mesh dimensions and element counts must be positive")

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
        nodes = np.arange(0, (self.nx + 1) * (self.ny + 1), self.nx + 1)
        return (2 * nodes[:, None] + np.arange(2)).ravel()

    @cached_property
    def free_dofs(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.dof_count), self.fixed_dofs)

    def cantilever_force(self, magnitude: float) -> np.ndarray:
        """Downward unit-point load at the free-edge midpoint, split on odd meshes."""
        force = np.zeros(self.dof_count)
        middle = self.ny / 2
        lower = int(np.floor(middle))
        fraction = middle - lower
        node = lower * (self.nx + 1) + self.nx
        force[2 * node + 1] = -magnitude * (1 - fraction)
        if fraction:
            force[2 * (node + self.nx + 1) + 1] = -magnitude * fraction
        return force
