"""Element regions excluded from the design variables."""

from dataclasses import dataclass

import numpy as np

from .boundaries import MeshSelection


@dataclass(frozen=True)
class NonDesignRegion:
    kind: str
    selection: MeshSelection

    def __post_init__(self):
        if self.kind not in ("solid", "void"):
            raise ValueError("A non-design region must be solid or void")
        if isinstance(self.selection, dict):
            try:
                object.__setattr__(self, "selection", MeshSelection(**self.selection))
            except TypeError as error:
                raise ValueError("Invalid non-design region selection") from error
        if not isinstance(self.selection, MeshSelection) or self.selection.kind != "surface":
            raise ValueError("Select a surface for a non-design region")

    def elements(self, nx, ny):
        coordinates = [int(np.floor(value * size + .5)) for point in
                       (self.selection.start, self.selection.end) for value, size in zip(point, (nx, ny))]
        x0, y0, x1, y1 = coordinates
        if x0 == x1 or y0 == y1:
            raise ValueError("The non-design region is smaller than one element; refine the mesh")
        return (np.arange(y0, y1)[:, None] * nx + np.arange(x0, x1)).ravel()


def normalise_regions(values):
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)) or len(values) > 128:
        raise ValueError("Provide at most 128 non-design regions")
    try:
        return tuple(value if isinstance(value, NonDesignRegion) else NonDesignRegion(**value) for value in values)
    except TypeError as error:
        raise ValueError("Invalid non-design region") from error


def region_densities(regions, nx, ny):
    """NaN marks a designable element; zero/one are fixed physical densities."""
    fixed = np.full(nx * ny, np.nan)
    for region in regions:
        indices, value = region.elements(nx, ny), float(region.kind == "solid")
        assigned = fixed[indices]
        if np.any(np.isfinite(assigned) & (assigned != value)):
            raise ValueError("Solid and void non-design regions overlap")
        fixed[indices] = value
    if not np.any(np.isnan(fixed)):
        raise ValueError("Leave at least one element available for design")
    return fixed
