"""Connection selections and an independent face-connected material diagnostic."""

from collections import deque
import numpy as np

from .boundaries import MeshSelection


def normalise_connection_regions(values):
    if not isinstance(values, (list, tuple)) or len(values) > 16:
        raise ValueError("Provide at most 16 connection regions of each type")
    try:
        return tuple(value if isinstance(value, MeshSelection) else MeshSelection(**value) for value in values)
    except TypeError as error:
        raise ValueError("Invalid connection selection") from error


def connection_elements(selection, nx, ny):
    # Validate snapping, then use selected surface cells or cells touching a
    # selected point/line. A cell-centred diffusion grid has no diagonal links.
    selection.nodes_and_weights(nx, ny)
    x0, y0 = (int(np.floor(v*n+.5)) for v, n in zip(selection.start, (nx, ny)))
    x1, y1 = (int(np.floor(v*n+.5)) for v, n in zip(selection.end or selection.start, (nx, ny)))
    xs = range(x0, x1) if x0 != x1 else range(max(0, x0-1), min(nx, x0+1))
    ys = range(y0, y1) if y0 != y1 else range(max(0, y0-1), min(ny, y0+1))
    return np.array([y*nx+x for y in ys for x in xs], dtype=int)


def connection_regions(anchors, targets, nx, ny):
    if not anchors and not targets:
        return np.array([], dtype=int), []
    if not anchors or not targets:
        raise ValueError("Assign a connection anchor and at least one connection target")
    sink = np.unique(np.concatenate([connection_elements(s, nx, ny) for s in anchors]))
    sources = [connection_elements(s, nx, ny) for s in targets]
    if len(sink) == nx*ny or any(np.intersect1d(sink, s).size for s in sources):
        raise ValueError("Connection anchors and targets must occupy separate mesh cells")
    return sink, sources


def connected_targets(density, nx, ny, anchors, targets, threshold=.5):
    solid = np.asarray(density) >= threshold
    reached = set(int(i) for i in anchors if solid[i])
    pending = deque(reached)
    while pending:
        i = pending.popleft()
        x, y = i % nx, i // nx
        for j in (i-1 if x else -1, i+1 if x+1<nx else -1,
                  i-nx if y else -1, i+nx if y+1<ny else -1):
            if j >= 0 and solid[j] and j not in reached:
                reached.add(j)
                pending.append(j)
    return [all(int(i) in reached for i in source) for source in targets]
