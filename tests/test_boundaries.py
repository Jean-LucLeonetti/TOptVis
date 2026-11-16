"""Selection mapping, force conservation, and boundary configuration validation."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from ml_opt.boundaries import BoundaryCondition, MeshSelection, resolve_conditions
from ml_opt.config import CantileverConfig
from ml_opt.fem.mesh import StructuredMesh


def fixed(start=(0, 0), end=(0, 1)):
    return BoundaryCondition("fixed", MeshSelection("line", start, end))


def force(kind="point", start=(1, .5), end=None, magnitude=1, angle=-90):
    return BoundaryCondition("force", MeshSelection(kind, start, end), magnitude, angle)


def test_manual_default_matches_existing_mesh_and_force():
    legacy = StructuredMesh(8, 4, 3, 1)
    manual = StructuredMesh(8, 4, 3, 1, (fixed(), force()))
    np.testing.assert_array_equal(manual.fixed_dofs, legacy.fixed_dofs)
    np.testing.assert_array_equal(manual.free_dofs, legacy.free_dofs)
    np.testing.assert_array_equal(manual.cantilever_force(99), legacy.cantilever_force(1, node=(8, 2)))


def test_custom_supports_replace_left_clamp_and_allow_left_edge_loads():
    mesh = StructuredMesh(8, 4, 3, 1, (fixed((0, 0), (1, 0)), force(start=(0, 1))))
    np.testing.assert_array_equal(mesh.fixed_dofs, np.arange(18))
    assert 2 * (4 * 9) in mesh.free_dofs
    assert mesh.cantilever_force(1)[2 * (4 * 9) + 1] == -1


@pytest.mark.parametrize("nx,ny", [(4, 2), (8, 4), (32, 16)])
@pytest.mark.parametrize("selection", [
    MeshSelection("point", (1, .5)),
    MeshSelection("line", (1, 0), (1, 1)),
    MeshSelection("line", (.25, 1), (.75, 1)),
    MeshSelection("surface", (.5, 0), (1, 1)),
])
def test_force_resultant_and_moment_are_preserved_on_refinement(nx, ny, selection):
    condition = BoundaryCondition("force", selection, 7, -37)
    _, loads = resolve_conditions((fixed(), condition), nx, ny)
    force_xy = loads.reshape(-1, 2)
    direction = 7 * np.array([np.cos(np.deg2rad(-37)), np.sin(np.deg2rad(-37))])
    np.testing.assert_allclose(force_xy.sum(axis=0), direction, atol=1e-14)
    mesh = StructuredMesh(nx, ny, 3, 1)
    centre = (np.array(selection.start) + np.array(selection.end or selection.start)) / 2 * [3, 1]
    moment = np.sum(mesh.coordinates[:, 0] * force_xy[:, 1] - mesh.coordinates[:, 1] * force_xy[:, 0])
    np.testing.assert_allclose(moment, centre[0] * direction[1] - centre[1] * direction[0], atol=1e-13)


def test_line_and_surface_use_consistent_endpoint_and_corner_weights():
    nodes, weights = MeshSelection("line", (1, 0), (1, 1)).nodes_and_weights(4, 2)
    np.testing.assert_array_equal(nodes, [4, 9, 14])
    np.testing.assert_allclose(weights, [.25, .5, .25])
    nodes, weights = MeshSelection("surface", (0, 0), (1, 1)).nodes_and_weights(2, 2)
    np.testing.assert_array_equal(nodes, np.arange(9))
    np.testing.assert_allclose(weights, np.array([1, 2, 1, 2, 4, 2, 1, 2, 1]) / 16)


def test_overlapping_forces_add_and_supported_load_components_are_retained():
    _, loads = resolve_conditions((fixed(), force("line", (0, 1), (1, 1), 2),
                                  force(start=(1, 1), magnitude=3)), 4, 2)
    assert loads[2 * (2 * 5) + 1] == -.25
    assert loads[2 * (2 * 5 + 4) + 1] == -3.25
    np.testing.assert_allclose(loads.reshape(-1, 2).sum(axis=0), [0, -5])


@pytest.mark.parametrize("conditions,message", [
    ((force(),), "two distinct"),
    ((BoundaryCondition("fixed", MeshSelection("point", (0, 0))), force()), "two distinct"),
    ((fixed(),), "nonzero"),
    ((fixed(), force(start=(0, .5))), "unconstrained"),
    ((fixed(), force(angle=-90), force(angle=90)), "nonzero"),
    ((BoundaryCondition("fixed", MeshSelection("surface", (0, 0), (1, 1))), force()), "unconstrained"),
])
def test_invalid_or_unloaded_support_layout_is_rejected(conditions, message):
    with pytest.raises(ValueError, match=message):
        resolve_conditions(conditions, 8, 4)


@pytest.mark.parametrize("kind,start,end", [
    ("point", (float("nan"), 0), None),
    ("point", (False, 0), None),
    ("point", (-.1, 0), None),
    ("line", (0, 0), (1, 1)),
    ("line", (0, 0), None),
    ("surface", (0, 0), (float("inf"), 1)),
])
def test_invalid_selection_coordinates_are_rejected(kind, start, end):
    with pytest.raises(ValueError):
        MeshSelection(kind, start, end)


@pytest.mark.parametrize("selection", [MeshSelection("line", (0, 0), (0, .01)),
                                         MeshSelection("surface", (0, 0), (.01, 1))])
def test_selections_that_collapse_on_a_coarse_mesh_are_rejected(selection):
    with pytest.raises(ValueError, match="refine"):
        selection.nodes_and_weights(4, 2)


def test_serialised_conditions_roundtrip_without_aliasing():
    values = [asdict(fixed()), asdict(force())]
    config = CantileverConfig(nx=8, ny=4, boundary_conditions=values)
    values[0]["selection"]["start"] = (1, 1)
    restored = CantileverConfig(**json.loads(json.dumps(asdict(config))))
    assert restored == config
    assert restored.boundary_conditions[0].selection.start == (0, 0)
    assert hash(restored.boundary_conditions) == hash(config.boundary_conditions)


def test_reversed_selections_are_canonical():
    assert MeshSelection("surface", (1, 1), (.25, .5)) == MeshSelection("surface", (.25, .5), (1, 1))
