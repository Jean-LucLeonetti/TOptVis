"""Exact affine element operators for frozen-basis Galerkin elasticity."""

import numpy as np

from ..fem.element import q4_stiffness
from ..fem.mesh import StructuredMesh


OPERATOR_FORMAT = "symmetric_upper_v1"
MAX_OPERATOR_BYTES = 512 * 1024**2


def operator_shape(element_count, rank):
    return (element_count, rank * (rank + 1) // 2)


def check_operators(operators, element_count, rank):
    shape = operator_shape(element_count, rank)
    if np.prod(shape) * 8 > MAX_OPERATOR_BYTES:
        raise ValueError("Reduced element operators exceed 512 MiB; choose fewer basis modes")
    if (operators.dtype != np.float64 or operators.shape != shape or
            not np.all(np.isfinite(operators))):
        raise ValueError("Saved reduced element operators have invalid dimensions or values")
    return operators


def prepare_element_operators(config, basis, *, stop_requested=lambda: False):
    """Pack the upper triangles of V_e.T K_e(E=1) V_e; no approximation.

    The element basis has zeros at clamped DOFs. Blocking bounds temporary
    storage; the persistent packed array is limited separately before allocation.
    Cancellation returns None without publishing a partially built operator.
    """
    if stop_requested():
        return None
    mesh = StructuredMesh(config.nx, config.ny, config.length, config.height, config.boundary_conditions, config.load_cases[0].id)
    rank = basis.shape[1]
    shape = operator_shape(mesh.element_count, rank)
    if np.prod(shape) * 8 > MAX_OPERATOR_BYTES:
        raise ValueError("Reduced element operators exceed 512 MiB; choose fewer basis modes")
    lifted = np.zeros((mesh.dof_count, rank))
    lifted[mesh.free_dofs] = basis
    stiffness = q4_stiffness(config.length/config.nx, config.height/config.ny,
                             config.poisson_ratio, config.thickness)
    upper = np.triu_indices(rank)
    operators = np.empty(shape)
    block_size = max(1, min(256, (16 * 1024**2) // (8 * (3*rank*rank + 16*rank))))
    for start in range(0, mesh.element_count, block_size):
        if stop_requested():
            return None
        element_basis = lifted[mesh.element_dofs[start:start+block_size]]
        blocks = element_basis.swapaxes(1, 2) @ (stiffness @ element_basis)
        blocks = .5 * (blocks + blocks.swapaxes(1, 2))
        operators[start:start+block_size] = blocks[:, upper[0], upper[1]]
    operators.flags.writeable = False
    return operators
