"""Frozen-basis Galerkin elasticity and portable basis loading."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix

from ..config import CantileverConfig
from ..fem.element import q4_stiffness
from ..fem.elasticity import ElasticityState
from ..fem.mesh import StructuredMesh


class GalerkinSolver:
    """Solve V.T K V a = V.T f with one frozen basis throughout a run."""

    def __init__(self, config: CantileverConfig, basis):
        self.mesh = StructuredMesh(config.nx, config.ny, config.length, config.height)
        self.basis = np.array(basis, dtype=np.float64, copy=True)
        count = len(self.mesh.free_dofs)
        if (self.basis.ndim != 2 or self.basis.shape[0] != count or not 1 <= self.basis.shape[1] <= count
                or not np.all(np.isfinite(self.basis))):
            raise ValueError("Basis must contain finite free-DOF columns for this mesh")
        if not np.allclose(self.basis.T @ self.basis, np.eye(self.basis.shape[1]), atol=1e-8, rtol=1e-8):
            raise ValueError("Expected an orthonormal displacement basis")
        self.basis.flags.writeable = False
        self.element_stiffness = q4_stiffness(config.length/config.nx, config.height/config.ny,
                                              config.poisson_ratio, config.thickness)
        node = None if config.load_node_x is None else (config.load_node_x, config.load_node_y)
        self.force = self.mesh.cantilever_force(config.load, config.load_angle, node)
        self.rhs = self.force[self.mesh.free_dofs]
        self.reduced_rhs = self.basis.T @ self.rhs
        mapping = np.full(self.mesh.dof_count, -1, dtype=np.int64)
        mapping[self.mesh.free_dofs] = np.arange(count)
        dofs = mapping[self.mesh.element_dofs]
        rows, columns = np.repeat(dofs, 8, axis=1).ravel(), np.tile(dofs, (1, 8)).ravel()
        self.mask = (rows >= 0) & (columns >= 0)
        self.rows, self.columns = rows[self.mask], columns[self.mask]
        self.last_state = None
        self.solve_count = 0
        self._closed = False

    def solve(self, moduli):
        if self._closed:
            raise RuntimeError("The reduced solver is closed")
        moduli = np.asarray(moduli, dtype=np.float64)
        if moduli.shape != (self.mesh.element_count,) or not np.all(np.isfinite(moduli)) or np.any(moduli <= 0):
            raise ValueError("Expected one finite positive modulus per element")
        entries = (moduli[:, None, None] * self.element_stiffness).ravel()[self.mask]
        matrix = coo_matrix((entries, (self.rows, self.columns)),
                            shape=(len(self.rhs), len(self.rhs))).tocsr()
        reduced = self.basis.T @ (matrix @ self.basis)
        reduced = .5 * (reduced + reduced.T)
        coefficients = cho_solve(cho_factor(reduced, check_finite=True), self.reduced_rhs)
        free_u = self.basis @ coefficients
        displacement = np.zeros(self.mesh.dof_count)
        displacement[self.mesh.free_dofs] = free_u
        element_u = displacement[self.mesh.element_dofs]
        energy = np.einsum("ei,ij,ej->e", element_u, self.element_stiffness, element_u)
        residual = float(np.linalg.norm(matrix @ free_u - self.rhs) / np.linalg.norm(self.rhs))
        if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(energy)) or not np.isfinite(residual):
            raise ValueError("The reduced solve produced nonfinite results")
        self.last_state = ElasticityState(displacement, float(self.force @ displacement), energy, residual, 1)
        self.solve_count += 1
        return self.last_state

    def close(self):
        self._closed = True


def load_basis(path, config: CantileverConfig, *, expected_sha256=None):
    """Read a portable basis once; reject incompatible geometry/material/DOFs.

    Load position, direction, magnitude, SIMP and filtering may vary. Their
    accuracy is governed by basis coverage, rather than coordinate compatibility.
    """
    try:
        payload = Path(path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("The selected ROM basis changed; refresh the models and start again")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            basis = archive["basis"]
            free_dofs = archive["free_dofs"]
            metadata = json.loads(str(archive["metadata_json"]))
        trained = CantileverConfig(**metadata["cases"][0])
        for name in ("nx", "ny", "length", "height", "poisson_ratio", "thickness"):
            if getattr(config, name) != getattr(trained, name):
                raise ValueError(f"ROM {name} must match its training model ({getattr(trained, name)})")
        mesh = StructuredMesh(config.nx, config.ny, config.length, config.height)
        if not np.array_equal(free_dofs, mesh.free_dofs):
            raise ValueError("ROM free displacement DOFs do not match the model")
        rank = metadata["basis"]["rank"]
        if (basis.dtype != np.float64 or basis.ndim != 2 or basis.shape != (len(mesh.free_dofs), rank)
                or not 1 <= rank <= 256 or not np.all(np.isfinite(basis))):
            raise ValueError("The saved ROM basis has invalid dimensions or values")
        if not np.allclose(basis.T @ basis, np.eye(rank), atol=1e-8, rtol=1e-8):
            raise ValueError("The saved ROM basis is not orthonormal")
        node = None if config.load_node_x is None else (config.load_node_x, config.load_node_y)
        force = mesh.cantilever_force(config.load, config.load_angle, node)[mesh.free_dofs]
        if np.linalg.norm(basis.T @ force) <= 1e-12 * np.linalg.norm(force):
            raise ValueError("The selected ROM cannot represent this load; choose a different basis or use FOM")
    except (OSError, KeyError, TypeError, IndexError) as error:
        raise ValueError("Could not load a complete ROM basis artifact") from error
    return basis, {"training_id": metadata["id"], "rank": rank, "basis_sha256": digest,
                   "built_at": metadata["basis"]["built_at"], "model": "frozen Galerkin ROM"}


