"""Train on full-order optimisation trajectories, on one fixed cantilever mesh.

Snapshot columns use free displacement DOFs, with no mean subtraction. POD
energy measures squared displacement snapshot norms, not elastic strain energy and
not an error bound on unseen cases or compliance sensitivities.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from ..config import CantileverConfig
from ..fem.mesh import StructuredMesh
from ..runner import optimise
from .operators import OPERATOR_FORMAT, prepare_element_operators


@dataclass(frozen=True)
class SnapshotConfig:
    nx: int = 30
    ny: int = 10
    length: float = 3.0
    height: float = 1.0
    load_heights: tuple[float, ...] = (0.25, 0.5, 0.75)
    load_angles: tuple[float, ...] = (-90.0,)
    volume_fractions: tuple[float, ...] = (0.3, 0.4, 0.5)
    iterations_per_case: int = 20
    snapshot_stride: int = 2
    filter_radius: float = 0.1
    projection_beta: float = 2.0
    projection_eta: float = 0.5

    def __post_init__(self):
        for name in ("nx", "ny", "iterations_per_case", "snapshot_stride"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("load_heights", "load_angles", "volume_fractions"):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must not be empty")
            if any(not np.isfinite(x) or (name == "load_heights" and not 0 <= x <= 1)
                    or (name == "volume_fractions" and not 0 < x < 1) for x in values):
                raise ValueError(f"{name} contains an invalid value")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct values")
        # Validate common physics once without materialising the Cartesian product.
        CantileverConfig(nx=self.nx, ny=self.ny, length=self.length, height=self.height,
            max_iterations=self.iterations_per_case, filter_radius=self.filter_radius,
            projection_beta=self.projection_beta, projection_eta=self.projection_eta)
        if len({int(np.floor(height*self.ny + .5)) for height in self.load_heights}) != len(self.load_heights):
            raise ValueError("Load heights must map to distinct mesh rows; use fewer heights or more y elements")

    @property
    def case_count(self):
        return len(self.load_heights) * len(self.load_angles) * len(self.volume_fractions)

    @property
    def maximum_snapshots(self):
        # Initial state, every stride, and the final state (also on early stop).
        per_case = 1 + (self.iterations_per_case + self.snapshot_stride - 1) // self.snapshot_stride
        return self.case_count * per_case

    @property
    def estimated_snapshot_bytes(self):
        """Displacement/density arrays only; saving and POD need additional RAM."""
        free_count = 2 * self.nx * (self.ny + 1)
        return 8 * (free_count + self.nx*self.ny) * self.maximum_snapshots

    def cases(self):
        return [CantileverConfig(nx=self.nx, ny=self.ny, length=self.length, height=self.height,
                    load_node_x=self.nx, load_node_y=int(np.floor(height * self.ny + 0.5)),
                    load_angle=angle, volume_fraction=volume, max_iterations=self.iterations_per_case,
                    filter_radius=self.filter_radius, projection_beta=self.projection_beta,
                    projection_eta=self.projection_eta)
                for height, angle, volume in product(self.load_heights, self.load_angles, self.volume_fractions)]


def _now():
    return datetime.now(timezone.utc).isoformat()


def save_metadata(directory: Path, metadata: dict):
    temporary = directory / "metadata.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "metadata.json")


def _save_arrays(path: Path, *, should_commit=lambda: True, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    if not should_commit():
        temporary.unlink(missing_ok=True)
        return False
    temporary.replace(path)
    return True


def read_metadata(directory: Path):
    return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))


def generate_snapshots(config: SnapshotConfig, directory: Path, *, emit=lambda event: None,
                       stop_requested=lambda: False):
    """Capture accepted outer iterates, never GCMMA trials or fabricated fields.

    Cancellation preserves collected snapshots as a partial dataset; each
    captured solution still satisfies the full-order equilibrium tolerance.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cases = config.cases()
    mesh = StructuredMesh(config.nx, config.ny, config.length, config.height)
    metadata = {"schema_version": 1, "id": directory.name, "created_at": _now(),
                "status": "generating", "settings": json.loads(json.dumps(asdict(config))),
                "case_count": len(cases), "completed_cases": 0, "snapshot_count": 0,
                "maximum_snapshots": config.maximum_snapshots,
                "estimated_snapshot_bytes": config.estimated_snapshot_bytes, "free_dof_count": len(mesh.free_dofs),
                "snapshot_layout": "columns; free displacement DOFs; zero clamped displacement",
                "cases": [asdict(case) for case in cases], "case_results": []}
    save_metadata(directory, metadata)
    displacements, densities, case_indices, iterations, compliances, residuals = [], [], [], [], [], []
    try:
        for case_index, case in enumerate(cases):
            if stop_requested():
                break
            emit({"event": "case_started", "case_index": case_index,
                  "case_count": len(cases), "config": asdict(case)})
            last_saved = -1

            def capture(row, displacement, density, *, force=False):
                nonlocal last_saved
                if row.iteration == last_saved or (not force and row.iteration % config.snapshot_stride):
                    return
                last_saved = row.iteration
                displacements.append(displacement[mesh.free_dofs].copy())
                densities.append(density.copy())
                case_indices.append(case_index)
                iterations.append(row.iteration)
                compliances.append(row.compliance)
                residuals.append(row.linear_residual)
                emit({"event": "snapshot", "case_index": case_index, "case_count": len(cases),
                      "iteration": row.iteration, "snapshot_count": len(displacements),
                      "maximum_snapshots": config.maximum_snapshots, "compliance": row.compliance,
                      "residual": row.linear_residual, "density": density.tolist()})

            result = optimise(case, state_snapshot_callback=capture, stop_requested=stop_requested)
            capture(result.history[-1], result.evaluation.displacement, result.evaluation.density, force=True)
            metadata["case_results"].append({"case_index": case_index, "stop_reason": result.stop_reason,
                        "iterations": result.history[-1].iteration, "compliance": result.evaluation.compliance})
            if result.stop_reason != "cancelled":
                metadata["completed_cases"] += 1
            if stop_requested():
                break
        emit({"event": "saving_snapshots", "snapshot_count": len(displacements)})
        metadata.update(status="cancelled" if stop_requested() else "snapshots_ready",
                        snapshot_count=len(displacements), updated_at=_now())
        if displacements:
            metadata["last_snapshot"] = {"case_index": case_indices[-1], "iteration": iterations[-1],
                                         "compliance": compliances[-1], "residual": residuals[-1]}
            _save_arrays(directory / "snapshots.npz", displacements=np.column_stack(displacements),
                densities=np.column_stack(densities), case_indices=np.array(case_indices, dtype=np.int64),
                iterations=np.array(iterations, dtype=np.int64), compliances=np.array(compliances),
                residuals=np.array(residuals), free_dofs=mesh.free_dofs, coordinates=mesh.coordinates,
                elements=mesh.elements, metadata_json=np.array(json.dumps(metadata)))
        save_metadata(directory, metadata)
        return metadata
    except Exception as error:
        metadata.update(status="failed", error=str(error), updated_at=_now())
        save_metadata(directory, metadata)
        raise


def pod_basis(snapshots: np.ndarray, energy_target=0.999, max_modes=32, normalize=True):
    """Uncentred Euclidean POD; optional unit column scaling weights states equally."""
    if not np.isfinite(energy_target) or not 0 < energy_target <= 1:
        raise ValueError("Energy target must lie in (0, 1]")
    if type(max_modes) is not int or not 1 <= max_modes <= 256:
        raise ValueError("Maximum modes must be an integer between 1 and 256")
    if type(normalize) is not bool:
        raise ValueError("Snapshot normalisation must be a boolean")
    snapshots = np.asarray(snapshots, dtype=np.float64)
    if snapshots.ndim != 2 or not all(snapshots.shape) or not np.all(np.isfinite(snapshots)):
        raise ValueError("Expected a nonempty finite displacement snapshot matrix")
    norms = np.linalg.norm(snapshots, axis=0)
    if np.any(norms <= np.finfo(float).tiny) or not np.all(np.isfinite(norms)):
        raise ValueError("Snapshots must have finite nonzero displacement norms")
    matrix = snapshots / norms if normalize else snapshots.copy()
    # A common scaling does not alter modes or energy fractions; it avoids overflow.
    scale = float(np.max(np.abs(matrix)))
    modes, singular, _ = svd(matrix / scale, full_matrices=False, check_finite=False)
    numeric_rank = int(np.count_nonzero(singular > np.finfo(float).eps * max(matrix.shape) * singular[0]))
    weights = (singular / singular[0])**2
    cumulative = np.minimum(np.cumsum(weights) / weights.sum(), 1.0)
    required = min(int(np.searchsorted(cumulative, energy_target)) + 1, numeric_rank)
    rank = min(required, max_modes)
    basis = modes[:, :rank].copy()
    errors = []
    for start in range(0, snapshots.shape[1], 32):
        block = snapshots[:, start:start+32]
        errors.extend((np.linalg.norm(block - basis @ (basis.T @ block), axis=0)
                       / norms[start:start+32]).tolist())
    summary = {"rank": rank, "numerical_rank": numeric_rank, "energy_target": energy_target,
               "max_modes": max_modes, "normalize_snapshots": normalize,
               "captured_energy": float(cumulative[rank-1]),
               "target_met": bool(cumulative[rank-1] >= energy_target - 32*np.finfo(float).eps),
               "mean_training_error": float(np.mean(errors)), "max_training_error": float(np.max(errors)),
               "singular_values": (singular * scale).tolist(), "cumulative_energy": cumulative.tolist(),
               "metric": "Euclidean displacement POD; no mean subtraction",
               "validation": "POD training reconstruction only; held-out validation reports are stored separately"}
    return basis, norms, np.array(errors), summary


def build_basis(directory: Path, *, energy_target=0.999, max_modes=32, normalize=True,
                emit=lambda event: None, stop_requested=lambda: False):
    directory = Path(directory)
    metadata = read_metadata(directory)
    if not metadata.get("snapshot_count") or metadata["status"] not in ("snapshots_ready", "cancelled", "basis_ready"):
        raise ValueError("Generate a nonempty snapshot dataset before building a basis")
    if stop_requested():
        return None
    emit({"event": "basis_started", "snapshot_count": metadata["snapshot_count"]})
    with np.load(directory / "snapshots.npz", allow_pickle=False) as archive:
        snapshots = archive["displacements"]
        free_dofs = archive["free_dofs"]
    basis, norms, errors, summary = pod_basis(snapshots, energy_target, max_modes, normalize)
    if stop_requested():
        return None
    emit({"event": "operators_started", "rank": summary["rank"]})
    config = CantileverConfig(**metadata["cases"][0])
    mesh = StructuredMesh(config.nx, config.ny, config.length, config.height)
    if not np.array_equal(free_dofs, mesh.free_dofs):
        raise ValueError("Snapshot free displacement DOFs do not match the model")
    operators = prepare_element_operators(config, basis, stop_requested=stop_requested)
    if operators is None or stop_requested():
        return None
    summary.update(operator_format=OPERATOR_FORMAT, operator_bytes=operators.nbytes)
    summary["built_at"] = _now()
    # A validation report belongs to one frozen basis, never to a later rebuild.
    metadata.pop("validation", None)
    metadata.update(status="basis_ready", basis=summary, updated_at=_now())
    # Include the DOF mapping and complete provenance in the portable basis artifact.
    saved = _save_arrays(directory / "basis.npz", should_commit=lambda: not stop_requested(),
                 basis=basis, free_dofs=free_dofs,
                 singular_values=np.array(summary["singular_values"]),
                 cumulative_energy=np.array(summary["cumulative_energy"]),
                 snapshot_norms=norms, training_relative_errors=errors,
                 reduced_element_stiffness=operators,
                 metadata_json=np.array(json.dumps(metadata)))
    if not saved:
        return None
    save_metadata(directory, metadata)
    return metadata
