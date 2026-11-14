"""Frozen-basis Galerkin validation on held-out full-order design trajectories."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from ..config import CantileverConfig
from ..fem.mesh import StructuredMesh
from ..problem import CantileverProblem
from ..runner import optimise
from .training import SnapshotConfig, _now, _save_arrays, read_metadata, save_metadata
from .solver import GalerkinSolver


@dataclass(frozen=True)
class ValidationConfig:
    load_heights: tuple[float, ...] = (.4, .6)
    load_angles: tuple[float, ...] = (-75.,)
    volume_fractions: tuple[float, ...] = (.35, .45)
    iterations_per_case: int = 10
    comparison_stride: int = 5
    displacement_tolerance: float = .05
    compliance_tolerance: float = .01
    gradient_tolerance: float = .1

    def __post_init__(self):
        for name in ("displacement_tolerance", "compliance_tolerance", "gradient_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        for name in ("iterations_per_case", "comparison_stride"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 300:
                raise ValueError(f"{name} must be an integer between 1 and 300")
        for name, lower, upper, exclusive in (("load_heights", 0, 1, False),
                    ("load_angles", -180, 180, False), ("volume_fractions", 0, 1, True)):
            values = getattr(self, name)
            if not values or len(values) > 16 or len(set(values)) != len(values):
                raise ValueError(f"{name} needs between 1 and 16 distinct values")
            if any(not np.isfinite(value) or (not lower < value < upper if exclusive else
                                              not lower <= value <= upper) for value in values):
                raise ValueError(f"{name} contains an invalid value")
        # Mesh-node snapping belongs to validation_cases, on the selected mesh.
        if self.case_count > 16 or self.maximum_states > 128:
            raise ValueError("Validation allows at most 16 cases and 128 comparison states")

    @property
    def case_count(self):
        return len(self.load_heights) * len(self.load_angles) * len(self.volume_fractions)

    @property
    def maximum_states(self):
        return self.case_count * (1 + (self.iterations_per_case + self.comparison_stride - 1) // self.comparison_stride)


def _case_key(config):
    angle = np.deg2rad(config.load_angle)
    return (config.load_node_x, config.load_node_y, round(float(np.cos(angle)), 12),
            round(float(np.sin(angle)), 12), config.volume_fraction)


def validation_cases(metadata, options: ValidationConfig):
    if metadata.get("status") != "basis_ready" or not metadata.get("basis"):
        raise ValueError("Build a reduced basis before running validation")
    base = CantileverConfig(**metadata["cases"][0])
    if getattr(base, "robust", False):
        raise ValueError("This validation workflow requires a single-field training dataset")
    settings = metadata["settings"]
    sampling = SnapshotConfig(**{**settings, "load_heights": options.load_heights,
        "load_angles": options.load_angles, "volume_fractions": options.volume_fractions,
        "iterations_per_case": options.iterations_per_case, "snapshot_stride": options.comparison_stride})
    cases = [replace(base, load_node_x=case.load_node_x, load_node_y=case.load_node_y,
                     load_angle=case.load_angle, volume_fraction=case.volume_fraction,
                     max_iterations=options.iterations_per_case) for case in sampling.cases()]
    keys = [_case_key(case) for case in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("Validation cases must be distinct after load-node and angle normalisation")
    training_keys = {_case_key(CantileverConfig(**case)) for case in metadata["cases"]}
    if any(key in training_keys for key in keys):
        raise ValueError("Validation must use unseen cases: a load-node, direction, and volume combination matches training")
    free = 2 * base.nx * (base.ny + 1)
    if 8 * (2*free + 4*base.nx*base.ny) * options.maximum_states > 128*1024**2:
        raise ValueError("Validation arrays exceed 128 MiB; reduce comparison states")
    return cases


def _relative_error(approximate, reference):
    numerator = float(np.linalg.norm(np.asarray(approximate) - reference))
    denominator = float(np.linalg.norm(reference))
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    ratio = numerator / denominator
    return ratio if np.isfinite(ratio) else None


def compare_evaluations(full, reduced, options: ValidationConfig):
    errors = {"displacement_error": _relative_error(reduced.displacement, full.displacement),
              "compliance_error": _relative_error(reduced.compliance, full.compliance),
              "gradient_error": _relative_error(reduced.objective_gradient, full.objective_gradient)}
    passed = all(errors[f"{name}_error"] is not None and errors[f"{name}_error"] <= getattr(options, f"{name}_tolerance")
                 for name in ("displacement", "compliance", "gradient"))
    return {**errors, "passed": passed, "full_compliance": full.compliance,
            "rom_compliance": reduced.compliance, "full_residual": full.residual, "rom_residual": reduced.residual}


def validate_basis(directory: Path, options: ValidationConfig, *, validation_id=None,
                   emit=lambda event: None, stop_requested=lambda: False):
    directory = Path(directory)
    metadata = read_metadata(directory)
    cases = validation_cases(metadata, options)
    if stop_requested():
        return None
    basis_path = directory / "basis.npz"
    basis_hash = hashlib.sha256(basis_path.read_bytes()).hexdigest()
    with np.load(basis_path, allow_pickle=False) as archive:
        basis = archive["basis"]
        free_dofs = archive["free_dofs"]
    mesh = StructuredMesh(cases[0].nx, cases[0].ny, cases[0].length, cases[0].height)
    if not np.array_equal(free_dofs, mesh.free_dofs) or basis.shape[1] != metadata["basis"]["rank"]:
        raise ValueError("The saved basis does not match its mesh or recorded rank")
    validation_id = validation_id or uuid.uuid4().hex
    if len(validation_id) != 32 or any(char not in "0123456789abcdef" for char in validation_id):
        raise ValueError("Invalid validation identifier")
    report = {"id": validation_id, "training_id": directory.name, "basis_sha256": basis_hash,
              "basis_rank": basis.shape[1], "created_at": _now(), "options": json.loads(json.dumps(asdict(options))),
              "case_count": len(cases), "completed_cases": 0, "maximum_states": options.maximum_states,
              "cases": [asdict(case) for case in cases], "results": [],
              "method": "Frozen Galerkin basis on full-order MMA trajectories; no enrichment or fitting",
              "gradient": "Compliance gradient with respect to raw design variables, including SIMP, Helmholtz and projection",
              "error_definition": "Relative Euclidean L2 error; scalar compliance relative absolute error; reference is full order",
              "scope": "Evidence for these held-out cases and sampled designs only; not a guarantee for arbitrary layouts"}
    arrays = {key: [] for key in ("designs", "densities", "full_displacements", "rom_displacements", "full_gradients", "rom_gradients")}
    for index, case in enumerate(cases):
        if stop_requested():
            break
        emit({"event": "validation_case_started", "case_index": index, "case_count": len(cases), "config": asdict(case)})
        last_saved = -1
        with CantileverProblem(case, state_solver=GalerkinSolver(case, basis)) as rom_problem:
            def compare(row, design, full, *, force=False):
                nonlocal last_saved
                if stop_requested() or row.iteration == last_saved or (not force and row.iteration % options.comparison_stride):
                    return
                reduced = rom_problem.evaluate(design)
                result = {"case_index": index, "iteration": row.iteration, **compare_evaluations(full, reduced, options)}
                last_saved = row.iteration
                report["results"].append(result)
                for key, values in (("designs", design), ("densities", full.density),
                    ("full_displacements", full.displacement[free_dofs]), ("rom_displacements", reduced.displacement[free_dofs]),
                    ("full_gradients", full.objective_gradient), ("rom_gradients", reduced.objective_gradient)):
                    arrays[key].append(values.copy())
                emit({"event": "validation_state", "result": result, "state_count": len(report["results"]),
                      "maximum_states": options.maximum_states, "density": full.density.tolist()})

            full_result = optimise(case, evaluation_callback=compare, stop_requested=stop_requested)
            compare(full_result.history[-1], full_result.design, full_result.evaluation, force=True)
            if full_result.stop_reason != "cancelled":
                report["completed_cases"] += 1
    if not report["results"]:
        return None
    report["status"] = "cancelled" if stop_requested() else "complete"
    report["state_count"] = len(report["results"])
    report["passed_states"] = sum(row["passed"] for row in report["results"])
    report["meets_tolerances"] = report["status"] == "complete" and report["passed_states"] == report["state_count"]
    report["metrics"] = {}
    for name in ("displacement", "compliance", "gradient"):
        values = [row[f"{name}_error"] for row in report["results"] if row[f"{name}_error"] is not None]
        report["metrics"][name] = {"max": max(values) if values else None,
                                   "mean": float(np.mean(values)) if values else None, "defined_states": len(values)}
    # Unique report paths and publishing metadata last preserve previous results
    # if a disconnected client stops during the final file writes.
    destination = directory / "validations" / validation_id
    destination.mkdir(parents=True, exist_ok=False)
    saved = _save_arrays(destination / "comparisons.npz", should_commit=lambda: not stop_requested() or report["status"] == "cancelled",
               **{key: np.column_stack(values) for key, values in arrays.items()}, free_dofs=free_dofs,
               case_indices=np.array([row["case_index"] for row in report["results"]]),
               iterations=np.array([row["iteration"] for row in report["results"]]),
               metadata_json=np.array(json.dumps(report)))
    if not saved:
        return None
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    metadata["validation"] = {key: value for key, value in report.items() if key not in ("results", "cases")}
    metadata["updated_at"] = _now()
    save_metadata(directory, metadata)
    return metadata
