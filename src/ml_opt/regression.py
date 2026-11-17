"""Small, inspectable compliance surrogates with whole-case held-out testing."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import CantileverConfig


@dataclass(frozen=True)
class RegressionSettings:
    nx: int = 30
    ny: int = 10
    length: float = 3.0
    height: float = 1.0
    load_height: float = 0.5
    load_angle: float = -90.0
    volume_min: float = 0.25
    volume_max: float = 0.65
    case_count: int = 20
    max_iterations: int = 200
    filter_radius: float = 0.1
    projection_beta: float = 2.0
    projection_eta: float = 0.5
    material_model: str = "simp"
    penalisation: float = 3.0
    problem: dict | None = None
    source_problem: dict | None = None
    problem_name: str | None = None

    def __post_init__(self):
        for name in ("nx", "ny", "case_count", "max_iterations"):
            value = getattr(self, name)
            if type(value) is not int or value < (6 if name == "case_count" else 1):
                raise ValueError(f"{name} must be an integer of at least {6 if name == 'case_count' else 1}")
        if not all(math.isfinite(value) for value in asdict(self).values() if isinstance(value, (int, float))):
            raise ValueError("Settings must be finite")
        if not 0 < self.volume_min < self.volume_max < 1:
            raise ValueError("Volume range must satisfy 0 < minimum < maximum < 1")
        if not 0 <= self.load_height <= 1 or not -180 <= self.load_angle <= 180:
            raise ValueError("Choose a load height in [0, 1] and angle in [-180, 180]")
        self.configuration(self.volume_min)
        self.configuration(self.volume_max)

    def configuration(self, volume):
        if self.problem is not None:
            # Imported setups retain every numerical setting except the sampled volume.
            return replace(CantileverConfig(**self.problem), volume_fraction=volume)
        return CantileverConfig(nx=self.nx, ny=self.ny, length=self.length, height=self.height,
            load_node_x=self.nx, load_node_y=int(math.floor(self.load_height*self.ny + 0.5)),
            load_angle=self.load_angle, volume_fraction=volume, max_iterations=self.max_iterations,
            filter_radius=self.filter_radius, projection_beta=self.projection_beta,
            projection_eta=self.projection_eta, material_model=self.material_model,
            penalisation=self.penalisation)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_dataset(directory):
    data = json.loads((Path(directory) / "dataset.json").read_text(encoding="utf-8"))
    if data.get("format") != "toptvis-regression-v1":
        raise ValueError("Unsupported regression dataset")
    return data


def dataset_summary(data):
    return {key: value for key, value in data.items() if key != "cases"} | {
        "completed_cases": len(data["cases"]),
        "eligible_cases": sum(case.get("converged", False) and case.get("feasible", False)
                              for case in data["cases"]),
        "feasible_cases": sum(case.get("feasible", False) and case.get("stop_reason") != "cancelled"
                              for case in data["cases"]),
    }


def generate_dataset(settings, directory, *, emit=lambda event: None, stop_requested=lambda: False):
    # Reuse the project's optimiser; this module adds no state-solver implementation.
    from .runner import optimise

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    data = {"format": "toptvis-regression-v1", "id": directory.name,
            "created_at": _now(), "status": "generating", "settings": asdict(settings),
            "fixed_config": json.loads(json.dumps(asdict(settings.configuration(settings.volume_min)))),
            "objective_quantity": "elastic_work" if settings.configuration(settings.volume_min).imposed_motion else "compliance",
            "cases": [], "generation_seconds": 0.0}

    def checkpoint():
        data["generation_seconds"] = perf_counter() - started
        _save_json(directory / "dataset.json", data)

    checkpoint()
    for index, volume in enumerate(np.linspace(settings.volume_min, settings.volume_max, settings.case_count)):
        if stop_requested():
            break
        config = settings.configuration(float(volume))
        emit({"event": "regression_case_started", "case_index": index,
              "case_count": settings.case_count, "config": asdict(config)})
        case_started = perf_counter()

        def snapshot(entry, density):
            if entry.iteration % 5 == 0:
                emit({"event": "regression_iteration", "case_index": index,
                      "iteration": entry.iteration, "compliance": entry.compliance,
                      "density": density.tolist()})

        try:
            result = optimise(config, snapshot_callback=snapshot, stop_requested=stop_requested)
            evaluation = result.evaluation
            density = np.asarray(evaluation.density, dtype=float)
            if (not math.isfinite(evaluation.compliance) or evaluation.compliance <= 0 or
                    not math.isfinite(evaluation.constraint) or not np.all(np.isfinite(density))):
                raise ValueError("The solver returned a non-finite or nonpositive result")
            constraints = {"volume": float(evaluation.constraint), **(evaluation.connection_constraints or {})}
            for name in ("epigraph_constraint", "max_size_constraint", "max_void_constraint"):
                value = getattr(evaluation, name)
                if value is not None:
                    constraints[name] = float(value)
            case = {"index": index, "volume_fraction": float(volume),
                    "compliance": float(evaluation.compliance), "density": density.tolist(),
                    "physical_volume": float(evaluation.volume_fraction),
                    "converged": result.converged,
                    "feasible": all(math.isfinite(value) and value <= config.feasibility_tolerance
                                    for value in constraints.values()),
                    "constraints": constraints,
                    "stop_reason": result.stop_reason, "iterations": result.history[-1].iteration,
                    "projection_beta": float(evaluation.projection_beta),
                    "field_compliances": evaluation.field_compliances,
                    "field_volumes": evaluation.field_volumes,
                    "load_case_compliances": evaluation.load_case_compliances,
                    "generation_seconds": perf_counter() - case_started}
        except Exception as error:
            case = {"index": index, "volume_fraction": float(volume), "converged": False,
                    "feasible": False, "stop_reason": "failed", "message": str(error),
                    "generation_seconds": perf_counter() - case_started}
        data["cases"].append(case)
        checkpoint()
        emit({"event": "regression_case", "case": case, "completed_cases": len(data["cases"]),
              "case_count": settings.case_count})
        if case["stop_reason"] == "cancelled":
            break
    data["status"] = "cancelled" if stop_requested() or len(data["cases"]) < settings.case_count else "complete"
    checkpoint()
    return data


def split_cases(data, *, test_fraction=0.25, seed=42, include_unconverged=False):
    if not math.isfinite(test_fraction) or not 0.1 <= test_fraction <= 0.5:
        raise ValueError("Test fraction must be between 0.1 and 0.5")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be a nonnegative 32-bit integer")
    if type(include_unconverged) is not bool:
        raise ValueError("include_unconverged must be a boolean")
    eligible = [case for case in data["cases"] if case.get("feasible") and
                case.get("stop_reason") not in ("failed", "cancelled") and
                (case.get("converged") or include_unconverged)]
    eligible.sort(key=lambda case: case["volume_fraction"])
    if len(eligible) < 6:
        raise ValueError("At least six eligible cases are needed. Generate more converged cases, or explicitly include feasible unfinished runs.")
    volumes = np.array([case["volume_fraction"] for case in eligible])
    targets = np.array([case["compliance"] for case in eligible])
    if (not np.all(np.isfinite(volumes)) or not np.all(np.isfinite(targets)) or
            np.any(targets <= 0) or len(np.unique(volumes)) != len(volumes)):
        raise ValueError("Cases require distinct finite volumes and positive finite compliance")
    # Keep endpoints in training. Test cases measure interpolation, not extrapolation.
    count = max(1, min(math.ceil(len(eligible)*test_fraction), len(eligible)-3))
    test = np.sort(np.random.default_rng(seed).choice(np.arange(1, len(eligible)-1), count, replace=False))
    train = np.setdiff1d(np.arange(len(eligible)), test)
    return eligible, train, test


def predict(model, volumes):
    values = np.asarray(volumes, dtype=float)
    if "scaled_coefficients" in model:
        values = (values - model["input_offset"]) / model["input_scale"]
        return np.polynomial.polynomial.polyval(values, model["scaled_coefficients"])
    return np.polynomial.polynomial.polyval(values, model["coefficients"])


def validate_regression(method, ridge_alpha):
    if method not in ("ols", "ridge"):
        raise ValueError("Regression method must be ols or ridge")
    if (isinstance(ridge_alpha, bool) or not isinstance(ridge_alpha, (int, float)) or
            not math.isfinite(ridge_alpha) or ridge_alpha < 0):
        raise ValueError("Ridge penalty must be finite and nonnegative")
    return float(ridge_alpha) if method == "ridge" else 0.0


def validate_degrees(degrees, training_count=None, *, method="ols", ridge_alpha=.01):
    alpha = validate_regression(method, ridge_alpha)
    if (not isinstance(degrees, (list, tuple)) or not degrees or
            any(type(degree) is not int or degree < 1 for degree in degrees) or
            len(set(degrees)) != len(degrees)):
        raise ValueError("Polynomial orders must be distinct positive integers")
    required = max(degrees) + 1
    if training_count is not None and alpha == 0 and training_count < required:
        raise ValueError(f"Order {max(degrees)} needs at least {required} training cases; "
                         f"this split has {training_count}. Generate more eligible cases, "
                         "reduce the test share, lower the polynomial order, or use ridge with a positive penalty.")
    return sorted(degrees)


def fit_polynomial(inputs, targets, degree, *, method="ols", ridge_alpha=.01):
    """Fit scaled monomials; ridge leaves the constant coefficient unpenalised."""
    alpha = validate_regression(method, ridge_alpha)
    matrix = np.polynomial.polynomial.polyvander(inputs, degree)
    if alpha == 0:
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, targets, rcond=None)
        if rank < degree + 1:
            raise ValueError("Training volumes do not identify the requested polynomial at numerical precision. "
                             "Lower the order or use ridge with a positive penalty.")
    else:
        # Centring removes the intercept from the penalty. Economy SVD also
        # supports more coefficients than training cases without a p-by-p solve.
        features = matrix[:, 1:]
        feature_mean, target_mean = features.mean(axis=0), targets.mean(axis=0)
        u, singular, vh = np.linalg.svd(features - feature_mean, full_matrices=False)
        weights = singular / (singular**2 + alpha)
        projected = u.T @ (targets - target_mean)
        slopes = vh.T @ (weights * projected if targets.ndim == 1 else weights[:, None] * projected)
        intercept = target_mean - feature_mean @ slopes
        coefficients = np.concatenate((np.expand_dims(intercept, 0), slopes), axis=0)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Polynomial fitting exceeded numerical precision; lower the order or increase the ridge penalty")
    return coefficients


def volume_coefficients(scaled_coefficients, offset, scale):
    """Original-volume coefficients are optional when their expansion overflows."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = np.polynomial.Polynomial(scaled_coefficients)(
                np.polynomial.Polynomial([-offset / scale, 1 / scale])).coef
        return values.tolist() if np.all(np.isfinite(values)) else None
    except FloatingPointError:
        return None


def _errors(actual, predicted):
    difference = predicted - actual
    return {"mae": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "mean_relative_error": float(np.mean(np.abs(difference) / actual))}


def fit_models(directory, *, test_fraction=0.25, seed=42, include_unconverged=False, degrees=(1, 2),
               method="ols", ridge_alpha=.01,
               stop_requested=lambda: False, allow_interrupted=False):
    directory = Path(directory)
    data = read_dataset(directory)
    if data["status"] == "generating" and not allow_interrupted:
        raise ValueError("Wait for dataset generation to finish before training")
    cases, train, test = split_cases(data, test_fraction=test_fraction, seed=seed,
                                   include_unconverged=include_unconverged)
    degrees = validate_degrees(degrees, len(train), method=method, ridge_alpha=ridge_alpha)
    volumes = np.array([case["volume_fraction"] for case in cases])
    targets = np.array([case["compliance"] for case in cases])
    baseline = float(np.mean(targets[train]))
    report = {"format": "toptvis-regression-models-v1", "dataset_id": data["id"],
              "dataset_sha256": hashlib.sha256((directory / "dataset.json").read_bytes()).hexdigest(),
              "trained_at": _now(), "test_fraction": test_fraction, "seed": seed,
              "include_unconverged": include_unconverged,
              "degrees": degrees,
              "method": method, "ridge_alpha": validate_regression(method, ridge_alpha),
              "train_indices": [cases[i]["index"] for i in train],
              "test_indices": [cases[i]["index"] for i in test],
              "excluded_indices": [case["index"] for case in data["cases"] if case not in cases],
              "unfinished_indices": [case["index"] for case in cases if not case.get("converged")],
              "training_range": [float(volumes[train].min()), float(volumes[train].max())],
              "baseline": {"value": baseline, "test": _errors(targets[test], np.full(len(test), baseline))},
              "models": {}}
    offset = float((volumes[train].min() + volumes[train].max()) / 2)
    scale = float((volumes[train].max() - volumes[train].min()) / 2)
    for degree in degrees:
        if stop_requested():
            return None
        name = {1: "linear", 2: "quadratic"}.get(degree, f"polynomial_{degree}")
        started = perf_counter()
        scaled_coefficients = fit_polynomial((volumes[train] - offset) / scale, targets[train], degree,
                                             method=method, ridge_alpha=ridge_alpha)
        # Retain original-volume coefficients for inspection and older consumers;
        # predictions use the scaled basis to avoid cancellation at higher orders.
        coefficients = volume_coefficients(scaled_coefficients, offset, scale)
        model = {"degree": degree, "coefficients": coefficients,
                 "method": method, "ridge_alpha": report["ridge_alpha"],
                 "scaled_coefficients": scaled_coefficients.tolist(),
                 "input_offset": offset, "input_scale": scale,
                 "training_seconds": perf_counter() - started}
        model["train"] = _errors(targets[train], predict(model, volumes[train]))
        model["test"] = _errors(targets[test], predict(model, volumes[test]))
        grid = np.linspace(*report["training_range"], 201)
        started = perf_counter()
        values = predict(model, grid)
        model["prediction_seconds"] = perf_counter() - started
        model["prediction_count"] = len(grid)
        model["curve"] = {"volumes": grid.tolist(), "compliances": values.tolist()}
        report["models"][name] = model
    # Cancelled fits leave the last saved model intact.
    if stop_requested():
        return None
    _save_json(directory / "models.json", report)
    return report


def read_models(directory):
    directory = Path(directory)
    path = directory / "models.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256((directory / "dataset.json").read_bytes()).hexdigest()
    if report.get("format") != "toptvis-regression-models-v1" or report.get("dataset_sha256") != fingerprint:
        raise ValueError("Saved models do not match this dataset; train again")
    return report
