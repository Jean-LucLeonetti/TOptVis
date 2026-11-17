"""Polynomial prediction of physical density fields on a fixed mesh."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import CantileverConfig
from .regression import (_now, _save_json, read_dataset, split_cases, validate_degrees,
                         validate_regression, fit_polynomial)
from .regions import region_densities


def topology_cases(data, **options):
    cases, train, test = split_cases(data, **options)
    config = CantileverConfig(**data["fixed_config"])
    try:
        densities = np.asarray([case["density"] for case in cases], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ValueError("Every eligible case needs a physical density field on the dataset mesh") from None
    if (densities.shape != (len(cases), config.nx * config.ny) or
            not np.all(np.isfinite(densities)) or np.any(densities < 0) or np.any(densities > 1)):
        raise ValueError("Every eligible case needs finite densities in [0, 1] on the dataset mesh")
    fixed = region_densities(config.non_design_regions, config.nx, config.ny)
    mask = np.isfinite(fixed)
    if not np.allclose(densities[:, mask], fixed[mask], atol=1e-12, rtol=0):
        raise ValueError("Reference densities do not preserve the fixed solid and void regions")
    return cases, train, test, densities, fixed


def predict_topology(model, volumes, *, bounded=True):
    """Return (cases, elements), or (elements,) for a scalar volume."""
    volumes = np.asarray(volumes, dtype=float)
    if not np.all(np.isfinite(volumes)):
        raise ValueError("Prediction volumes must be finite")
    inputs = (np.atleast_1d(volumes) - model["input_offset"]) / model["input_scale"]
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.polynomial.polynomial.polyvander(inputs, model["degree"]) @ np.asarray(model["coefficients"])
    if not np.all(np.isfinite(values)):
        raise ValueError("Prediction exceeds numerical range; choose a volume closer to training or lower the polynomial order")
    if bounded:
        values = np.clip(values, 0, 1)
    return values[0] if volumes.ndim == 0 else values


def _density_errors(actual, predicted):
    difference = predicted - actual
    return {"mae": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "max_case_mae": float(np.max(np.mean(np.abs(difference), axis=1)))}


def fit_topology_models(directory, *, test_fraction=.25, seed=42, include_unconverged=False,
                        degrees=(1, 2), method="ols", ridge_alpha=.01,
                        stop_requested=lambda: False, allow_interrupted=False):
    directory = Path(directory)
    data = read_dataset(directory)
    if data["status"] == "generating" and not allow_interrupted:
        raise ValueError("Wait for dataset generation to finish before training")
    cases, train, test, densities, fixed = topology_cases(data, test_fraction=test_fraction,
        seed=seed, include_unconverged=include_unconverged)
    degrees = validate_degrees(degrees, len(train), method=method, ridge_alpha=ridge_alpha)
    volumes = np.array([case["volume_fraction"] for case in cases])
    low, high = float(volumes[train].min()), float(volumes[train].max())
    offset, scale = (low + high) / 2, (high - low) / 2
    mean = densities[train].mean(axis=0)
    report = {"format": "toptvis-topology-regression-v1", "target": "physical_density",
        "dataset_id": data["id"],
        "dataset_sha256": hashlib.sha256((directory / "dataset.json").read_bytes()).hexdigest(),
        "trained_at": _now(), "test_fraction": test_fraction, "seed": seed,
        "include_unconverged": include_unconverged, "degrees": degrees,
        "method": method, "ridge_alpha": validate_regression(method, ridge_alpha),
        "train_indices": [cases[i]["index"] for i in train],
        "test_indices": [cases[i]["index"] for i in test],
        "excluded_indices": [case["index"] for case in data["cases"] if case not in cases],
        "unfinished_indices": [case["index"] for case in cases if not case.get("converged")],
        "training_range": [low, high], "element_count": densities.shape[1],
        "postprocessing": "clip_to_unit_interval", "models": {},
        "baseline": {"train": _density_errors(densities[train], np.broadcast_to(mean, densities[train].shape)),
                     "test": _density_errors(densities[test], np.broadcast_to(mean, densities[test].shape))}}
    fixed_mask = np.isfinite(fixed)
    for degree in degrees:
        if stop_requested():
            return None
        started = perf_counter()
        coefficients = fit_polynomial((volumes[train] - offset) / scale, densities[train], degree,
                                       method=method, ridge_alpha=ridge_alpha)
        coefficients[:, fixed_mask] = 0
        coefficients[0, fixed_mask] = fixed[fixed_mask]
        fitted = {"degree": degree, "input_offset": offset, "input_scale": scale,
            "method": method, "ridge_alpha": report["ridge_alpha"],
            "coefficients": coefficients.tolist(), "training_seconds": perf_counter() - started}
        for name, indices in (("train", train), ("test", test)):
            raw = predict_topology(fitted, volumes[indices], bounded=False)
            predicted = np.clip(raw, 0, 1)
            fitted[name] = _density_errors(densities[indices], predicted)
            fitted[name]["raw_out_of_bounds_fraction"] = float(np.mean((raw < -1e-10) | (raw > 1+1e-10)))
        fitted["cases"] = []
        for i, case in enumerate(cases):
            predicted = predict_topology(fitted, volumes[i])
            fitted["cases"].append({"index": case["index"], "volume_fraction": volumes[i],
                "use": "test" if i in test else "training",
                "mae": float(np.mean(np.abs(predicted - densities[i]))),
                "predicted_volume": float(predicted.mean()), "reference_volume": float(densities[i].mean())})
        report["models"][str(degree)] = fitted
    if stop_requested():
        return None
    _save_json(directory / "topology-models.json", report)
    return report


def read_topology_models(directory):
    directory = Path(directory)
    path = directory / "topology-models.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256((directory / "dataset.json").read_bytes()).hexdigest()
    if report.get("format") != "toptvis-topology-regression-v1" or report.get("dataset_sha256") != fingerprint:
        raise ValueError("Saved topology models do not match this dataset; train again")
    return report


def evaluate_topology_prediction(directory, *, degree=None, model_key=None, volume, stop_requested=lambda: False):
    from .problem import CantileverProblem
    from .neural_regression import read_neural_models, predict_network

    directory = Path(directory)
    data = read_dataset(directory)
    report = read_neural_models(directory, "topology") if model_key else read_topology_models(directory)
    key = model_key or str(degree)
    if report is None or key not in report["models"]:
        raise ValueError("Train the selected topology model first")
    if not np.isfinite(volume) or not 0 < volume < 1:
        raise ValueError("Prediction volume must lie between 0 and 1")
    raw = predict_network(report["models"][key], volume) if model_key else predict_topology(report["models"][key], volume, bounded=False)
    density = np.clip(raw, 0, 1)
    # These targets are already physical, nominal fields. Do not filter/project
    # twice or reinterpret a nominal prediction as a robust three-field design.
    config = replace(CantileverConfig(**data["fixed_config"]), volume_fraction=volume,
        filter_radius=0., projection_beta=0., projection_schedule="fixed", robust=False,
        state_solver="fom", rom_basis_path=None, rom_basis_sha256=None, field_execution="sequential")
    reference = next((case for case in data["cases"] if case.get("density") is not None and
                      np.isclose(case["volume_fraction"], volume, atol=1e-12, rtol=0)), None)
    result = {"dataset_id": data["id"], "degree": degree, "model_key": key, "volume_fraction": volume,
              "objective_quantity": "elastic_work" if config.imposed_motion else "compliance",
              "field": "nominal", "robust_dataset": bool(data["fixed_config"].get("robust")),
              "clipped_elements": int(np.count_nonzero((raw < -1e-10) | (raw > 1+1e-10))),
              "reference_index": reference["index"] if reference else None}
    def summary(evaluation):
        constraints = {"volume": evaluation.constraint, **(evaluation.connection_constraints or {})}
        for name in ("max_size_constraint", "max_void_constraint"):
            value = getattr(evaluation, name)
            if value is not None:
                constraints[name] = value
        return {"objective": evaluation.compliance, "physical_volume": evaluation.volume_fraction,
                "residual": evaluation.residual, "constraints": constraints,
                "nominal_feasible": all(value <= config.feasibility_tolerance for value in constraints.values())}
    if stop_requested():
        return None
    started = perf_counter()
    with CantileverProblem(config) as problem:
        result["prediction"] = summary(problem.evaluate(density, sensitivities=False))
        if stop_requested():
            return None
        if reference is not None:
            result["reference"] = summary(problem.evaluate(np.asarray(reference["density"]), sensitivities=False))
            denominator = abs(result["reference"]["objective"])
            result["relative_objective_change"] = ((result["prediction"]["objective"] -
                result["reference"]["objective"]) / denominator if denominator > 0 else None)
    result["evaluation_seconds"] = perf_counter() - started
    return None if stop_requested() else result
