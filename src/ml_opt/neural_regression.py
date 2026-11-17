"""Small deterministic PyTorch regressors with independent validation and test cases."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .regression import _now, _save_json, _errors, read_dataset, split_cases
from .topology_regression import _density_errors, topology_cases


@dataclass(frozen=True)
class NeuralSettings:
    neurons: int = 8
    epochs: int = 2000
    learning_rate: float = .01
    weight_penalty: float = 0.
    network_seed: int = 42
    validation_fraction: float = .2

    def __post_init__(self):
        for name in ("neurons", "epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.network_seed) is not int or not 0 <= self.network_seed <= 2**32-1:
            raise ValueError("Network seed must be a nonnegative 32-bit integer")
        for name in ("learning_rate", "weight_penalty", "validation_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.weight_penalty < 0:
            raise ValueError("Learning rate must be positive and weight penalty nonnegative")
        if not .1 <= self.validation_fraction <= .5:
            raise ValueError("Validation fraction must be between 0.1 and 0.5")


def neural_split(data, *, test_fraction=.25, seed=42, include_unconverged=False, validation_fraction=.2):
    cases, outer_train, test = split_cases(data, test_fraction=test_fraction, seed=seed,
                                         include_unconverged=include_unconverged)
    if not math.isfinite(validation_fraction) or not .1 <= validation_fraction <= .5:
        raise ValueError("Validation fraction must be between 0.1 and 0.5")
    # Keep both range endpoints in fitting, as in polynomial regression. Test
    # selection is unchanged; validation is drawn only from remaining cases.
    count = min(max(1, math.ceil(len(outer_train)*validation_fraction)), len(outer_train)-2)
    validation = np.sort(np.random.default_rng(seed).choice(outer_train[1:-1], count, replace=False))
    train = np.setdiff1d(outer_train, validation)
    return cases, train, validation, test


def neural_filename(target):
    if target not in ("compliance", "topology"):
        raise ValueError("Neural target must be compliance or topology")
    return f"neural-{target}.json"


def read_neural_models(directory, target):
    directory = Path(directory)
    path = directory / neural_filename(target)
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256((directory / "dataset.json").read_bytes()).hexdigest()
    if (report.get("format") != "toptvis-neural-regression-v1" or report.get("target") != target or
            report.get("dataset_sha256") != fingerprint):
        raise ValueError(f"Saved {target} neural network does not match this dataset; train again")
    return report


def predict_network(model, volumes):
    """Portable JSON weights, evaluated by the same float64 PyTorch operations."""
    values = np.asarray(volumes, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Prediction volumes must be finite")
    inputs = torch.tensor(np.atleast_1d(values).reshape(-1, 1), dtype=torch.float64)
    weights = model["weights"]
    with torch.inference_mode():
        x = (inputs-model["input_offset"])/model["input_scale"]
        hidden = torch.tanh(nn.functional.linear(x, torch.tensor(weights["hidden_weight"], dtype=torch.float64),
                                                torch.tensor(weights["hidden_bias"], dtype=torch.float64)))
        output = nn.functional.linear(hidden, torch.tensor(weights["output_weight"], dtype=torch.float64),
                                      torch.tensor(weights["output_bias"], dtype=torch.float64))
        if model["output_activation"] == "sigmoid":
            output = torch.sigmoid(output)
        output = output.numpy()*model["target_scale"] + model["target_offset"]
    if not np.all(np.isfinite(output)):
        raise ValueError("Network prediction exceeds numerical range")
    if model["target"] == "topology":
        full = np.tile(np.asarray(model["fixed_density"]), (len(output), 1))
        full[:, model["output_indices"]] = output
        return full[0] if values.ndim == 0 else full
    output = output[:, 0]
    return float(output[0]) if values.ndim == 0 else output


def fit_neural_models(directory, *, target, test_fraction=.25, seed=42, include_unconverged=False,
                      neurons=8, epochs=2000, learning_rate=.01, weight_penalty=0., network_seed=42,
                      validation_fraction=.2, emit=lambda event: None, stop_requested=lambda: False,
                      allow_interrupted=False):
    settings = NeuralSettings(neurons, epochs, learning_rate, weight_penalty, network_seed, validation_fraction)
    directory = Path(directory)
    filename = neural_filename(target)
    data = read_dataset(directory)
    if data["status"] == "generating" and not allow_interrupted:
        raise ValueError("Wait for dataset generation to finish before training")
    cases, train, validation, test = neural_split(data, test_fraction=test_fraction, seed=seed,
        include_unconverged=include_unconverged, validation_fraction=validation_fraction)
    fixed = indices = None
    if target == "topology":
        checked_cases, _, _, densities, fixed = topology_cases(data, test_fraction=test_fraction,
            seed=seed, include_unconverged=include_unconverged)
        assert [case["index"] for case in checked_cases] == [case["index"] for case in cases]
        indices = np.flatnonzero(np.isnan(fixed))
        targets = densities[:, indices]
    else:
        targets = np.array([case["compliance"] for case in cases])[:, None]
    volumes = np.array([case["volume_fraction"] for case in cases])
    low, high = float(volumes[train].min()), float(volumes[train].max())
    offset, scale = (low+high)/2, (high-low)/2
    target_offset = float(targets[train].mean()) if target == "compliance" else 0.
    target_scale = float(targets[train].std()) if target == "compliance" else 1.
    if target_scale < 1e-12:
        target_scale = 1.
    x = torch.tensor(((volumes-offset)/scale)[:, None], dtype=torch.float64)
    y = torch.tensor((targets-target_offset)/target_scale, dtype=torch.float64)
    # Avoid touching test targets during fitting or checkpoint selection.
    x_train, y_train, x_val, y_val = x[train], y[train], x[validation], y[validation]
    history, best, best_epoch, best_loss = [], None, 0, float("inf")
    started = perf_counter()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings.network_seed)
            hidden = nn.Linear(1, settings.neurons, dtype=torch.float64)
            output = nn.Linear(settings.neurons, targets.shape[1], dtype=torch.float64)
            network = nn.Sequential(hidden, nn.Tanh(), output, *([nn.Sigmoid()] if target == "topology" else []))
            optimiser = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
            stride = max(1, math.ceil(settings.epochs/200))
            best_row = None
            for epoch in range(settings.epochs+1):
                if stop_requested():
                    return None
                if epoch:
                    network.train(); optimiser.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(network(x_train), y_train)
                    # L2 on weight matrices only, using mean squared error units.
                    penalty = hidden.weight.square().sum() + output.weight.square().sum()
                    (loss + settings.weight_penalty*penalty).backward()
                    optimiser.step()
                network.eval()
                with torch.inference_mode():
                    training_loss = float(nn.functional.mse_loss(network(x_train), y_train))
                    validation_loss = float(nn.functional.mse_loss(network(x_val), y_val))
                if not math.isfinite(training_loss) or not math.isfinite(validation_loss):
                    raise ValueError("Network training diverged; reduce the learning rate")
                row = {"epoch": epoch, "train_loss": training_loss, "validation_loss": validation_loss,
                       "elapsed_seconds": perf_counter()-started}
                if validation_loss < best_loss:
                    best_loss, best_epoch, best_row = validation_loss, epoch, row
                    best = {name: parameter.detach().clone() for name, parameter in network.state_dict().items()}
                if epoch % stride == 0 or epoch == settings.epochs:
                    history.append(row)
                    emit({"event": "neural_training", "target": target, "epochs": settings.epochs, **row})
            if all(row["epoch"] != best_epoch for row in history):
                history.append(best_row); history.sort(key=lambda row: row["epoch"])
            network.load_state_dict(best)
            weights = {"hidden_weight": hidden.weight.detach().tolist(), "hidden_bias": hidden.bias.detach().tolist(),
                       "output_weight": output.weight.detach().tolist(), "output_bias": output.bias.detach().tolist()}
    finally:
        torch.set_num_threads(previous_threads)
    model = {"kind": "neural_network", "target": target, "activation": "tanh",
             "output_activation": "sigmoid" if target == "topology" else "linear",
             "neurons": settings.neurons, "weights": weights,
             "input_offset": offset, "input_scale": scale,
             "target_offset": target_offset, "target_scale": target_scale,
             "best_epoch": best_epoch, "best_validation_loss": best_loss,
             "training_history": history, "training_seconds": perf_counter()-started}
    if target == "topology":
        model.update(output_indices=indices.tolist(), fixed_density=np.nan_to_num(fixed).tolist())
    predicted = predict_network(model, volumes)
    actual = densities if target == "topology" else targets[:, 0]
    error = _density_errors if target == "topology" else _errors
    mean = actual[train].mean(axis=0)
    baseline = {}
    for name, subset in (("train", train), ("validation", validation), ("test", test)):
        model[name] = error(actual[subset], predicted[subset])
        baseline[name] = error(actual[subset], np.broadcast_to(mean, actual[subset].shape))
    if target == "topology":
        model["cases"] = [{"index": case["index"], "volume_fraction": float(volumes[i]),
            "use": "test" if i in test else "validation" if i in validation else "training",
            "mae": float(np.mean(np.abs(predicted[i]-actual[i]))),
            "predicted_volume": float(predicted[i].mean()), "reference_volume": float(actual[i].mean())}
            for i, case in enumerate(cases)]
    else:
        grid = np.linspace(low, high, 201)
        stamp = perf_counter(); values = predict_network(model, grid)
        model.update(prediction_seconds=perf_counter()-stamp, prediction_count=201,
                     curve={"volumes": grid.tolist(), "compliances": values.tolist()})
    report = {"format": "toptvis-neural-regression-v1", "target": target, "method": "neural",
        "dataset_id": data["id"], "dataset_sha256": hashlib.sha256((directory/"dataset.json").read_bytes()).hexdigest(),
        "trained_at": _now(), "test_fraction": test_fraction, "seed": seed,
        "include_unconverged": include_unconverged, **vars(settings),
        "train_indices": [cases[i]["index"] for i in train],
        "validation_indices": [cases[i]["index"] for i in validation],
        "test_indices": [cases[i]["index"] for i in test],
        "excluded_indices": [case["index"] for case in data["cases"] if case not in cases],
        "unfinished_indices": [case["index"] for case in cases if not case.get("converged")],
        "training_range": [low, high], "baseline": baseline, "models": {"neural_network": model}}
    if stop_requested():
        return None
    _save_json(directory/filename, report)
    return report
