"""Reproduce small connection-constraint experiments using the production solver.

Run from the repository with PYTHONPATH=src. Experimental row transformations
are confined to this script; history and stopping retain the physical constraint.
"""

import argparse
from copy import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ml_opt.config import CantileverConfig
from ml_opt.optimizers._mmapy import warmup_optimizer
import ml_opt.runner as runner


BaseOptimizer = runner.MovingAsymptotes


class ScaledConnections(BaseOptimizer):
    transformation = "log"

    def step(self, design, current, evaluate):
        if not hasattr(self, "row_scales"):
            self.row_scales = {key: max(1., value + 1.) for key, value in
                               (current.connection_constraints or {}).items()}

        def scaled(evaluation):
            result = copy(evaluation)
            values, derivatives = {}, {}
            for key, value in (evaluation.connection_constraints or {}).items():
                ratio = 1. + value
                if self.transformation == "log":
                    values[key], derivatives[key] = np.log(ratio), 1. / ratio
                elif self.transformation == "root":
                    values[key], derivatives[key] = ratio**(1./3.) - 1., ratio**(-2./3.) / 3.
                else:
                    values[key], derivatives[key] = value / self.row_scales[key], 1. / self.row_scales[key]
            result.connection_constraints = values
            if evaluation.connection_gradients is not None:
                result.connection_gradients = {key: gradient * derivatives[key]
                    for key, gradient in evaluation.connection_gradients.items()}
            return result

        return super().step(design, scaled(current), lambda x, sensitivities: scaled(evaluate(x, sensitivities)))


def cases(max_iterations):
    cantilever = CantileverConfig(nx=30, ny=10, max_iterations=max_iterations,
        filter_radius=.1, projection_beta=2.,
        connection_anchors=[{"kind": "point", "start": [0, 0]}],
        connection_targets=[{"kind": "point", "start": [1, .5]}])
    mbb = replace(cantilever, robust=True, optimizer="gcmma", projection_schedule="geometric",
        connection_anchors=[{"kind": "point", "start": [1, 0]}],
        connection_targets=[{"kind": "point", "start": [0, 1]}],
        boundary_conditions=[
            {"kind": "restrain_x", "selection": {"kind": "line", "start": [0, 0], "end": [0, 1]}},
            {"kind": "restrain_y", "selection": {"kind": "point", "start": [1, 0]}},
            {"kind": "force", "case_id": "case-1", "selection": {"kind": "point", "start": [0, 1]}}],
        load_cases=[{"id": "case-1", "name": "Load case 1", "weight": 1.}])
    return {"cantilever": cantilever,
            "off-load target": replace(cantilever, connection_targets=[{"kind": "point", "start": [1, 1]}]),
            "MBB robust": mbb}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/connection-benchmark"))
    parser.add_argument("--max-iterations", type=int, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configurations = cases(args.max_iterations)
    report = {"cases": {name: asdict(config) for name, config in configurations.items()}, "runs": []}
    densities = {}
    warmup_optimizer()
    try:
        for name, config in configurations.items():
            for variant in ("none", "linear", "log", "root", "initial", "move .05", "limit 10", "limit 20"):
                runner.MovingAsymptotes = ScaledConnections if variant in ("log", "root", "initial") else BaseOptimizer
                ScaledConnections.transformation = variant
                setting = config
                if variant == "none":
                    setting = replace(config, connection_anchors=(), connection_targets=())
                elif variant == "move .05":
                    setting = replace(config, move_limit=.05)
                elif variant.startswith("limit"):
                    setting = replace(config, connection_max_temperature=float(variant.split()[1]))
                started = perf_counter()
                result = runner.optimise(setting)
                evaluation = result.evaluation
                row = {"case": name, "variant": variant, "iterations": result.history[-1].iteration,
                    "converged": result.converged, "objective": evaluation.compliance,
                    "volume": evaluation.volume_fraction, "constraint": evaluation.connection_constraints,
                    "ratios": evaluation.connection_ratios, "connected": evaluation.connection_connected,
                    "inner_iterations": sum(h.inner_iterations for h in result.history),
                    "inner_limit_hits": sum(h.gcmma_inner_limit_reached for h in result.history),
                    "seconds": perf_counter() - started}
                report["runs"].append(row)
                densities[f"{name}/{variant}"] = evaluation.density
                print(json.dumps(row), flush=True)
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(args.output / "densities.npz", **densities)
    finally:
        runner.MovingAsymptotes = BaseOptimizer


if __name__ == "__main__":
    main()
