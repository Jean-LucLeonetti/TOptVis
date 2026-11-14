"""Compare full robust evaluations, separating cold startup from warm timings.

python benchmarks/robust_fields.py --nx 60 --ny 20 --repeats 5
Use the checkout's src on PYTHONPATH when sharing a worktree environment.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import time

import numpy as np

from ml_opt.config import CantileverConfig
from ml_opt.problem import CantileverProblem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=60)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    config = CantileverConfig(nx=args.nx, ny=args.ny, robust=True,
                              filter_radius=.1, projection_beta=4)
    designs = [np.random.default_rng(index).uniform(.3, .7, args.nx * args.ny)
               for index in range(args.repeats + 1)]
    measurements, references = {}, []
    for mode in ("sequential", "parallel"):
        started = time.perf_counter()
        with CantileverProblem(replace(config, field_execution=mode)) as problem:
            problem.evaluate(designs[0])
            cold = time.perf_counter() - started
            samples = []
            for index, design in enumerate(designs[1:]):
                started = time.perf_counter()
                evaluation = problem.evaluate(design)
                samples.append(time.perf_counter() - started)
                if mode == "sequential":
                    references.append(evaluation)
                else:
                    reference = references[index]
                    for attribute in ("compliance", "volume_fraction", "objective_gradient", "constraint_gradient"):
                        np.testing.assert_allclose(getattr(evaluation, attribute),
                                                   getattr(reference, attribute), rtol=1e-9, atol=1e-10)
                    for name in ("eroded", "nominal", "dilated"):
                        np.testing.assert_allclose(evaluation.field_gradients[name], reference.field_gradients[name],
                                                   rtol=1e-9, atol=1e-10)
            measurements[mode] = {"cold_setup_and_first_evaluation_seconds": cold,
                "warm_evaluation_seconds": samples, "warm_median_seconds": statistics.median(samples),
                "state_solves": problem.solve_count}
    report = {"config": asdict(config), "evaluations_include_gradients": True,
              "measurements": measurements, "warm_speedup":
              measurements["sequential"]["warm_median_seconds"] / measurements["parallel"]["warm_median_seconds"]}
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
