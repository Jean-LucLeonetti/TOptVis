"""Serial paired timings of the constrained MMA/GCMMA numerical kernels.

Run separately from tests, profiling and other benchmark jobs. Every variant
uses identical constraints, tolerances and iteration budgets.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from types import FunctionType, ModuleType

import numpy as np
from scipy.linalg.lapack import dgesv

from ml_opt.config import CantileverConfig
import ml_opt.optimizers._mmapy as adapter
import ml_opt.optimizers.moving_asymptotes as methods
from ml_opt.runner import optimise


def dot(left, right, out=None):
    if (out is None and left.ndim == right.ndim == 2 and left.shape[0] == right.shape[1] == 1
            and left.base is right and right.size >= 8192):
        return np.asarray(np.einsum("ij,ij->", right, right)).reshape(1, 1)
    return np.dot(left, right, out=out)


def solve(matrix, rhs):
    _, _, solution, info = dgesv(matrix, rhs)
    if info:
        raise RuntimeError(f"DGESV failed: {info}")
    return solution


def variant(name):
    baseline, _ = adapter._bind_functions(compiled=False)
    namespace = dict(baseline.__globals__)
    # All replacements stay in the cloned functions' private namespace.
    if name in ("norm", "both"):
        array_api = ModuleType("mma_numpy")
        array_api.__dict__.update(vars(np))
        array_api.dot = dot
        namespace["np"] = array_api
    if name in ("lapack", "both"):
        namespace["solve"] = solve
    for key in ("subsolv", "mmasub", "gcmmasub"):
        original = namespace[key]
        namespace[key] = FunctionType(original.__code__, namespace, original.__name__,
                                       original.__defaults__, original.__closure__)
    if name == "compiled":
        from ml_opt.optimizers._subproblem import subsolv
        namespace["subsolv"] = subsolv
    return namespace["mmasub"], namespace["gcmmasub"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", nargs="+", default=["baseline", "compiled"])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/2026-11-15-size-kernels.json"))
    options = parser.parse_args()
    kernels = {name: variant(name) for name in options.variants}
    saved = methods.mmasub, methods.gcmmasub
    report = {"date_utc": datetime.now(timezone.utc).isoformat(),
              "versions": {name: version(name) for name in ("numpy", "scipy", "numba", "mmapy")},
              "iterations": options.iterations, "repeats": options.repeats,
              "records": [], "summary": []}
    try:
        compile_started = perf_counter()
        adapter.warmup_optimizer()
        report["compile_or_cache_load_seconds"] = perf_counter()-compile_started
        for nx, ny in ((60, 20), (120, 40)):
            for method in ("mma", "gcmma"):
                config = CantileverConfig(nx=nx, ny=ny, optimizer=method, projection_beta=8,
                    filter_radius=.1, max_solid_size=.45, max_void_size=.8,
                    max_iterations=options.iterations, min_iterations=options.iterations+1)
                expected = None
                timings = {name: [] for name in kernels}
                for repeat in range(options.repeats):
                    order = list(kernels) if repeat % 2 == 0 else list(reversed(kernels))
                    for name in order:
                        methods.mmasub, methods.gcmmasub = kernels[name]
                        start = perf_counter()
                        result = optimise(config)
                        elapsed = perf_counter()-start
                        if expected is None:
                            expected = result
                        error = float(np.max(np.abs(expected.design-result.design)))
                        if error > 1e-7:
                            raise RuntimeError(f"{name} changed the design by {error}")
                        record = {"config": asdict(config), "repeat": repeat, "variant": name,
                            "seconds": elapsed, "update_seconds": result.history[-1].elapsed_seconds-result.history[0].elapsed_seconds,
                            "design_max_error": error, "compliance": result.evaluation.compliance,
                            "volume_constraint": result.evaluation.constraint,
                            "solid_constraint": result.evaluation.max_size_constraint,
                            "void_constraint": result.evaluation.max_void_constraint,
                            "state_solves": result.history[-1].state_solves,
                            "inner_retries": sum(row.inner_iterations for row in result.history)}
                        report["records"].append(record)
                        timings[name].append(elapsed)
                        print(f"{nx}x{ny} {method} {name} #{repeat+1}: {elapsed:.4f}s; design error {error:.3g}", flush=True)
                        options.output.parent.mkdir(parents=True, exist_ok=True)
                        options.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                report["summary"].append({"mesh": [nx, ny], "optimizer": method,
                    "median_seconds": {name: median(values) for name, values in timings.items()}})
        options.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        methods.mmasub, methods.gcmmasub = saved


if __name__ == "__main__":
    main()
