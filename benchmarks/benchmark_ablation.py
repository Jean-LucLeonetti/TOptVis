"""Alternate original, PETSc-only, and combined implementations in one process.

Load the original solver from its exact Git revision without changing the worktree.
Run serially after other benchmarks: python benchmarks/benchmark_ablation.py
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

import mmapy.mma as upstream
import numpy as np

from benchmark import environment, summary
from ml_opt.config import CantileverConfig
from ml_opt.fem.elasticity import ElasticitySolver
from ml_opt.optimizers._mmapy import mmasub, gcmmasub
import ml_opt.optimizers.moving_asymptotes as methods
import ml_opt.fem.state_solver as solver_module
from ml_opt.runner import optimise

BASELINE = "b6596a7"


def original_solver():
    source = subprocess.check_output(["git", "show", f"{BASELINE}:src/ml_opt/fem/elasticity.py"], text=True)
    module = ModuleType("ml_opt.fem._benchmark_original")
    module.__package__ = "ml_opt.fem"
    sys.modules[module.__name__] = module
    exec(compile(source, f"{BASELINE}/elasticity.py", "exec"), module.__dict__)
    # Original solve already destroys all its native resources on every call.
    module.ElasticitySolver.close = lambda self: None
    return module.ElasticitySolver


def benchmark(repeats):
    old_solver = original_solver()
    variants = {
        "original": (old_solver, upstream.mmasub, upstream.gcmmasub),
        "petsc_reuse": (ElasticitySolver, upstream.mmasub, upstream.gcmmasub),
        "petsc_and_broadcast": (ElasticitySolver, mmasub, gcmmasub),
    }
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "baseline_commit": BASELINE,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "environment": environment(), "repeats": repeats, "cases": []}
    try:
        for nx, ny in ((60, 20), (120, 40)):
            for method in ("mma", "gcmma"):
                config = CantileverConfig(nx=nx, ny=ny, optimizer=method, max_iterations=20)
                trials = {name: [] for name in variants}
                for solver, mma, gcmma in variants.values():
                    solver_module.ElasticitySolver, methods.mmasub, methods.gcmmasub = solver, mma, gcmma
                    optimise(replace(config, max_iterations=2))
                for repetition in range(repeats):
                    names = list(variants)
                    names = names[repetition % len(names):] + names[:repetition % len(names)]
                    for name in names:
                        solver_module.ElasticitySolver, methods.mmasub, methods.gcmmasub = variants[name]
                        started = time.perf_counter()
                        run = optimise(config)
                        elapsed = time.perf_counter() - started
                        assert np.isfinite(run.evaluation.compliance) and run.evaluation.residual < 1e-7
                        trials[name].append({"seconds": elapsed, "compliance": run.evaluation.compliance,
                            "volume": run.evaluation.volume_fraction, "residual": run.evaluation.residual,
                            "state_solves": run.history[-1].state_solves,
                            "inner_retries": sum(entry.inner_iterations for entry in run.history)})
                case = {"nx": nx, "ny": ny, "method": method, "iterations": 20,
                        "variants": {name: {"timing": summary([item["seconds"] for item in samples]),
                                            "trials": samples} for name, samples in trials.items()}}
                result["cases"].append(case)
                print(f"{nx}x{ny} {method}: " + "; ".join(
                    f"{name} {item['timing']['median']:.3f}s" for name, item in case["variants"].items()), flush=True)
    finally:
        solver_module.ElasticitySolver, methods.mmasub, methods.gcmmasub = ElasticitySolver, mmasub, gcmmasub
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/2026-11-14-ablation.json"))
    options = parser.parse_args()
    if options.repeats < 1:
        parser.error("repeats must be positive")
    options.output.write_text(json.dumps(benchmark(options.repeats), indent=2))
