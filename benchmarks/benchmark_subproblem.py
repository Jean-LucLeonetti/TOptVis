"""Compare mmapy's sparse diagonal arithmetic with broadcasting on real inputs.

This is an isolated experiment: no upstream module globals are modified.
Run: .venv/Scripts/python.exe benchmarks/benchmark_subproblem.py
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import time

import mmapy.mma as upstream
import numpy as np

from ml_opt.config import CantileverConfig
from ml_opt.optimizers._mmapy import mmasub, gcmmasub
import ml_opt.optimizers.moving_asymptotes as methods
from ml_opt.runner import optimise
from benchmark import environment


def copy_inputs(args, kwargs):
    def copy(value):
        return value.copy() if isinstance(value, np.ndarray) else value
    return tuple(map(copy, args)), {key: copy(value) for key, value in kwargs.items()}


def capture(nx, ny, method):
    name = "mmasub" if method == "mma" else "gcmmasub"
    previous = getattr(methods, name)
    original = getattr(upstream, name)
    calls = []
    def record(*args, **kwargs):
        calls.append(copy_inputs(args, kwargs))
        return original(*args, **kwargs)
    setattr(methods, name, record)
    try:
        optimise(CantileverConfig(nx=nx, ny=ny, optimizer=method, max_iterations=20))
    finally:
        setattr(methods, name, previous)
    return name, calls


def benchmark(repeats=3, loops=10):
    fast = {"mmasub": mmasub, "gcmmasub": gcmmasub}
    results = []
    for nx, ny in ((60, 20), (120, 40)):
        for method in ("mma", "gcmma"):
            name, all_calls = capture(nx, ny, method)
            calls = [all_calls[index] for index in sorted({0, len(all_calls)//2, len(all_calls)-1})]
            original, experimental = getattr(upstream, name), fast[name]
            maximum_error = 0.0
            # Check every captured primal/dual/slack/asymptote output, not just x.
            for args, kwargs in all_calls:
                reference_args, reference_kwargs = copy_inputs(args, kwargs)
                candidate_args, candidate_kwargs = copy_inputs(args, kwargs)
                reference = original(*reference_args, **reference_kwargs)
                candidate = experimental(*candidate_args, **candidate_kwargs)
                for left, right in zip(reference, candidate):
                    np.testing.assert_allclose(right, left, rtol=1e-10, atol=1e-11)
                    maximum_error = max(maximum_error, float(np.max(np.abs(np.asarray(left)-right))))
            samples = {"upstream": [], "broadcast": []}
            for repetition in range(repeats):
                order = (("upstream", original), ("broadcast", experimental))
                if repetition % 2:
                    order = order[::-1]
                for label, function in order:
                    started = time.perf_counter()
                    for _ in range(loops):
                        for args, kwargs in calls:
                            function(*args, **kwargs)
                    samples[label].append((time.perf_counter()-started)/(loops*len(calls)))
            item = {"nx": nx, "ny": ny, "method": method, "captured_calls": len(all_calls),
                    "maximum_output_absolute_error": maximum_error, "seconds_per_subproblem": samples,
                    "median_seconds": {key: statistics.median(value) for key, value in samples.items()}}
            item["speedup"] = item["median_seconds"]["upstream"] / item["median_seconds"]["broadcast"]
            results.append(item)
            print(f"{nx}x{ny} {method}: {item['speedup']:.2f}x; max output error {maximum_error:.3g}", flush=True)
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "environment": environment(), "repeats": repeats,
            "loops_per_representative_input": loops, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/2026-11-14-subproblem.json"))
    options = parser.parse_args()
    options.output.write_text(json.dumps(benchmark(), indent=2))
