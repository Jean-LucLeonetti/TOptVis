"""Compare exact-state caching, warm field pools, and PETSc elasticity methods.

Run heavy phases serially in the normal app environment. State screening uses
the same LU trajectory designs for every method; full runs check the resulting
optimisation trajectories rather than assuming residual agreement is enough.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from benchmark import environment, summary
from ml_opt.config import CantileverConfig
from ml_opt.fem.cache import StateCache
from ml_opt.fem.worker_pool import FieldWorkerPool
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise

ROOT = Path(__file__).resolve().parents[1]


def configuration(nx, ny, method="gcmma", execution="sequential", iterations=100, solver="lu"):
    return CantileverConfig(nx=nx, ny=ny, robust=True, optimizer=method, field_execution=execution,
        filter_radius=.1, projection_beta=4, max_iterations=iterations,
        min_iterations=iterations+1, linear_solver=solver)


def identity(c):
    return f"{c.nx}x{c.ny}-{c.optimizer}-{c.field_execution}-{c.linear_solver}"


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-b) / max(np.linalg.norm(b), 1e-30))


def compare(actual, expected):
    return {"compliance_relative_error": relative(actual.compliance, expected.compliance),
        "displacement_relative_error": relative(actual.displacement, expected.displacement),
        "max_field_gradient_relative_error": max(relative(actual.field_gradients[name], expected.field_gradients[name])
                                                  for name in ("eroded", "nominal", "dilated")),
        "max_field_compliance_relative_error": max(relative(actual.field_compliances[name], expected.field_compliances[name])
                                                    for name in ("eroded", "nominal", "dilated"))}


@contextmanager
def cache_setting(enabled):
    original = StateCache.get
    if not enabled:
        StateCache.get = lambda self, moduli: None
    try:
        yield
    finally:
        StateCache.get = original


def run(c, *, pool=None, cached=True):
    with cache_setting(cached):
        started = time.perf_counter()
        result = optimise(c, field_pool=pool)
        elapsed = time.perf_counter()-started
    last = result.history[-1]
    return result, {"seconds": elapsed, "initial_seconds": result.history[0].elapsed_seconds,
        "updates_seconds": last.elapsed_seconds-result.history[0].elapsed_seconds,
        "cleanup_seconds": elapsed-last.elapsed_seconds, "state_solves": last.state_solves,
        "iterations": last.iteration, "inner_retries": sum(row.inner_iterations for row in result.history),
        "compliance": result.evaluation.compliance, "residual": result.evaluation.residual,
        "volume_constraint": result.evaluation.constraint, "epigraph_constraint": result.evaluation.epigraph_constraint}


def improvements(meshes, iterations, repeats, save):
    data = []
    optimise(configuration(6, 2, iterations=2))
    for nx, ny in meshes:
        for execution in ("sequential", "parallel"):
            c = configuration(nx, ny, execution=execution, iterations=iterations)
            with FieldWorkerPool() as pool:
                for repetition in range(repeats):
                    records, outputs = {}, {}
                    order = ("baseline", "improved") if repetition % 2 == 0 else ("improved", "baseline")
                    for variant in order:
                        if variant == "baseline":
                            pool.clear()  # Do not keep idle extra workers during the baseline.
                            result, timing = run(c, cached=False)
                        else:
                            # Warm the compatible pool, outside the measured run, like
                            # a second Run click. All per-problem caches start empty.
                            optimise(replace(c, max_iterations=2, min_iterations=3), field_pool=pool)
                            result, timing = run(c, pool=pool)
                        records[variant], outputs[variant] = timing, result
                        print(f"improvements {identity(c)} {variant} #{repetition+1}: {timing['seconds']:.3f}s; {timing['state_solves']} solves", flush=True)
                    errors = compare(outputs["improved"].evaluation, outputs["baseline"].evaluation)
                    errors["max_design_difference"] = float(np.max(np.abs(outputs["improved"].design-outputs["baseline"].design)))
                    np.testing.assert_array_equal(outputs["improved"].design, outputs["baseline"].design)
                    data.append({"config": asdict(c), "repetition": repetition, "runs": records, "errors": errors})
                    save("improvements", data)


def screen(meshes, repeats, save):
    data = []
    for nx, ny in meshes:
        c = configuration(nx, ny, iterations=50)
        designs = []
        def capture(row, design, evaluation):
            if row.iteration in (0, 10, 25, 50):
                designs.append((row.iteration, design, evaluation))
        optimise(c, evaluation_callback=capture)
        for solver in ("lu", "cholesky", "cg-icc", "cg-gamg"):
            entry = {"mesh": [nx, ny], "linear_solver": solver, "samples": [], "status": "ok"}
            try:
                with CantileverProblem(replace(c, linear_solver=solver)) as problem:
                    problem.evaluate(designs[-1][1])  # Warm graph/factors; next design differs.
                    for repetition in range(repeats):
                        for iteration, design, expected in designs:
                            started = time.perf_counter()
                            result = problem.evaluate(design)
                            elapsed = time.perf_counter()-started
                            entry["samples"].append({"iteration": iteration, "repetition": repetition,
                                "seconds": elapsed, "residual": result.residual, "linear_iterations": result.linear_iterations,
                                **compare(result, expected)})
                entry["seconds"] = summary([s["seconds"] for s in entry["samples"]])
                print(f"screen {nx}x{ny} {solver}: {entry['seconds']['median']:.5f}s / three-field evaluation", flush=True)
            except Exception as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                print(f"screen {nx}x{ny} {solver}: FAILED {str(error)[:180]}", flush=True)
            data.append(entry)
            save("screen", data)


def optimisations(meshes, iterations, repeats, solvers, save):
    data = []
    optimise(configuration(6, 2, iterations=2))
    for nx, ny in meshes:
        for method in ("mma", "gcmma"):
            c = configuration(nx, ny, method=method, iterations=iterations)
            references = {}
            for repetition in range(repeats):
                # Always retain one LU reference for each same-machine comparison.
                candidates = ["lu"] + [s for s in solvers if s != "lu"]
                for solver in candidates:
                    entry = {"config": asdict(replace(c, linear_solver=solver)), "repetition": repetition}
                    try:
                        result, timing = run(replace(c, linear_solver=solver))
                        entry.update(status="ok", timing=timing)
                        if solver == "lu":
                            references[repetition] = result
                        reference = references[repetition]
                        entry["errors"] = {**compare(result.evaluation, reference.evaluation),
                            "max_design_difference": float(np.max(np.abs(result.design-reference.design)))}
                        print(f"optimise {nx}x{ny} {method} {solver} #{repetition+1}: {timing['seconds']:.3f}s", flush=True)
                    except Exception as error:
                        entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                        print(f"optimise {nx}x{ny} {method} {solver}: FAILED {str(error)[:180]}", flush=True)
                    data.append(entry)
                    save("optimisations", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("improvements", "screen", "optimisations"), required=True)
    parser.add_argument("--meshes", default="60x20,120x40")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--solvers", default="lu,cholesky,cg-icc,cg-gamg")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/2026-11-14-solver-experiments.json")
    args = parser.parse_args()
    meshes = [tuple(map(int, pair.split("x"))) for pair in args.meshes.split(",")]
    report = json.loads(args.output.read_text()) if args.output.exists() else {}
    report.setdefault("source_commit", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip())
    report.setdefault("invocations", []).append({"phase": args.phase, "iterations": args.iterations,
        "repeats": args.repeats, "environment": environment()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        report[name] = data
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    if args.phase == "improvements":
        improvements(meshes, args.iterations, args.repeats, save)
    elif args.phase == "screen":
        screen(meshes, args.repeats, save)
    else:
        optimisations(meshes, args.iterations, args.repeats, args.solvers.split(","), save)


if __name__ == "__main__":
    main()
