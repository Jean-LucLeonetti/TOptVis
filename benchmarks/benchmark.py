"""Reproduce core, PETSc-event, payload, startup, and HTTP-stream timings.

Run with the project's Python: python benchmarks/benchmark.py
Benchmarks run serially. Profiling is separate from uninstrumented wall timings.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/benchmarks"


def memory():
    if sys.platform != "win32":
        return {}
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "available", "total_page",
                "available_page", "total_virtual", "available_virtual", "extended")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {"total_ram_bytes": status.total, "available_ram_bytes": status.available,
            "memory_load_percent": status.load}


def environment():
    import importlib.metadata
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "torch", "petsc4py", "mmapy", "fastapi", "uvicorn")},
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "parallel_info": torch.__config__.parallel_info(), "memory": memory()}


def summary(samples):
    return {"samples": samples, "median": statistics.median(samples),
            "min": min(samples), "max": max(samples)}


def configuration(case):
    from ml_opt.config import CantileverConfig
    return CantileverConfig(nx=case["nx"], ny=case["ny"], optimizer=case["method"],
                            max_iterations=case["iterations"])


def core(case, repeats):
    from dataclasses import replace
    from ml_opt.runner import optimise
    config = configuration(case)
    optimise(replace(config, max_iterations=2))
    trials = []
    for _ in range(repeats):
        before = memory()
        started = time.perf_counter()
        result = optimise(config)
        elapsed = time.perf_counter() - started
        if not math.isfinite(result.evaluation.compliance) or result.evaluation.residual > 1e-7:
            raise RuntimeError("Benchmark solution failed its equilibrium check")
        trials.append({"seconds": elapsed, "iterations": result.history[-1].iteration,
                       "state_solves": result.history[-1].state_solves,
                       "inner_retries": sum(entry.inner_iterations for entry in result.history),
                       "compliance": result.evaluation.compliance,
                       "volume": result.evaluation.volume_fraction,
                       "residual": result.evaluation.residual, "stop_reason": result.stop_reason,
                       "memory_before": before, "memory_after": memory()})
    return {"case": case, "timing": summary([trial["seconds"] for trial in trials]),
            "trials": trials, "environment": environment()}


def function_name(filename, line, name):
    text = filename.replace("\\", "/")
    root = ROOT.as_posix() + "/"
    if text.startswith(root):
        text = text[len(root):]
    if "/site-packages/" in text:
        text = "site-packages/" + text.split("/site-packages/", 1)[1]
    return f"{text}:{line}({name})"


def profile(case):
    import cProfile
    import pstats
    import numpy as np
    from petsc4py import PETSc
    from ml_opt.fem.elasticity import ElasticitySolver
    from ml_opt.problem import CantileverProblem
    import ml_opt.optimizers.moving_asymptotes as methods
    import torch
    from ml_opt.runner import optimise
    config = configuration(case)
    from dataclasses import replace
    optimise(replace(config, max_iterations=2))
    original = ElasticitySolver.solve
    phase_times = defaultdict(float)
    phase_calls = defaultdict(int)
    replacements = []
    def timed(owner, attribute, label):
        function = getattr(owner, attribute)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                phase_times[label] += time.perf_counter() - started
                phase_calls[label] += 1
        replacements.append((owner, attribute, function))
        setattr(owner, attribute, wrapped)
    previous = None
    repeated = 0
    repeated_seconds = 0.0
    def observed(solver, moduli):
        nonlocal previous, repeated, repeated_seconds
        duplicate = previous is not None and np.array_equal(previous, moduli)
        previous = np.array(moduli, copy=True)
        started = time.perf_counter()
        result = original(solver, moduli)
        elapsed = time.perf_counter() - started
        phase_times["elasticity_total"] += elapsed
        phase_calls["elasticity_total"] += 1
        if duplicate:
            repeated += 1
            repeated_seconds += elapsed
        return result
    PETSc.Log.begin()
    names = ("MatLUFactorSym", "MatLUFactorNum", "MatSolve", "PCSetUp", "KSPSetUp",
             "KSPSolve", "MatAssemblyBegin", "MatAssemblyEnd")
    events = {name: PETSc.Log.Event(name) for name in names}
    before = {name: event.getPerfInfo() for name, event in events.items()}
    ElasticitySolver.solve = observed
    timed(CantileverProblem, "evaluate", "evaluation_total")
    for function in ("mmasub", "gcmmasub", "asymp", "raaupdate", "concheck"):
        timed(methods, function, function)
    timed(torch.autograd, "grad", "torch_grad")
    started = time.perf_counter()
    try:
        result = optimise(config)
    finally:
        diagnostic_elapsed = time.perf_counter() - started
        ElasticitySolver.solve = original
        for owner, attribute, function in reversed(replacements):
            setattr(owner, attribute, function)
    native = {name: {key: event.getPerfInfo()[key] - before[name][key]
                     for key in ("count", "time", "flops")} for name, event in events.items()}
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.runcall(optimise, config)
    elapsed = time.perf_counter() - started
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(OUTPUT / f"{case['nx']}x{case['ny']}-{case['method']}.prof"))
    stats = pstats.Stats(profiler)
    functions = [{"function": function_name(*key), "calls": value[1],
                  "self_seconds": value[2], "cumulative_seconds": value[3]}
                 for key, value in stats.stats.items()]
    requested = {}
    for label, filename, name in (
        ("elasticity_total", "fem/elasticity.py", "solve"),
        ("evaluation_total", "problem.py", "evaluate"),
        ("optimizer_inclusive", "moving_asymptotes.py", "step"),
        ("mma_subproblems", "mmapy/mma.py", "mmasub"),
        ("gcmma_subproblems", "mmapy/mma.py", "gcmmasub"),
        ("primal_dual_subsolver", "mmapy/mma.py", "subsolv"),
        ("torch_grad", "autograd/__init__.py", "grad"),
        ("coo_to_csr", "sparse/_coo.py", "tocsr")):
        matches = [value for key, value in stats.stats.items()
                   if key[0].replace("\\", "/").endswith(filename) and key[2] == name]
        requested[label] = {"calls": sum(value[1] for value in matches),
                            "seconds": sum(value[3] for value in matches)}
    last = result.history[-1]
    from multiprocessing.reduction import ForkingPickler
    payload_times = defaultdict(list)
    event = None
    for _ in range(100):
        start = time.perf_counter()
        event = {"event": "iteration", "history": asdict(last), "density": result.evaluation.density.tolist()}
        payload_times["snapshot_seconds"].append(time.perf_counter() - start)
        start = time.perf_counter()
        serialized = json.dumps(event, allow_nan=False)
        payload_times["json_seconds"].append(time.perf_counter() - start)
        start = time.perf_counter()
        ForkingPickler.dumps(event)
        payload_times["pickle_seconds"].append(time.perf_counter() - start)
    return {"case": case, "profiled_seconds": elapsed, "diagnostic_seconds": diagnostic_elapsed,
            "phase_times": dict(phase_times), "phase_calls": dict(phase_calls), "iterations": last.iteration,
            "state_solves": last.state_solves,
            "inner_retries": sum(entry.inner_iterations for entry in result.history),
            "consecutive_duplicate_solves": repeated, "duplicate_solve_seconds": repeated_seconds,
            "functions": requested, "petsc_events": native,
            "top_self_functions": sorted(functions, key=lambda item: -item["self_seconds"])[:25],
            "top_cumulative_functions": sorted(functions, key=lambda item: -item["cumulative_seconds"])[:25],
            "payload": {"bytes": len(serialized.encode()),
                        **{key: summary(value) for key, value in payload_times.items()}}}


@contextmanager
def web_log_directory():
    import tempfile
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="web-", dir=OUTPUT) as temporary:
        assert Path(temporary).resolve().parent == OUTPUT.resolve()
        try:
            yield temporary
        finally:
            # A forcefully stopped Windows parent cannot join its worker. Wait
            # for parent-death detection and Python teardown to release the log.
            deadline = time.perf_counter() + 10
            while True:
                try:
                    (Path(temporary) / "server.log").unlink(missing_ok=True)
                    break
                except PermissionError:
                    if time.perf_counter() >= deadline:
                        raise RuntimeError("The worker retained its server log after parent exit")
                    time.sleep(.05)


def web_benchmark(repeats):
    import socket
    import urllib.error
    import urllib.request
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with web_log_directory() as temporary:
        with (Path(temporary) / "server.log").open("w") as log:
            started = time.perf_counter()
            server = subprocess.Popen([sys.executable, "-m", "ml_opt.web", "--port", str(port)],
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                deadline = time.perf_counter() + 90
                while time.perf_counter() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError("Benchmark server exited:\n" + (Path(temporary) / "server.log").read_text())
                    try:
                        with urllib.request.urlopen(base + "/api/health", timeout=.2) as response:
                            health = json.load(response)
                        break
                    except (OSError, urllib.error.URLError):
                        time.sleep(.05)
                else:
                    raise RuntimeError("Benchmark server startup timed out")
                readiness = time.perf_counter() - started
                # Record the one-time import cost separately from warm Run clicks.
                while "worker_ready" in health and not health["worker_ready"]:
                    if time.perf_counter() >= deadline:
                        raise RuntimeError("Worker warmup timed out")
                    time.sleep(.05)
                    with urllib.request.urlopen(base + "/api/health", timeout=1) as response:
                        health = json.load(response)
                worker_readiness = time.perf_counter() - started
                runs = []
                for method in ("mma", "gcmma"):
                    for repetition in range(repeats):
                        body = {"nx": 60, "ny": 20, "load_node_x": 60, "load_node_y": 10,
                                "optimizer": method, "max_iterations": 20, "filter_radius": 0,
                                "projection_beta": 0}
                        request = urllib.request.Request(base + "/api/run", json.dumps(body).encode(),
                                                          {"Content-Type": "application/json"})
                        started = time.perf_counter()
                        first = previous = None
                        interframes = []
                        byte_count = frames = 0
                        memory_before = memory()
                        with urllib.request.urlopen(request, timeout=120) as response:
                            for line in response:
                                byte_count += len(line)
                                if not line.strip():
                                    continue
                                event = json.loads(line)
                                if event["event"] == "error":
                                    raise RuntimeError(event["message"])
                                if event["event"] == "iteration":
                                    now = time.perf_counter()
                                    first = now if first is None else first
                                    if previous is not None:
                                        interframes.append(now - previous)
                                    previous = now
                                    frames += 1
                        elapsed = time.perf_counter() - started
                        if frames != 21:
                            raise RuntimeError("Incomplete benchmark stream")
                        runs.append({"method": method, "repetition": repetition,
                                     "total_seconds": elapsed, "first_frame_seconds": first - started,
                                     "after_first_frame_seconds": elapsed - (first - started),
                                     "interframe_seconds": summary(interframes), "frames": frames,
                                     "bytes": byte_count, "memory_before": memory_before, "memory_after": memory()})
                        print(f"HTTP {method} trial {repetition + 1}: {elapsed:.3f}s; first frame {first - started:.3f}s", flush=True)
                return {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        "server_readiness_seconds": readiness,
                        "worker_readiness_seconds": worker_readiness, "runs": runs}
            finally:
                server.terminate()
                server.wait(timeout=20)


def worker_import_probe(repeats):
    samples = []
    code = ("import json,time; started=time.perf_counter(); import ml_opt.web.worker; "
            "print(json.dumps({'import_seconds':time.perf_counter()-started}))")
    for _ in range(repeats):
        started = time.perf_counter()
        child = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                               text=True, check=True, timeout=90)
        samples.append({**json.loads(child.stdout), "process_seconds": time.perf_counter() - started})
    return {"trials": samples, "import_seconds": summary([item["import_seconds"] for item in samples]),
            "process_seconds": summary([item["process_seconds"] for item in samples])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--child", choices=("core", "profile"))
    parser.add_argument("--case")
    parser.add_argument("--profiles-only", action="store_true", help="Refresh diagnostics in an existing result file")
    parser.add_argument("--startup-only", action="store_true", help="Refresh the worker import probe")
    parser.add_argument("--web-only", action="store_true", help="Refresh HTTP timings and worker import probe")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results" / f"{datetime.now(timezone.utc).date().isoformat()}.json")
    options = parser.parse_args()
    if options.child:
        case = json.loads(options.case)
        result = core(case, options.repeats) if options.child == "core" else profile(case)
        options.destination.write_text(json.dumps(result, indent=2))
        return
    if options.repeats < 1:
        parser.error("repeats must be positive")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "repeats": options.repeats, "initial_memory": memory(), "core": [], "profiles": []}
    if options.profiles_only or options.startup_only or options.web_only:
        result = json.loads(options.output.read_text())
    if options.web_only:
        result["web"] = web_benchmark(options.repeats)
        result["worker_import_probe"] = worker_import_probe(options.repeats)
        options.output.write_text(json.dumps(result, indent=2))
        return
    if options.startup_only:
        result["worker_import_probe"] = worker_import_probe(options.repeats)
        options.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result["worker_import_probe"], indent=2))
        return
    if options.profiles_only:
        result["profiles"] = []
    cases = [{"nx": nx, "ny": ny, "method": method, "iterations": steps}
             for nx, ny, steps in ((30, 10, 20), (60, 20, 20), (120, 40, 20), (60, 20, 100))
             for method in ("mma", "gcmma")]
    for mode, selected in (("core", [] if options.profiles_only else cases),
                           ("profile", [case for case in cases if case["iterations"] == 20])):
        for case in selected:
            destination = OUTPUT / "child-result.json"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", mode,
                            "--case", json.dumps(case), "--repeats", str(options.repeats),
                            "--destination", str(destination)], cwd=ROOT, check=True, timeout=600)
            data = json.loads(destination.read_text())
            result["core" if mode == "core" else "profiles"].append(data)
            options.output.write_text(json.dumps(result, indent=2))
            timing = data.get("timing", {}).get("median", data.get("profiled_seconds"))
            print(f"{mode} {case['nx']}x{case['ny']} {case['method']} {case['iterations']}: {timing:.3f}s", flush=True)
    if not options.profiles_only:
        result["web"] = web_benchmark(options.repeats)
        result["worker_import_probe"] = worker_import_probe(options.repeats)
    options.output.write_text(json.dumps(result, indent=2))
    print(f"Results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
