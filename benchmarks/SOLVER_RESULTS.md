# Exact-state reuse and PETSc solver comparison

Measured on 14 November 2026 with Python 3.14.4, PyTorch 2.15.0,
PETSc/petsc4py 3.26.1, Windows 11, and 20 logical CPUs. Raw timings,
environment/memory snapshots, configurations and numerical errors are in
[`results/2026-11-14-solver-experiments.json`](results/2026-11-14-solver-experiments.json).
The measured implementation is commit `6ce50ca`, built on `c3eeee1`;
independent morphology and ROM assembly updates were subsequently integrated
and the combined implementation passed the full 248-test suite. All measured
runs use full-order elasticity and disable the optional maximum-size constraint.

## Improvements

The exact-modulus one-entry cache avoids solving an accepted GCMMA trial a
second time. It copies cached states, rebuilds the autograd graph on each
evaluation, and checks the actual element moduli rather than the raw design.
Changes in material law or projection therefore cannot return stale states.
The formulation remains epigraph MMA/GCMMA with three compliance inequalities
and a dilated-volume constraint.

The app also keeps one exclusively leased compatible group of three spawned
field workers. Each worker uses one PyTorch thread and its own PETSc objects.
Per-problem caches and counters reset on every run. Compatibility includes
mesh, geometry, material constants, load, tolerance, solver method and ROM
basis identity. An idle group expires after 60 seconds; incompatible jobs,
nonparallel runs, training and server shutdown retire it. Python callers may
explicitly share `ml_opt.fem.worker_pool.FieldWorkerPool` through
`optimise(config, field_pool=pool)` for serial run submission. Without a shared
pool, `optimise` closes its own workers after the run.

Wall time in seconds for 100 robust GCMMA updates, median of three runs:

| Mesh / execution | Cache disabled, fresh workers | Exact cache / compatible warm pool | Speedup |
|---|---:|---:|---:|
| 60×20 sequential | 4.731 | 3.372 | 1.40× |
| 60×20 parallel | 5.957 | 1.816 | 3.28× |
| 120×40 sequential | 35.584 | 24.227 | 1.47× |
| 120×40 parallel | 26.673 | 16.900 | 1.58× |

The baseline disables cache hits in the same implementation; it is not a
checkout of historical code. Baseline/improved order alternates each repeat.
Improved parallel timing represents a **repeat Run click**: a separate two-update
warmup creates a compatible pool outside the timed interval. A cold first run
still pays worker/import startup. Sequential comparisons do not benefit from a
field pool. Timing includes problem setup and cleanup, excludes that warmup and
browser/network time, and runs without concurrent test/benchmark jobs.

Each 100-update run avoids 300 duplicate field solves: 660→360 at 60×20 and
657→357 at 120×40. Final designs and all compared gradients/compliances are
identical with caching enabled or disabled. Remaining work includes numerical
factorisation for each changed density, MMA/GCMMA subproblems and, in parallel,
queue transfer. Keeping workers alive removes startup, not these costs.

## Alternatives to LU

All choices use the same assembled positive-definite clamped elasticity
system. Cholesky uses PETSc `preonly`/`cholesky`; CG uses an unpreconditioned
residual norm with ICC or GAMG, relative tolerance 1e-10, absolute tolerance
1e-14 and an explicit true equilibrium residual check. GAMG receives block
size two and all three rigid-body near-nullspace modes. Failed convergence is
reported; there is no silent LU fallback. Factors/preconditioners update when
matrix values change.

PETSc documents [Cholesky for SPD systems](https://petsc.org/release/manualpages/PC/PCCHOLESKY/)
and the [near-nullspace and block-size guidance for GAMG](https://petsc.org/release/manualpages/PC/PCGAMG/).

Warm three-field evaluation time in seconds, median across four frozen LU
trajectory designs (iterations 0, 10, 25 and 50), repeated three times:

| Mesh | LU | Cholesky | CG/ICC | CG/GAMG |
|---|---:|---:|---:|---:|
| 60×20 | 0.01529 | 0.00994 | 0.02407 | 0.03307 |
| 120×40 | 0.12336 | 0.08535 | 0.30395 | 0.24955 |

The design changes between evaluations, preventing cache hits. Times include
assembly, state solves, filtering, projection and all design gradients. Maximum
relative field-compliance error versus LU was 8.62e-12; maximum relative field
gradient error was 2.05e-11. Maximum true residual was 9.76e-11. CG/GAMG reduced
iteration counts versus ICC but hierarchy setup still made it slower on these
meshes. These are empirical small/medium mesh results, not large-mesh scaling
claims. This native build uses MPIUNI; field parallelism is three local
processes, not distributed-matrix MPI or GPU execution.

Full 100-update sequential optimisation, exact cache enabled for both solvers,
median of three runs:

| Mesh / optimiser | LU (s) | Cholesky (s) |
|---|---:|---:|
| 60×20 MMA | 2.117 | 1.864 |
| 60×20 GCMMA | 2.398 | 2.590 |
| 120×40 MMA | 21.173 | 17.652 |
| 120×40 GCMMA | 25.391 | 19.755 |

The full-run phase ran LU before Cholesky within each repeat. Small-case timings
vary and do not show a universal improvement; the machine also had substantial
background memory use (recorded in the raw data). LU remains the compatibility
default, and **Optimise → Full-order method** exposes all four methods. The
120×40 GCMMA result improves by 22% with Cholesky. Comparing complete Cholesky
and LU runs, maximum absolute design difference was 1.88e-9, maximum relative
field-compliance difference 4.09e-12 and field-gradient difference 7.44e-11;
solve/retry counts matched. CG alternatives were screened on frozen evolving
designs and tested on shorter optimisations; 100-update CG timings are not
claimed here.

## Reproduction and checks

From the checkout whose source is under test, with the environment installed:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
python benchmarks/solver_experiments.py --phase screen
python benchmarks/solver_experiments.py --phase improvements --iterations 100 --repeats 3
python benchmarks/solver_experiments.py --phase optimisations --solvers lu,cholesky --iterations 100 --repeats 3
python -m pytest -q
```

Use the checkout's Python environment (`.venv/Scripts/python.exe` in the main
checkout on this machine). Each phase updates the same JSON, retaining the
other phases. `--output` selects another file. `--meshes` accepts `60x20,120x40`.
The screen phase always builds a 50-update reference trajectory.

Tests cover exact cache invalidation/copy isolation, cached versus uncached
optimisation trajectories, finite-difference gradients, disconnected high
contrast designs, all solver methods sequentially and in parallel, repeated
HTTP app runs, worker configuration changes, expiry, failure recovery and
cleanup. The merged suite passed **248 tests** with one dependency deprecation
warning. No fresh numerical factors are skipped for changed material.
