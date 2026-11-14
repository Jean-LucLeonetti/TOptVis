# Performance improvements — 14 November 2026

Implemented the prewarmed persistent worker (option 2), persistent PETSc
structure (option 3), and the useful arithmetic improvement found while
benchmarking option 4. Accepted GCMMA trial-state caching (option 1) is deferred.
The model remains unfiltered, with the same MMA/GCMMA algorithms and defaults.

Numerical implementation: `a14b509`. Worker parent-exit protection and final
HTTP measurements: `3872ab2`. Original solver reference: `b6596a7`.
Raw data: [controlled comparison](results/2026-11-14-ablation.json),
[subproblem comparison](results/2026-11-14-subproblem.json),
[full benchmark and HTTP measurements](results/2026-11-14-optimised.json).
The [original measurements](RESULTS.md) are retained.

## Controlled comparison

Median seconds for **20 outer updates**, including initial evaluation and
problem creation/cleanup, excluding imports and HTTP. Three repetitions of
each implementation run in one process, rotating implementation order each
repetition. Each implementation first receives a two-update warm-up. The
original solver is loaded directly from its Git revision; the same runner,
configuration, and optimiser wrapper are used throughout. Only the elasticity
implementation and upstream versus broadcast subproblem routines are switched.

| Mesh | Method | Original | PETSc reuse only | PETSc + broadcasting | Total reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 60 × 20 | MMA | 0.335 s | 0.270 s | 0.175 s | 47.6% |
| 60 × 20 | GCMMA | 0.567 s | 0.382 s | 0.263 s | 53.6% |
| 120 × 40 | MMA | 1.438 s | 1.069 s | 0.948 s | 34.1% |
| 120 × 40 | GCMMA | 2.545 s | 1.742 s | 1.539 s | 39.5% |

PETSc reuse alone reduced these end-to-end times by 19–33%. Broadcasting
provides a further reduction of 31–35% at 60 × 20 and 11–12% at 120 × 40.
The largest relative final-compliance difference between original and updated
implementations was `2.20e-10`; state-solve and GCMMA retry counts agree.

The earlier fresh-process benchmark suite was also repeated. It had more
variable timings: 120 × 40 MMA ranged from 1.490 to 1.920 seconds, versus
0.910–1.107 seconds for the combined implementation in the alternating test.
Use the controlled table to compare implementations. These remain local
workstation measurements with other applications open and limited free RAM;
no affinity, power, or thread settings were changed. Raw files retain samples,
environment details, and residuals rather than hiding this variability.

## Worker reuse and interface latency

The server starts one background worker during application startup. That
process imports the numerical stack once and accepts sequential run commands.
Every run creates its own geometry, force vector, problem, and optimiser.
Run-tagged messages prevent stale frames from reaching a subsequent run.
Cancellation drains the old run before its cancel flag can be cleared.
Failures and disconnected clients are handled; a dead worker is recreated on
the next request. Server shutdown releases the worker, and forced parent exit
is detected during idle polling and between optimisation updates.
The imported numerical stack remains resident while the server is open;
per-problem PETSc resources are released after each run.

Default 60 × 20 mesh, 20 updates, three HTTP runs per method:

| Measurement | Original MMA | Updated MMA | Original GCMMA | Updated GCMMA |
| --- | ---: | ---: | ---: | ---: |
| Run request to first evaluated frame, median | 2.397 s | 0.030 s | 2.407 s | 0.035 s |
| Run request to stream EOF, median | 3.218 s | 0.217 s | 3.499 s | 0.315 s |

Updated measurements start after the worker is ready. The HTTP server became
ready at 2.951 seconds after launch; the worker at 5.381 seconds. This one-time
2.430-second background warm-up overlaps time spent editing the model. A Run
click during warm-up still waits for imports. `/api/health` exposes
`worker_ready` separately from server readiness. Iterations and densities are
still streamed from actual evaluated states.

## PETSc structure reuse

The fixed element-to-matrix mapping is preallocated once with
`Mat.setPreallocationCOO`. Each evaluation updates values with `INSERT_VALUES`,
which sums shared-element contributions and replaces the previous design's
values. PETSc handles the assembly, as documented for
[`MatSetValuesCOO`](https://petsc.org/release/manualpages/Mat/MatSetValuesCOO/).
The matrix, KSP, RHS, solution, and residual vector live for the problem's
lifetime. A fixed nonzero pattern allows PETSc's LU implementation to reuse
the symbolic factors while rebuilding numerical factors; see the
[PETSc LU implementation](https://petsc.org/release/src/ksp/pc/impls/factor/lu/lu.c.html).
Preconditioner reuse is deliberately disabled to avoid retaining stale factors.

PETSc event counts for 20 updates confirm:

| Mesh / method | Original symbolic LU | Updated symbolic LU | Updated numerical LU |
| --- | ---: | ---: | ---: |
| 60 × 20 MMA | 21 | 1 | 21 |
| 60 × 20 GCMMA | 43 | 1 | 43 |
| 120 × 40 MMA | 21 | 1 | 21 |
| 120 × 40 GCMMA | 45 | 1 | 45 |

The updated profiles have zero SciPy COO-to-CSR conversions. The runner closes
native resources on success, cancellation, and exceptions, which matters when
the worker stays alive across runs. A regression test compares multiple
changed-modulus solves against independent dense assembly and checks symbolic
versus numerical factorisation counts.

## Option 4: subproblem arithmetic

mmapy constructs sparse diagonal matrices to scale dense rows. A private
adapter replaces those operations with NumPy multiplication and adds small
diagonals directly. It binds the existing upstream function code to a private
globals dictionary; it does not modify installed mmapy files or upstream module
globals. Its MMA/GCMMA equations, convergence checks, and return structure are
unchanged. The dependency remains pinned to mmapy 0.3.1 because this adapter
depends on that version's internal subproblem routine.

The microbenchmark captures every subproblem input from 20 actual outer updates,
checks every returned primal/dual/slack/asymptote output, and times the first,
middle, and last captured inputs. It alternates implementation order across
three repetitions, with ten loops over those representative inputs per timing.
All captured outputs were identical, with maximum absolute difference **zero**.
The dedicated regression tests also require exact array equality.

| Mesh | Method | Upstream per subproblem | Broadcasting | Speedup |
| --- | --- | ---: | ---: | ---: |
| 60 × 20 | MMA | 18.99 ms | 7.38 ms | 2.57× |
| 60 × 20 | GCMMA | 13.30 ms | 5.95 ms | 2.24× |
| 120 × 40 | MMA | 44.72 ms | 24.76 ms | 1.81× |
| 120 × 40 | GCMMA | 35.50 ms | 23.12 ms | 1.54× |

These medians describe the representative-input microbenchmark, not complete
optimisation runtimes; the controlled comparison above measures the latter.

## Remaining bottlenecks

- Numerical LU is now the main large-mesh cost. In separate diagnostic runs it
  takes about 43% of total MMA time and 55% of GCMMA time at 120 × 40. Total
  elasticity work accounts for about 53% and 67%, respectively.
- The primal-dual subproblem still matters: about 59% of diagnostic MMA time
  at 60 × 20 and 43% at 120 × 40. Sparse diagonal allocations are removed, but
  its Newton iterations and dense vector work remain.
- GCMMA still performs 20 consecutive duplicate state solves in these 20-update
  cases. Accepted-trial caching was not part of this change. The large-mesh
  duplicate solves cost about 29% of the separate diagnostic run.
- The numerical imports still cost a few seconds once per server launch.
  Worker reuse removes that cost from subsequent Run clicks.

Validation: **26 tests passed**, including independent elasticity assembly,
finite-difference gradients, both optimisers, upstream arithmetic equivalence,
multiple changed configurations in one worker, cancellation, disconnects,
worker errors, crash recovery, and native cleanup on exceptions. The repeated
full numerical suite's maximum equilibrium residual was `2.01e-12`. HTTP
benchmarks also verified that the worker releases its inherited log after a
forced server exit.

## Reproduction

Run these serially, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe benchmarks\benchmark_ablation.py
.\.venv\Scripts\python.exe benchmarks\benchmark_subproblem.py
.\.venv\Scripts\python.exe benchmarks\benchmark.py --output benchmarks/results/2026-11-14-optimised.json
```

`benchmark.py --web-only --output <existing-json>` refreshes HTTP measurements
and the import probe without repeating core/profiling cases. Timings from
profiling and diagnostic wrappers are kept separate from uninstrumented wall
times.
