# Solid/void constraint iteration performance

Measured 15 November 2026 in the project's native Windows Python/PETSc environment.

The optional solid/void constraints increase work in MMA/GCMMA's convex
subproblem solver. The compact morphology FFTs were a small part of the
iteration cost. The implementation now compiles the pinned primal-dual loop
and fuses its repeated residual evaluations, avoiding temporary residual arrays
and repeated small matrix/vector calls during backtracking. Equations, barrier
levels, Newton/backtracking limits and feasibility tolerances remain unchanged.
Fast-math reassociation is disabled. Single-constraint runs retain the existing
NumPy kernel.

## Paired warm timings

These are medians of three serial repeats, alternating baseline/improved order,
for **ten fixed iterations**, including problem setup and initial evaluation.
They compare the prior broadcast-diagonal Python kernel against the final
compiled/fused implementation. Neither profiling nor first-time compilation
ran alongside these timings.

Both size constraints were enabled: solid diameter 0.45, void diameter 0.8,
erosion/dilation sharpness 16, allowed fraction 0.001. Other settings: 3×1 domain,
Helmholtz length 0.1, projection β=8 and η=0.5, target volume 0.4 with an upper
bound, SIMP p=3, single field and PETSc LU. Convergence stopping was disabled
for the fixed iteration budget; these are latency comparisons, not claims that
ten iterations yield a converged or feasible topology.

| Mesh | Optimiser | Previous | Compiled/fused | Speedup |
|---|---|---:|---:|---:|
| 60×20 | MMA | 3.25 s | 1.26 s | 2.58× |
| 60×20 | GCMMA | 13.08 s | 4.75 s | 2.75× |
| 120×40 | MMA | 11.26 s | 7.26 s | 1.55× |
| 120×40 | GCMMA | 14.31 s | 9.42 s | 1.52× |

The maximum final-design difference was 9.2e-10. The raw records include
constraint values, compliance, state-solve counts and GCMMA inner retries.
Timings depend on hardware, loading, chosen sizes, β and which constraints are
active. Additional constraints still require more numerical work; the speedup
does not remove that cost.

## Remaining bottlenecks

A separate profile of the improved 120×40 GCMMA case took 9.17 s:

| Work | Profiled cumulative time | Approximate share |
|---|---:|---:|
| Primal-dual Newton kernel, 37 subproblems | 5.41 s | 59% |
| PETSc elasticity, 38 actual solves | 3.22 s | 35% |
| Other work, including morphology and autograd | remainder | 6% |

GCMMA evaluated 37 trial subproblems for ten updates. Accepted-state caching
avoided repeating ten of the state solves. The Newton solve remains the main
cost, followed by elasticity as the mesh grows. Simple norm/LAPACK substitutions
and scoped BLAS thread limits were explored but were not selected; residual
fusion gave the clearer gain.

## Preparation and validation

First preparation of the final kernel took 133.4 s in a separate cold-cache
observation. This is a one-time compilation cost for the source/runtime/CPU
combination. The browser worker prepares it before enabling Run and later
processes reuse the cache. The paired-run cache-load time is recorded separately
from iteration timings.

Validation covers all primal/dual outputs against the reference algorithm,
KKT residuals, both Newton-system branches, complete optimisation trajectories,
both volume bounds and equality, robust fields, solid/void gradients, and real
streamed results. The final fused implementation passed the complete 349-test
suite, in addition to its focused reference and numerical checks.

Reproduce from this checkout using its source:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe benchmarks/size_constraints.py --iterations 10 --repeats 3 --variants baseline compiled
```

Run separately from tests, profiling and other benchmark jobs.

- [Raw paired timings](results/2026-11-15-size-kernels.json)
- [Separate profile](results/2026-11-15-size-profile.json)
- [Compiled Newton loop](../src/ml_opt/optimizers/_subproblem.py)
- [Fused residual equations](../src/ml_opt/optimizers/_residual.py)
