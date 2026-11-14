# Reduced element assembly benchmark

The ROM now precomputes `V_e.T K_e V_e` during basis building and stores its
upper triangle for each element. Online solves form the weighted sum using
the current element moduli. No full stiffness matrix is assembled or
projected. Displacement, consistent element-energy gradients, and the actual
full equilibrium residual are retained; the latter uses an element scatter.

## Measurements

Measured on 14 November 2026 on this Windows workstation, using the existing
Python/PETSc environment and thread settings. Trials ran serially after the
process was warmed up, alternating the solver order over three repetitions.
Raw samples, environment, basis identities, state counts, errors, and source
fingerprints are in [the JSON report](results/2026-11-14-rom-online.json).

Training used nine full-order MMA trajectories: load height 0.25/0.5/0.75,
vertical downward load, volume 0.3/0.4/0.5, 20 outer updates, every second state
saved (99 snapshots per mesh). Normalised Euclidean POD used energy target
`1 - 1e-8` and a 64-mode cap. The selected ranks were 48 and 46; both met the
POD energy target. Packed operator storage was 43.1 and 158.3 MiB.

Each elasticity timing averages the 21 states of the same independent FOM
trajectory before taking the median over repetitions. All three solvers see
identical physical moduli and all compute element energies and full residuals.
The previous ROM is loaded from Git revision `c3eeee1` without modifying the
checkout. Its reduced compliance and the new assembly agree within relative
`5.3e-12` across the sampled states on both meshes.

| Mesh | FOM state solve | Previous ROM | Reduced element ROM | FOM / new ROM |
| --- | ---: | ---: | ---: | ---: |
| 120×40 | 24.78 ms | 12.37 ms | 4.11 ms | 6.0× |
| 240×80 | 205.75 ms | 47.99 ms | 14.62 ms | 14.1× |

Complete optimisation runs use an unseen combination: load height 0.4,
angle -75°, and volume 0.35. Every run performs 20 outer updates plus the
initial evaluation, with Helmholtz length 0.1, fixed projection beta 2,
and SIMP exponent 3. Per-run model loading, setup, gradient calculations,
and teardown are included. Interpreter/server startup, HTTP/browser work,
and final output writing are excluded.

| Mesh | Optimiser | FOM median (range), s | ROM median (range), s | Speedup | FOM / ROM state solves |
| --- | --- | ---: | ---: | ---: | ---: |
| 120×40 | MMA | 1.888 (1.594–1.891) | 1.313 (1.187–1.411) | 1.44× | 21 / 21 |
| 120×40 | GCMMA | 3.298 (3.225–3.730) | 1.828 (1.800–1.979) | 1.80× | 53 / 48 |
| 240×80 | MMA | 7.112 (7.039–7.294) | 4.947 (4.849–5.035) | 1.44× | 21 / 21 |
| 240×80 | GCMMA | 16.664 (16.244–18.887) | 8.295 (7.978–9.162) | 2.01× | 52 / 53 |

These are equal-iteration timing comparisons. GCMMA trial counts can differ
because the ROM changes the objective and gradients. The isolated state
comparison above holds the physical designs fixed. The available RAM at the
start was approximately 1.89 GiB; these are local workstation measurements,
not a dedicated idle-machine throughput limit.

## Accuracy and amortisation

The FOM was independently solved on each final ROM design, including the
same filtering, projection, material law, and raw-design gradient chain.
All final designs satisfied the physical volume constraint, but **none met
the comparison tolerances** (5% displacement, 1% compliance, 10% gradient).

| Mesh | Optimiser | Displacement error | Compliance error | Gradient error | FOM compliance of ROM design / FOM-optimised design |
| --- | --- | ---: | ---: | ---: | ---: |
| 120×40 | MMA | 97.20% | 97.21% | 99.93% | 23.41× |
| 120×40 | GCMMA | 95.33% | 95.69% | 99.90% | 16.59× |
| 240×80 | MMA | 93.37% | 92.75% | 99.90% | 10.04× |
| 240×80 | GCMMA | 71.96% | 75.87% | 96.99% | 3.13× |

The errors compare ROM and FOM on the **same final ROM design**; the last
column separately compares the physical quality of the two optimised designs.
The new assembly preserves the previous ROM equations. Its speed gain does
not correct frozen-basis coverage or make POD snapshot energy an optimisation
accuracy certificate. Basis enrichment or a guarded FOM/ROM strategy remains
necessary before treating these optimisations as reliable replacements.

| Mesh | Snapshot generation | Snapshot write | POD + operators + basis write | Offline total | Timing-only break-even, MMA / GCMMA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 120×40 | 15.71 s | 0.23 s | 1.52 s | 17.45 s | 31 / 12 runs |
| 240×80 | 81.97 s | 0.76 s | 5.28 s | 88.01 s | 41 / 11 runs |

Break-even is `ceil(offline cost / (FOM median - ROM median))`. These counts
describe timing only; the accuracy failures mean they do not establish a
break-even point for equally accurate solutions. They also exclude any future
enrichment or periodic FOM checks.

## Reproduce

From the desired checkout, using the project environment:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe benchmarks/rom_online.py --repeats 3
```

For a worktree sharing the main checkout's environment, use its absolute
Python path. The default output is `outputs/benchmarks/rom-online/report.json`;
generated snapshot/basis artifacts and final design arrays remain under that
ignored output directory. `--reuse-training` reuses the generated training
artifacts and preserves their originally measured offline costs. `--meshes`,
`--iterations`, `--training-iterations`, `--energy`, and `--modes` allow other
budgets. Benchmark-only training uses the same POD/operator builder with
explicit `CantileverConfig` cases, leaving the interactive mesh limits intact.

New training artifacts contain the operators. Older bases work by preparing
them once at solver construction; rebuild them in the Training tab to save
the operators and move preparation offline. The operator-array limit is
512 MiB per model, with a clear error requesting fewer modes above that limit.
