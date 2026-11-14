# TOptVis

A Python project for topology optimisation with PyTorch and PETSc through
`petsc4py`.

## Performance

Measured timings, bottleneck locations, methodology, and raw data are in
[`benchmarks/IMPROVEMENTS.md`](benchmarks/IMPROVEMENTS.md), with the original
baseline retained in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). Source comments `PERF B1`–`B4`
mark the measured hotspots. The app now prewarms and reuses one solver worker;
each problem retains its PETSc matrix graph, vectors, KSP, and symbolic LU
factorisation. Numerical LU factors are updated for every density evaluation.
MMA/GCMMA diagonal scaling uses NumPy broadcasting through an isolated adapter
to the pinned mmapy 0.3.1 routines. Reproduce the serial Python/PETSc/HTTP benchmarks
with:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark.py
```

For the browser's actual canvas drawing functions, run
`python benchmarks/render_benchmark.py` and open its printed loopback URL.

## Visual interface

Double-click **Start TOptVis.bat** in the project folder. It uses the existing
`.venv`, starts the app, and opens the interface in your default browser once
the server is ready. Keep the launcher window open while using the app.
The launcher reuses a running server only when it matches this checkout's
current source. If an older server is still holding port 8000, it starts the
updated app on an available nearby port and opens that URL. Existing runs
stay undisturbed; the launcher prints the address it uses.

Start the local interface from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui]" --no-build-isolation
.\.venv\Scripts\python.exe -m ml_opt.web
```

Open [the cantilever workspace](http://127.0.0.1:8000/) in your browser.
The Windows setup script includes the interface dependencies. You can also
launch it with `toptvis-ui` after activating the environment; use `--port 8001`
if port 8000 is already occupied.

The top tabs separate **Cantilever 2D** from **ROM Training**. Switching tabs
preserves settings, results, and any active job. ROM Training generates
full-order displacement snapshots and builds reusable POD reduced bases.
The workspaces can be opened
directly with `/#cantilever-2d` and `/#rom-training`.

Within Cantilever 2D, the far-left tabs separate **Geometry**, **Load**, **Filters**,
**Optimisation**, and **Visualisation**. The configuration panel shows the selected
tab; the central workspace displays the mesh/topology or convergence graphs. Tab changes retain
all values. Use arrow keys, Home, and End to navigate the tab rail.

1. In **Geometry**, set the domain length, height, and number of elements. The geometry and mesh
   preview updates immediately.
2. Click any free mesh node to place the load, or enter its column and row in **Load**.
   The left edge remains clamped. Set the magnitude and direction; the arrow
   and force components update immediately. Angles are counterclockwise from
   the positive x direction, so -90 degrees points downward.
3. In **Material**, choose SIMP or RAMP and its penalty value.
4. In **Filters**, set the Helmholtz length and the Heaviside projection's
   sharpness β and threshold η. Larger β sharpens the solid/void transition;
   fixed β = 0 disables projection. Use **β schedule** for optional capped
   increases during the run; length and threshold remain fixed.
5. In **Optimisation**, choose MMA or GCMMA, the material volume, and the iteration limit. Click
   **Run optimisation** to see the topology, compliance, volume, and iteration
   values update as the actual solver runs. **Stop** retains the last evaluated
   design.

### Convergence visualisation

Select **Visualisation** on the far-left rail to see five live graphs:
compliance, physical material volume, maximum design change, relative
objective change, and cumulative elapsed solver time versus outer iteration.
The graphs use the actual evaluated
history rows received by the interface, including the projected physical volume.
Dashed reference lines show the selected volume target and the design/objective
stopping tolerances reported by the server when the run starts.

Compliance and volume include iteration 0; change curves start at iteration 1
because there is no previous design for the initial evaluation. Hover over a
graph to inspect the nearest iteration across all plots, or focus a graph
and use Left/Right, Home, and End. The sidebar always shows the latest iteration.

History continues collecting while the mesh is visible. Switching tabs preserves
it, and **Mesh & topology** returns to the model view. A completed or stopped run
keeps its history; a new run or valid model edit clears the old history. A flat
curve by itself is not a convergence certificate; the displayed run status reports
the solver's actual stopping reason.

The plots are rendered locally with canvas and no additional dependencies.
Only scalar history is retained; density images are not saved for every iteration.

**Elapsed solver time** uses a monotonic wall clock started before problem
creation. Each evaluated history row records cumulative `elapsed_seconds`,
including setup, the initial solve, previous callbacks, and all optimisation
work through that iteration. Process/import warm-up, browser/network delays,
and cleanup after the last sample are excluded. Timing resets for each run,
and stopping retains the last evaluated sample. The same values are saved in
`history.csv` and the final sample is included in `summary.json`. An older
running server without timing data is labelled as unavailable until restarted.

Model inputs are locked during a run; tabs remain available for inspection.
Editing the model afterward clears
the previous result so it cannot be mistaken for the updated load case.
The interface uses the same solver as the command-line example. The
**Length · ℓ** control in the Filters tab sets the density filter length
in model units (initially 0.1). Drag the slider or enter a value: the purple
circle shows the Helmholtz length ℓ as its radius
at the actual geometry scale. The readout gives ℓ, its diameter,
and its radius in x/y cell spacings. The same reference is drawn over the
live topology so it can be compared with the members. Geometry and mesh
changes update that reference
immediately. Set ℓ to 0 to disable filtering. Controls lock during a run;
changing the size afterward clears the previous result.
Mesh and load edits update the preview before the run starts. A local worker
process starts in the background with the server and streams each evaluated
outer iteration to the browser. It stays warm between runs; geometry, loads,
and optimiser state are recreated for each run. Closing the live connection
cancels that run; the server allows one active optimisation at a time.
`/api/health` reports `worker_ready` separately from HTTP server readiness and
indicates projection support. If an older server is still running after an
update, the page asks for a restart before enabling Run. Close the original
launcher window, reopen **Start TOptVis.bat**, and refresh the page.

## Offline ROM training

Open **ROM Training** and configure its independent mesh, right-edge load
heights (fractions of domain height), load angles, volume fractions, filter,
projection, iteration limit, and snapshot spacing. Comma-separated lists form
the Cartesian product of training cases. Click **Generate snapshots** to run
the existing full-order MMA/PETSc optimiser for each case. Initial states,
every selected outer iterate, and final states are captured; trial evaluations
are excluded. Loads are unit magnitude, rounded to the nearest right-edge
node. Material defaults and support conditions match the cantilever solver.

Progress shows actual full-order compliance, equilibrium residuals, and the
latest physical density. **Stop training** preserves collected snapshots as
a partial dataset. Training and cantilever optimisation share one worker and
cannot run simultaneously. Switching workspaces keeps a job running;
disconnecting its live stream cancels it. The server supports up to 24 cases,
512 snapshots, and 128 MiB of uncompressed displacement/density arrays per
dataset, rejecting larger requests before starting work.

Choose a saved dataset under **Build the reduced basis**, set the target POD
energy and maximum mode count, then click **Build reduced basis**. This is
uncentred displacement POD on free DOFs, using an economy SVD. Equal snapshot
weighting (the default) normalises each column before SVD, preventing large
displacements in soft designs from dominating. Turning it off uses the raw
snapshot magnitudes. Rank selection respects numerical rank and the mode
limit, and explicitly reports an unmet energy target when the cap is too low.
The spectrum and maximum per-snapshot relative reconstruction error describe
training displacements, not elastic strain energy or unseen-case accuracy.
Use **Validate on unseen cases** for held-out accuracy checks. Using the ROM
to drive Cantilever 2D optimisation is not enabled; its state solves remain
full order.

Datasets persist in `outputs/rom-training/<id>/`, relative to the source
checkout. Refreshing or restarting reloads the dataset library. Downloads:

- `snapshots.npz`: free-DOF displacement columns, corresponding physical
  densities, case indices, outer iteration numbers, compliances, residuals,
  free-DOF indices, mesh coordinates/connectivity, and JSON provenance.
- `basis.npz`: orthonormal free-DOF basis columns, free-DOF mapping, singular
  values, cumulative POD energy, original snapshot norms, reconstruction
  errors, and JSON provenance. Clamped displacement components are zero.
- `metadata.json`: complete training cases, settings, completion state, and
  basis diagnostics. Load NPZ files with `numpy.load(..., allow_pickle=False)`.

A dataset can be used to rebuild a basis with different settings. Cancelled
builds leave the previous saved basis intact. The numerical workflow is also
available through `ml_opt.rom` (`SnapshotConfig`, `generate_snapshots`, and
`build_basis`). HTTP endpoints are `POST /api/rom/snapshots`,
`POST /api/rom/<id>/basis`, `GET /api/rom/trainings`, and
`GET /api/rom/<id>/artifacts/{snapshots,basis,metadata}`. Generation and building
stream NDJSON events; `POST /api/run/<job-id>/stop` cancels either stage.

### Held-out ROM validation

Select a dataset with a built basis, then configure independent load heights,
directions, and volume fractions under **Validate on unseen cases**. The mesh,
material, support conditions, filter, and projection are inherited from that
single-field training dataset. Every combination is checked against all
declared training cases after snapping to mesh nodes and normalising load
directions, including the equivalence of -180 and 180 degrees. Training
overlap and duplicate physical cases are rejected before a worker job starts.

Full-order MMA generates each new design trajectory. At the initial design,
every comparison interval, and the final design, validation solves the frozen
Galerkin system `(V.T K V) a = V.T f` and lifts `u_ROM = V a`. Validation never
adds snapshots to training, updates the basis, or uses ROM gradients to drive
these trajectories. The compared quantities are:

- Displacement: `||u_ROM - u_FOM||_2 / ||u_FOM||_2` on nodal DOFs (the fixed
  components are zero). This is a Euclidean norm, not an energy norm.
- Compliance: `abs(C_ROM - C_FOM) / abs(C_FOM)`, with `C = f.T u`.
- Design gradient: `||g_ROM - g_FOM||_2 / ||g_FOM||_2` for the raw design
  variables, through the same SIMP, Helmholtz-filter and projection autograd
  pipeline. The fixed-basis self-adjoint ROM derivative is checked against
  finite differences in the tests. A zero reference with nonzero difference
  is reported as undefined and cannot pass its tolerance.

The tab reports per-design comparisons, mean and maximum errors, selected
tolerances, ROM equilibrium residuals, and the number of designs passing all
three thresholds. A completed report is within tolerances only when every
sampled design passes; a stopped report is always partial. These are empirical
checks on the sampled held-out trajectories, not guarantees for arbitrary
material distributions or unsampled intermediate designs. Validation is
limited to 16 cases, 128 comparison states and 128 MiB of comparison arrays.

Reports persist under `outputs/rom-training/<id>/validations/<validation-id>/`.
`report.json` records the complete validation cases, settings, thresholds,
errors, and SHA-256 of the exact frozen basis artifact. `comparisons.npz`
contains raw designs, physical densities, full and reduced displacements and
design gradients, DOF mapping, case/iteration indices, and JSON provenance.
Cancellation preserves collected comparisons as partial evidence; cancelling
before a comparison leaves any previous report intact. Rebuilding the basis
invalidates the current validation link; prior files remain on disk.

The API adds `POST /api/rom/<id>/validate` (streamed),
`GET /api/rom/<id>/validation` and downloads at
`GET /api/rom/<id>/artifacts/{validation,comparisons}`. Python callers can use
`ValidationConfig` and `validate_basis` from `ml_opt.rom`.

## 2D cantilever example

The example minimises compliance `C = f^T u` subject to `K(rho) u = f`,
`mean(rho) / volume_fraction - 1 <= 0`, and element density bounds `0 <= rho <= 1`.
The rectangular cantilever is clamped along its left edge and loaded downward
at the midpoint of its right edge by default. An odd number of elements through the height
splits the load equally between the two adjacent nodes.

Defaults: 60 x 20 bilinear quadrilateral elements, a 3 x 1 domain, plane stress,
unit thickness and load, Young's modulus 1, Poisson ratio 0.3, volume fraction
0.4, and SIMP penalisation 3. The material law is
`E(rho) = E_min + (E_solid - E_min) * rho**p`, with `E_min = 1e-6`.
The **Material** tab selects SIMP (`p >= 1`) or RAMP (`q >= 0`) and the
penalty value. RAMP uses
`E(rho) = E_min + (E_solid - E_min) * rho / (1 + q * (1 - rho))`;
`q = 0` and SIMP `p = 1` give linear interpolation. Both use the physical
density after filtering/projection, including all three robust fields.
See the [SIMP/RAMP material interpolation reference](https://doc.comsol.com/6.3/doc/com.comsol.help.opt/opt_ug_opt_interface.6.21.html).
Python uses `material_model="simp"` or `"ramp"` with `penalisation` for p or q;
CLI flags are `--material-model ramp --penalisation 3`. The material defaults
remain SIMP with p = 3; volume is never stiffness-penalised.
These are nondimensional benchmark parameters. The Python/CLI default
`filter_radius=0` and `projection_beta=0` retain the original unfiltered
benchmark; the visual interface initially uses `filter_radius=0.1`,
`projection_beta=2`, and `projection_eta=0.5`.

### Helmholtz density filter

For a positive `filter_radius = ℓ`, the smoothed density solves
`rho_smooth - ℓ² ∇²rho_smooth = x` with zero normal flux on every domain boundary. Here `x`
is the bounded element design variable. A cell-centred finite-volume
discretisation uses the actual x/y mesh spacings, including rectangular cells.
The formulation preserves constant densities and material volume and keeps
filtered densities within the input range, up to floating-point roundoff.
The material interpolation, volume constraint, saved density plots, and live topology frames all
use the final physical density after any projection. Saved `design` arrays remain the raw
optimiser variables, while saved `density` arrays hold the physical result.

PETSc factors the fixed filter matrix once per problem and reuses those
factors for the forward filter and its transpose adjoint. PyTorch propagates
compliance and volume gradients back through the filter to the design
variables. As with the elasticity bridge, this supports first derivatives on
CPU float64 tensors.

### Heaviside projection

The pointwise normalised smooth Heaviside projection follows Helmholtz
smoothing, using the formulation of
[Wang, Lazarov, and Sigmund (2011)](https://doi.org/10.1007/s00158-010-0602-y):

```text
rho_physical = [tanh(beta * eta) + tanh(beta * (rho_smooth - eta))]
               / [tanh(beta * eta) + tanh(beta * (1 - eta))]
```

`projection_beta >= 0` controls sharpness; 0 bypasses the projection exactly.
`0 < projection_eta < 1` sets its threshold. The UI starts at β = 2, η = 0.5;
the Python/CLI default β = 0 preserves the previous benchmark. Fixed β is
the default; eta stays fixed throughout the run.

In **Filters → β schedule**, select **Scheduled increase** to multiply β after
each full interval of accepted optimisation iterations, capped at the chosen
maximum. Defaults are an interval of 25, multiplier 2, and maximum β = 16;
the sharpness input gives the starting β and must be positive for scheduling.
For a start of 2, iterations 0–25 use β = 2, 26–50 use 4, 51–75 use 8, and
iteration 76 onward uses 16. A short iteration limit can stop before the
maximum is reached; the UI shows this when configuring the schedule.

At each increase the current design is re-evaluated at the new β before
stepping, and MMA/GCMMA asymptotes, scaling, and convergence patience restart.
Convergence stopping is enabled only at the maximum β. Robust optimisation
updates all three projections together and reinitialises its epigraph bound.
The active β appears in the live view, Visualisation summary, each history
row, and `summary.json` as `final_projection_beta`; the saved config retains
the starting β. Scheduling does not modify the Helmholtz length or penalty.

```powershell
python -m ml_opt --material-model ramp --penalisation 3 --filter-radius 0.1 `
  --projection-beta 2 --projection-schedule geometric --projection-beta-interval 25 `
  --projection-beta-multiplier 2 --projection-beta-max 16 --max-iterations 100
```

Projection is generally not volume preserving, so the volume constraint uses
`mean(rho_physical)`, not the raw or smoothed density mean. PyTorch carries
both compliance and volume sensitivities through the projection and the
Helmholtz adjoint. The live frames and saved `density` array contain exactly
the physical density used for the state solve.

ℓ is the coefficient length in the PDE. The displayed circle has radius ℓ;
for ℓ = 0.1 it shows radius 0.1 and diameter 0.2. This is a smoothing-length
reference, not a guaranteed member thickness. The Helmholtz influence extends
beyond the circle. For comparison with a classical finite-support linear
filter, its conventional approximate equivalent radius is `R ≈ 2√3 ℓ`;
this larger R is not used for the visual circle. See
[the DTU filter comparison, Eq. 17](https://www.topopt.mek.dtu.dk/-/media/subsites/topopt/apps/dokumenter-og-filer-til-apps/topopt88.pdf).
The Helmholtz PDE with homogeneous
Neumann boundary conditions follows the approach of
[Lazarov and Sigmund (2011)](https://orbit.dtu.dk/en/publications/filters-in-topology-optimization-based-on-helmholtztype-different/).

Element contributions update a preallocated PETSc sparse matrix through COO
assembly and use PETSc's sequential LU solver. PyTorch differentiates the SIMP/RAMP material law and
volume constraint. The custom compliance operator supplies the implicit,
self-adjoint derivative `dC/dE_e = -u_e^T K_e(E=1) u_e` without differentiating
through LU. It supports first derivatives on CPU float64 tensors.

### Run

After installing the project and dependencies:

```powershell
.\.venv\Scripts\python.exe -m ml_opt --optimizer mma
.\.venv\Scripts\python.exe -m ml_opt --optimizer gcmma
.\.venv\Scripts\python.exe -m ml_opt --optimizer mma --filter-radius 0.1
.\.venv\Scripts\python.exe -m ml_opt --optimizer gcmma --filter-radius 0.1 --projection-beta 4 --projection-eta 0.5
```

The equivalent console command is `toptvis-cantilever --optimizer mma` with
the environment activated. For a smaller mesh:

```powershell
.\.venv\Scripts\python.exe examples\cantilever.py --nx 30 --ny 10 --optimizer gcmma --max-iterations 100
```

The load can also be changed from the command line, for example:

```powershell
.\.venv\Scripts\python.exe -m ml_opt --load 2 --load-angle -60 --load-node-x 45 --load-node-y 12
```

Each run writes `result.npz` (design, density, mesh, displacement, force),
`history.csv`, `summary.json`, `density.png`, and `convergence.png` to
`outputs/cantilever-mma` or `outputs/cantilever-gcmma`. Override the destination
with `--output PATH`. Run `python -m ml_opt --help` for the remaining parameters.
The history includes iteration zero and records quantities at the same design.

The default move limit is 0.2 per iteration. Stopping requires stable maximum
density change, relative objective change, and volume feasibility over five
consecutive iterations after iteration 20. A run can also reach its iteration
limit; `summary.json` reports the actual stop reason and feasibility separately.
This stopping test does not certify KKT stationarity or a global optimum.

### Three-field robust formulation

Enable `robust=True` in `CantileverConfig`, or `--robust` on the command line.
One Helmholtz-filtered design is projected at three thresholds:
`eta_eroded = projection_eta + robust_eta_offset`,
`eta_nominal = projection_eta`, and
`eta_dilated = projection_eta - robust_eta_offset`. The default offset is 0.1.
All thresholds must lie strictly inside (0, 1), and robust mode requires
`projection_beta > 0`. A positive filter length is recommended for spatial
regularisation. Threshold offsets are density thresholds, not geometric
manufacturing tolerances in model units.

```text
minimise_x,t  t
subject to C_i(x) <= t, i = eroded, nominal, dilated
           mean(rho_dilated(x)) / volume_fraction - 1 <= 0
           K(rho_i(x)) u_i = f, i = eroded, nominal, dilated
           0 <= x <= 1, t >= 0
```

For the same fixed load and monotone SIMP/RAMP material law, the eroded field is
the worst compliance case. All three states are nevertheless evaluated and
reported. MMA and GCMMA solve the epigraph using mmapy's native nonnegative
auxiliary variable `z`, with `t = C_ref * z`. `C_ref` is the initial worst
compliance at the start of each β stage and stays fixed within that stage. The density-dependent objective is zero,
`a0 = 1`, and the four rows are `[C_eroded/C_ref, C_nominal/C_ref,
C_dilated/C_ref, volume_constraint]` with `a = [1, 1, 1, 0]`.
Thus `z` couples to every compliance constraint and cannot relax the volume
limit. mmapy's usual penalised feasibility slacks remain in the subproblems;
the evaluated physical volume and epigraph residual are checked separately.
All three smooth compliance gradients pass through the projection and shared
Helmholtz adjoint, with no extra elasticity solve in backward. GCMMA checks
conservativeness for every compliance row and the volume row at each trial.
The maximum compliance remains a reporting metric, and its active gradient
is available for diagnostics; it is not the gradient used to optimise the
epigraph. Thresholds and the dilated volume limit remain fixed during the run;
optional β continuation follows the schedule described above, without volume rescaling.

```powershell
# Run from this checkout, using its source when sharing another worktree's environment.
$env:PYTHONPATH = Join-Path $PWD 'src'
python -m ml_opt --robust --filter-radius 0.1 --projection-beta 4 --field-execution sequential
python -m ml_opt --robust --filter-radius 0.1 --projection-beta 4 --field-execution parallel
```

In the interface, open **Filters**, select **Three fields · robust**, and choose
**Sequential** or **Parallel · 3 processes**. The live material selector shows
each field and its compliance and volume. The headline metrics always show
worst compliance and dilated volume, whichever field is displayed. The volume
target therefore permits less nominal material than a single-field run.

`Evaluation.density`, the ordinary snapshot callback, `density.png`, and the
saved `density` array use nominal material. `Evaluation.displacement` and the
saved `displacement` use the worst field; `volume_fraction` is the dilated
volume used in the constraint. The evaluation also exposes `field_densities`,
`field_compliances`, `field_volumes`, `field_displacements`, and `worst_field`.
The optional state snapshot callback receives a matching displacement/density
pair for the worst field in robust mode, suitable for equilibrium snapshots.
Robust outputs include `density_eroded/nominal/dilated` and corresponding
`displacement_*` arrays in `result.npz`, a `robust-fields.png` comparison,
per-field history columns, and field metadata in `summary.json`. History and
summary also record `epigraph_compliance` (t) and the normalised residual
`(max(C_i) - t) / C_ref`. The stopping check requires this residual and the
volume constraint to meet the feasibility tolerance. Relative objective change
uses `t` in robust mode and compliance in single-field mode. MMA iterates may violate
the true epigraph inequality before convergence; GCMMA uses conservative trial
approximations. These checks do not certify KKT stationarity. State-solve
counts include every field and all GCMMA trial evaluations; residuals report
the maximum across fields and linear iterations sum across fields.

**Parallelisation:** sequential mode reuses one PETSc solver for the three
states. Parallel mode uses three persistent spawned processes, each owning
its own `COMM_SELF` matrix, vectors, KSP, and LU factors. This works with the
bundled Windows MPIUNI build. Only NumPy moduli, states, and element energies
cross process boundaries; the PyTorch graph stays in the parent. The filter
is solved once per evaluation in the parent. Workers limit PyTorch threads
to one to reduce oversubscription, following the
[PyTorch multiprocessing guidance](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html).
User scripts selecting parallel mode must call optimisation inside an
`if __name__ == "__main__":` guard. Prefer sequential mode for small meshes:
process startup, array transfers, and three copies of the solver consume time
and memory, so parallel mode is not an automatic speedup. Both bridges still
require CPU float64 and support first derivatives only.

Reproduce cold and warm evaluation timings, including all field gradients and
checking mode agreement, with `python benchmarks/robust_fields.py --nx 60 --ny 20`.

For larger distributed meshes, a natural next implementation is an MPI-enabled
PETSc build with three communicator groups, one per material field, and a
distributed matrix/KSP within each group. PETSc provides
[subcommunicators](https://petsc.org/release/manualpages/Sys/PetscSubcommType/)
and [parallel KSP/AMG solvers](https://petsc.org/release/manual/ksp/).
That would require distributing the mesh, assembly, filter, and gradients;
the current process option parallelises independent fields, not the DOFs of
one state solve. PyTorch DDP does not distribute this custom PETSc operator.

### Optimiser choice

**MMA is the default** for this basic, single-load compliance problem. It takes
one moving-asymptote step per outer iteration. **GCMMA** adds trial evaluations
and updates the approximation curvature until the objective and constraint
approximations are conservative. It is useful when more conservative updates
are wanted and can require more state solves. Both use the same volume
constraint, move limit, and initial-compliance scaling.

Both backends use [`mmapy` 0.3.1](https://github.com/arjendeetman/GCMMA-MMA-Python),
which is distributed under **GPL-3.0-or-later**; its implementation remains an
external dependency. The backend credits Krister Svanberg's original MMA/GCMMA
methods and Arjen Deetman's Python implementation. GCMMA raises an error if it
cannot obtain a conservative trial within the configured inner iteration limit.

For the default 60 x 20 mesh, a local run gave the following results. Both
met the documented stability criteria and retained a physical volume fraction
of 0.40000; these are local designs, not certified global optima.

| Optimiser | Initial compliance | Final compliance | Outer iterations | State solves |
| --- | ---: | ---: | ---: | ---: |
| MMA | 1841.457 | 232.349 | 89 | 90 |
| GCMMA | 1841.457 | 240.175 | 50 | 121 |

## Layout

```text
src/ml_opt/config.py          Benchmark and optimisation parameters
src/ml_opt/fem/               Mesh, Q4 integration, PETSc solve/autograd bridge
src/ml_opt/problem.py         SIMP objective, volume constraint, sensitivities
src/ml_opt/projection.py      Differentiable smooth Heaviside projection
src/ml_opt/optimizers/        MMA/GCMMA state and inner iterations
src/ml_opt/runner.py          Outer optimisation and stopping criteria
src/ml_opt/rom/               Offline snapshots, POD, and training artifacts
src/ml_opt/output.py          Result files and figures
src/ml_opt/cli.py             Command-line entry point
src/ml_opt/web/               Local HTTP server, solver worker, streaming API
src/ml_opt/web/static/        Visual interface and canvas mesh/topology views
src/ml_opt/web/static/convergence.js  Live convergence plots and iteration inspection
examples/cantilever.py        Runnable cantilever example
tests/                       Numerical and optimisation tests
scripts/                     Environment setup and validation
pyproject.toml               Package metadata and dependencies
```

## Development

The declared dependencies are NumPy, PyTorch (`torch`), PETSc's Python
bindings (`petsc4py`), SciPy, Matplotlib, and `mmapy`. SciPy provides the
optimiser backend's dense linear algebra; state solves and sparse assembly
are performed by PETSc.

Run the numerical tests with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" --no-build-isolation
.\.venv\Scripts\python.exe -m pytest -q
```

When a Git worktree shares another checkout's environment, set
`$env:PYTHONPATH = Join-Path $PWD 'src'` before running Python or tests, so
the worktree's source is used. The launcher sets this automatically.

### Local Windows environment

Verified locally on 14 November 2026: Python 3.14.4, PyTorch 2.15.0+cpu,
NumPy 2.5.4, PETSc 3.26.1, and petsc4py 3.26.1. PETSc's bundled check example,
the Python environment check, and `pip check` passed. The local package
snapshot is saved in `.venv/environment-freeze.txt`.

The setup targets 64-bit Windows Python 3.14 and Visual Studio C++ Build Tools.
Run from PowerShell:

```powershell
.\scripts\setup_windows.ps1
```

The script installs Python dependencies in `.venv`, downloads local Cygwin
build tools, and compiles PETSc 3.26.1 with MSVC. Cygwin root registration and
shortcuts are disabled. PETSc uses static libraries, the MSVC DLL runtime,
real double precision, 32-bit indices, and MPIUNI for single-process execution.
BLAS/LAPACK is built from PETSc's `f2cblaslapack` package. PyTorch uses the CPU
build. Distributed MPI and GPU support require a separate configuration.

`stage_petsc_windows.py` copies the compiled headers and libraries into
`.venv/petsc` and converts the linker metadata for native Windows Python.
The original Cygwin build configuration is retained. The setup then builds
`petsc4py`, installs this project in editable mode, and checks the environment.

Activate the environment, or run its interpreter directly:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe scripts\check_environment.py
```

The check exercises PyTorch autograd and a PETSc conjugate-gradient solve.
Build logs are kept in `.venv/build-native-petsc.log` and
`.venv/build-petsc4py.log`. Source files and compiler configuration are retained
in `.venv/build/petsc-3.26.1`. After a successful PETSc build, use
`setup_windows.ps1 -SkipPetscBuild` to restage PETSc and rebuild the bindings.

The native build follows the [PETSc Windows installation guide](https://petsc.org/release/install/windows/)
and [petsc4py source installation instructions](https://petsc.org/release/petsc4py/install.html).

### Other environments

Build or install real double-precision PETSc and `petsc4py` for your platform,
then install this package and its remaining dependencies:

```sh
python -m pip install -e .
```

Generated results belong in `outputs/` and model checkpoints in `checkpoints/`;
both directories are excluded from Git.
