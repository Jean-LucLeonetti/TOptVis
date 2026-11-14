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
If TOptVis is already running on port 8000, the launcher opens that instance.

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
preserves the cantilever settings, results, and any active run. ROM Training
is a dedicated workspace for the planned offline training workflow; training
and model generation are not available yet. The workspaces can be opened
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
3. In **Filters**, set the Helmholtz length and the Heaviside projection's
   sharpness β and threshold η. Larger β sharpens the solid/void transition;
   β = 0 disables projection. These values remain fixed during a run.
4. In **Optimisation**, choose MMA or GCMMA, the material volume, and the iteration limit. Click
   **Run optimisation** to see the topology, compliance, volume, and iteration
   values update as the actual solver runs. **Stop** retains the last evaluated
   design.

### Convergence visualisation

Select **Visualisation** on the far-left rail to see five live graphs:
compliance, physical material volume, maximum design change, relative
compliance change, and cumulative elapsed solver time versus outer iteration.
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
SIMP, the volume constraint, saved density plots, and live topology frames all
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
the Python/CLI default β = 0 preserves the previous benchmark. Beta and eta
are fixed throughout a run, with no automatic continuation.

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
assembly and use PETSc's sequential LU solver. PyTorch differentiates the SIMP material law and
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
density change, relative compliance change, and volume feasibility over five
consecutive iterations after iteration 20. A run can also reach its iteration
limit; `summary.json` reports the actual stop reason and feasibility separately.
This stopping test does not certify KKT stationarity or a global optimum.

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
