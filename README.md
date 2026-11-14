# TOptVis

A Python project for topology optimisation with PyTorch and PETSc through
`petsc4py`.

## 2D cantilever example

The example minimises compliance `C = f^T u` subject to `K(rho) u = f`,
`mean(rho) / volume_fraction - 1 <= 0`, and element density bounds `0 <= rho <= 1`.
The rectangular cantilever is clamped along its left edge and loaded downward
at the midpoint of its right edge. An odd number of elements through the height
splits the load equally between the two adjacent nodes.

Defaults: 60 x 20 bilinear quadrilateral elements, a 3 x 1 domain, plane stress,
unit thickness and load, Young's modulus 1, Poisson ratio 0.3, volume fraction
0.4, and SIMP penalisation 3. The material law is
`E(rho) = E_min + (E_solid - E_min) * rho**p`, with `E_min = 1e-6`.
**No density filter is applied.** These are nondimensional benchmark parameters.

Element contributions are assembled into sparse CSR format and passed to
PETSc's sequential LU solver. PyTorch differentiates the SIMP material law and
volume constraint. The custom compliance operator supplies the implicit,
self-adjoint derivative `dC/dE_e = -u_e^T K_e(E=1) u_e` without differentiating
through LU. It supports first derivatives on CPU float64 tensors.

### Run

After installing the project and dependencies:

```powershell
.\.venv\Scripts\python.exe -m ml_opt --optimizer mma
.\.venv\Scripts\python.exe -m ml_opt --optimizer gcmma
```

The equivalent console command is `toptvis-cantilever --optimizer mma` with
the environment activated. For a smaller mesh:

```powershell
.\.venv\Scripts\python.exe examples\cantilever.py --nx 30 --ny 10 --optimizer gcmma --max-iterations 100
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
src/ml_opt/optimizers/        MMA/GCMMA state and inner iterations
src/ml_opt/runner.py          Outer optimisation and stopping criteria
src/ml_opt/output.py          Result files and figures
src/ml_opt/cli.py             Command-line entry point
examples/cantilever.py        Runnable cantilever example
tests/                       Numerical and optimisation tests
scripts/                     Environment setup and validation
pyproject.toml               Package metadata and dependencies
```

## Development

The declared dependencies are NumPy, PyTorch (`torch`), PETSc's Python
bindings (`petsc4py`), SciPy, Matplotlib, and `mmapy`. SciPy provides sparse
assembly utilities and the optimiser backend's linear algebra; state solves
are performed by PETSc.

Run the numerical tests with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" --no-build-isolation
.\.venv\Scripts\python.exe -m pytest -q
```

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
