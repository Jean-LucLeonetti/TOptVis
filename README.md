# TOptVis

A Python project for topology optimisation with PyTorch and PETSc through
`petsc4py`.

## Initial scope

This repository contains the package scaffold. The finite element model,
PyTorch/PETSc interface, objective functions, and optimisation loop are still
to be implemented.

## Layout

```text
src/ml_opt/       Python package
scripts/          Environment setup and validation
pyproject.toml   Package metadata and dependencies
```

## Development

The declared dependencies are NumPy, PyTorch (`torch`), and PETSc's Python
bindings (`petsc4py`).

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

Build or install PETSc and `petsc4py` for your platform, install PyTorch and
NumPy, then install this package:

```sh
python -m pip install --no-deps -e .
```

Generated results belong in `outputs/` and model checkpoints in `checkpoints/`;
both directories are excluded from Git.
