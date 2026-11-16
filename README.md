# TOptVis

An interactive demonstrator for topology optimisation, reduced models, and
machine learning. Explore structural designs, compare methods, and inspect
results as experiments run.

## Capabilities

- Topology optimisation with editable geometry, supports, loads, and non-design regions.
- SIMP and RAMP materials, density filtering, projection, and robust formulations.
- Volume and maximum feature-size constraints with MMA and GCMMA optimisation.
- Live topology and convergence views, with portable project files.
- Snapshot generation, POD reduced models, and held-out validation.
- Linear and quadratic regression experiments using optimisation results.

The numerical backend uses PETSc for finite-element solves and PyTorch for
sensitivities. Method references are available in the interface and the
[detailed guide](docs/GUIDE.md).

## Getting started

Requires Python 3.11 or newer and real, double-precision PETSc with `petsc4py`
configured for your platform. See the [PETSc installation guide](https://petsc.org/release/install/)
for prerequisites.

### Local macOS environment

From the project folder, create an environment and install the interface:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[ui]"
.venv/bin/python -m ml_opt.web --open-browser
```

After setup, double-click **Start TOptVis.command** to launch the app.

### Windows

Run the setup script from PowerShell:

```powershell
.\scripts\setup_windows.ps1
```

Then double-click **Start TOptVis.bat**. See the
[Windows setup details](docs/GUIDE.md#local-windows-environment) for build prerequisites.

### Command line

With the environment activated:

```sh
python -m ml_opt --optimizer mma
python -m ml_opt --help
```

## Documentation

- [Usage and numerical methods](docs/GUIDE.md)
- [Regression experiments](docs/GUIDE.md#regression-experiments)
- [Reduced-model training and validation](docs/GUIDE.md#offline-rom-training)
- [Development and tests](docs/GUIDE.md#development)
- [Solver benchmarks](benchmarks/SOLVER_RESULTS.md)

## License

Original TOptVis code is available under the [MIT License](LICENSE), except
where a file specifies otherwise. The MMA/GCMMA implementation includes
GPL-3.0-or-later components; distribution of the combined application must
comply with the GPL. See [third-party licensing](THIRD_PARTY_NOTICES.md).
