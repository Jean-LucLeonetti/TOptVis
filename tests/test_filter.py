"""Helmholtz modes, conservation, adjoints, and filtered optimisation."""

from dataclasses import replace

import numpy as np
from petsc4py import PETSc
import pytest

from ml_opt.config import CantileverConfig
from ml_opt.fem.filter import HelmholtzFilter
from ml_opt.fem.mesh import StructuredMesh
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise


def test_helmholtz_response_to_known_neumann_eigenmode_on_rectangular_cells():
    mesh = StructuredMesh(7, 4, 2.3, .6)
    radius = .17
    kx, ky = 2, 1
    x = np.cos(kx * np.pi * (np.arange(mesh.nx) + .5) / mesh.nx)
    y = np.cos(ky * np.pi * (np.arange(mesh.ny) + .5) / mesh.ny)
    mode = (y[:, None] * x[None, :]).ravel()
    eigenvalue = 4 * (np.sin(kx*np.pi/(2*mesh.nx))**2 / (mesh.length/mesh.nx)**2
                     + np.sin(ky*np.pi/(2*mesh.ny))**2 / (mesh.height/mesh.ny)**2)
    with HelmholtzFilter(mesh, radius) as density_filter:
        np.testing.assert_allclose(density_filter.solve(mode), mode/(1 + radius**2*eigenvalue),
                                   atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("radius", [0, .1, .5])
def test_filter_preserves_constants_mean_and_density_bounds(radius):
    mesh = StructuredMesh(9, 5, 3, 1)
    design = np.random.default_rng(7).uniform(0, 1, mesh.element_count)
    with HelmholtzFilter(mesh, radius) as density_filter:
        np.testing.assert_allclose(density_filter.solve(np.ones(mesh.element_count)), 1, atol=1e-13)
        physical = density_filter.solve(design)
        np.testing.assert_allclose(physical.mean(), design.mean(), atol=1e-13)
        assert physical.min() >= design.min() - 1e-13
        assert physical.max() <= design.max() + 1e-13
        if radius == 0:
            np.testing.assert_array_equal(physical, design)
        else:
            assert np.var(physical) < np.var(design)


def test_size_uses_physical_units_and_larger_size_smooths_more():
    mesh = StructuredMesh(12, 6, 3, 1)
    design = (np.indices((mesh.ny, mesh.nx)).sum(axis=0) % 2).ravel().astype(float)
    with HelmholtzFilter(mesh, .05) as small, HelmholtzFilter(mesh, .2) as large:
        smoothed = large.solve(design)
        assert np.var(smoothed) < np.var(small.solve(design))
    with HelmholtzFilter(replace(mesh, length=6, height=2), .4) as scaled:
        np.testing.assert_allclose(scaled.solve(design), smoothed, rtol=1e-13, atol=1e-14)


def test_filtered_compliance_and_volume_gradients_against_central_differences():
    config = CantileverConfig(nx=6, ny=3, filter_radius=.18)
    with CantileverProblem(config) as problem:
        design = np.random.default_rng(4).uniform(.3, .8, problem.mesh.element_count)
        evaluation = problem.evaluate(design)
        assert not np.allclose(evaluation.density, design)
        np.testing.assert_allclose(evaluation.volume_fraction, design.mean(), atol=1e-13)
        np.testing.assert_allclose(evaluation.constraint_gradient, 1/(design.size*config.volume_fraction),
                                   rtol=1e-12)
        for index in (0, 5, 8, 17):
            perturbation = np.zeros_like(design)
            perturbation[index] = 1e-5
            plus = problem.evaluate(design + perturbation, False)
            minus = problem.evaluate(design - perturbation, False)
            np.testing.assert_allclose(evaluation.objective_gradient[index],
                (plus.compliance-minus.compliance)/2e-5, rtol=2e-5)
            np.testing.assert_allclose(evaluation.constraint_gradient[index],
                (plus.constraint-minus.constraint)/2e-5, rtol=1e-8)


def test_filter_reuses_numeric_factorization_for_forward_and_adjoint_rhs():
    PETSc.Log.begin()
    events = {name: PETSc.Log.Event(name) for name in ("MatLUFactorSym", "MatLUFactorNum")}
    before = {name: event.getPerfInfo()["count"] for name, event in events.items()}
    with HelmholtzFilter(StructuredMesh(6, 3, 3, 1), .1) as density_filter:
        for seed in range(3):
            density_filter.solve(np.random.default_rng(seed).normal(size=18))
        assert density_filter.solve_count == 3
    for name, event in events.items():
        assert event.getPerfInfo()["count"] - before[name] == 1
    density_filter.close()
    with pytest.raises(RuntimeError, match="closed"):
        density_filter.solve(np.ones(18))


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_optimizers_use_and_stream_physical_filtered_densities(method):
    frames = []
    result = optimise(CantileverConfig(nx=18, ny=6, filter_radius=.08,
                       optimizer=method, max_iterations=35),
                       snapshot_callback=lambda entry, density: frames.append(density))
    assert result.evaluation.compliance < .8 * result.history[0].compliance
    assert result.evaluation.constraint <= result.config.feasibility_tolerance
    assert result.evaluation.residual < 1e-7
    assert np.linalg.norm(result.evaluation.density-result.design) > .01
    assert result.evaluation.density.min() >= -1e-12 and result.evaluation.density.max() <= 1+1e-12
    np.testing.assert_allclose(result.evaluation.volume_fraction, result.design.mean(), atol=1e-12)
    np.testing.assert_array_equal(frames[-1], result.evaluation.density)


@pytest.mark.parametrize("radius", [-.1, float("inf"), float("nan")])
def test_invalid_filter_size(radius):
    with pytest.raises(ValueError, match="finite|nonnegative"):
        CantileverConfig(filter_radius=radius)
    with pytest.raises(ValueError, match="finite|nonnegative"):
        HelmholtzFilter(StructuredMesh(4, 2, 3, 1), radius)
