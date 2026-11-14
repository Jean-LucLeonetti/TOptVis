"""Projection limits, physical volume, and the complete filtered adjoint."""

import math

import numpy as np
import pytest
import torch

from ml_opt.config import CantileverConfig
from ml_opt.problem import CantileverProblem
from ml_opt.projection import heaviside_projection
from ml_opt.runner import optimise


def test_zero_beta_is_exact_identity():
    density = torch.linspace(0, 1, 11, dtype=torch.float64, requires_grad=True)
    assert heaviside_projection(density, 0, .4) is density
    np.testing.assert_array_equal(torch.autograd.grad(density.sum(), density)[0], 1)


@pytest.mark.parametrize("beta,eta", [(2, .5), (8, .35), (64, .7)])
def test_projection_bounds_endpoints_and_analytic_derivative(beta, eta):
    x = np.linspace(0, 1, 41)
    density = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    projected = heaviside_projection(density, beta, eta)
    result = projected.detach().numpy()
    denominator = math.tanh(beta*eta) + math.tanh(beta*(1-eta))
    expected = (math.tanh(beta*eta) + np.tanh(beta*(x-eta))) / denominator
    np.testing.assert_allclose(result, expected, atol=2e-16)
    np.testing.assert_allclose(result[[0, -1]], [0, 1], atol=1e-15)
    assert result.min() >= -1e-15 and result.max() <= 1+1e-15
    assert np.all(np.diff(result) >= -1e-15)
    derivative = torch.autograd.grad(projected.sum(), density)[0].numpy()
    np.testing.assert_allclose(derivative, beta*(1-np.tanh(beta*(x-eta))**2)/denominator,
                               atol=2e-14, rtol=1e-10)


def test_larger_beta_sharpens_and_threshold_changes_material_fraction():
    density = torch.tensor([.2, .4, .6, .8], dtype=torch.float64)
    soft = heaviside_projection(density, 2, .5)
    sharp = heaviside_projection(density, 8, .5)
    assert torch.sum(sharp*(1-sharp)) < torch.sum(soft*(1-soft))
    assert torch.all(heaviside_projection(density, 4, .3) > heaviside_projection(density, 4, .7))


@pytest.mark.parametrize("radius", [0, .15])
def test_projected_compliance_and_physical_volume_gradients(radius):
    config = CantileverConfig(nx=6, ny=3, filter_radius=radius,
                             projection_beta=4, projection_eta=.37)
    with CantileverProblem(config) as problem:
        design = np.random.default_rng(19).uniform(.25, .75, 18)
        evaluation = problem.evaluate(design)
        filtered = problem.density_filter.solve(design)
        expected = heaviside_projection(torch.tensor(filtered), 4, .37).numpy()
        np.testing.assert_allclose(evaluation.density, expected, atol=1e-14)
        np.testing.assert_allclose(evaluation.volume_fraction, expected.mean(), atol=1e-14)
        assert abs(evaluation.volume_fraction-design.mean()) > .02
        assert np.ptp(evaluation.constraint_gradient) > .01
        for index in (0, 5, 8, 17):
            perturbation = np.zeros(18)
            perturbation[index] = 1e-5
            plus = problem.evaluate(design+perturbation, False)
            minus = problem.evaluate(design-perturbation, False)
            np.testing.assert_allclose(evaluation.objective_gradient[index],
                (plus.compliance-minus.compliance)/2e-5, rtol=3e-5)
            np.testing.assert_allclose(evaluation.constraint_gradient[index],
                (plus.constraint-minus.constraint)/2e-5, rtol=1e-7)


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_optimizers_use_projected_material_and_respect_physical_volume(method):
    config = CantileverConfig(nx=18, ny=6, filter_radius=.08, projection_beta=4,
                             optimizer=method, max_iterations=40)
    frames = []
    result = optimise(config, snapshot_callback=lambda row, density: frames.append(density))
    assert result.evaluation.compliance < .8*result.history[0].compliance
    assert result.evaluation.constraint <= config.feasibility_tolerance
    assert result.evaluation.residual < 1e-7
    assert np.linalg.norm(result.evaluation.density-result.design) > .01
    np.testing.assert_array_equal(frames[-1], result.evaluation.density)
    np.testing.assert_allclose(result.evaluation.volume_fraction, result.evaluation.density.mean())


@pytest.mark.parametrize("options", [{"projection_beta": -1}, {"projection_beta": float("inf")},
    {"projection_eta": 0}, {"projection_eta": 1}, {"projection_eta": float("nan")}])
def test_invalid_projection_configuration(options):
    with pytest.raises(ValueError, match="finite|projection"):
        CantileverConfig(**options)
