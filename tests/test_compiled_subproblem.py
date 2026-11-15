"""Compiled Newton-loop parity and KKT residuals, including both Schur branches."""

import numpy as np
import pytest

from ml_opt.optimizers import _mmapy
from ml_opt.optimizers._subproblem import subsolv
from ml_opt.config import CantileverConfig
from ml_opt.runner import optimise


@pytest.mark.parametrize("m,n", [(2, 40), (3, 40), (6, 40), (7, 4)])
def test_compiled_subproblem_matches_all_primal_dual_variables_and_kkt(m, n):
    rng = np.random.default_rng(m+n)
    low, upp = np.zeros((n, 1)), np.ones((n, 1))
    alfa, beta = .1*np.ones((n, 1)), .9*np.ones((n, 1))
    p0, q0 = rng.uniform(.05, .2, (n, 1)), rng.uniform(.05, .2, (n, 1))
    P, Q = rng.uniform(.001, .01, (m, n)), rng.uniform(.001, .01, (m, n))
    a, b, c, d = np.zeros((m, 1)), .5*np.ones((m, 1)), 1000*np.ones((m, 1)), np.ones((m, 1))
    if m >= 6:
        a[:3] = 1
    args = m, n, 1e-7, low, upp, alfa, beta, p0, q0, P, Q, 1.0, a, b, c, d
    reference, _ = _mmapy._bind_functions(compiled=False)
    expected = reference.__globals__["subsolv"](*args)
    actual = subsolv(*args)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-9)
    x, y, z, lam, xsi, eta, mu, zet, slack = actual
    residuals = [
        (p0+P.T@lam)/(upp-x)**2-(q0+Q.T@lam)/(x-low)**2-xsi+eta,
        c+d*y-mu-lam, 1-zet-a.T@lam,
        P@(1/(upp-x))+Q@(1/(x-low))-a*z-y+slack-b,
        xsi*(x-alfa), eta*(beta-x), mu*y, zet*z, lam*slack]
    assert max(float(np.max(np.abs(row))) for row in residuals) < 2e-6
    assert np.all(x > alfa) and np.all(x < beta)
    assert all(np.all(value > 0) for value in (y, z, lam, xsi, eta, mu, zet, slack))


@pytest.mark.parametrize("method", ["mma", "gcmma"])
@pytest.mark.parametrize("robust", [False, True])
def test_compiled_geometry_trajectory_matches_baseline(monkeypatch, method, robust):
    import ml_opt.optimizers.moving_asymptotes as methods

    c = CantileverConfig(nx=12, ny=4, projection_beta=4, filter_radius=.1,
        max_solid_size=.7, max_void_size=1.2, robust=robust, optimizer=method,
        max_iterations=15, min_iterations=16)
    actual = optimise(c)
    baseline_mma, baseline_gcmma = _mmapy._bind_functions(compiled=False)
    monkeypatch.setattr(methods, "mmasub", baseline_mma)
    monkeypatch.setattr(methods, "gcmmasub", baseline_gcmma)
    expected = optimise(c)
    np.testing.assert_allclose(actual.design, expected.design, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(actual.evaluation.compliance, expected.evaluation.compliance, rtol=1e-8)
    np.testing.assert_allclose(actual.evaluation.max_void_constraint, expected.evaluation.max_void_constraint,
                               rtol=1e-6, atol=1e-7)


def test_missing_optional_runtime_keeps_reference_solver_usable(monkeypatch):
    monkeypatch.setattr(_mmapy, "_compiled_kernel", lambda: None)
    result = optimise(CantileverConfig(nx=8, ny=4, length=2,
        max_solid_size=.8, max_void_size=1.0, max_iterations=2))
    assert len(result.history) == 3
