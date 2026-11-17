"""A crossing-field minimax verifies MMA's epigraph independently of elasticity."""

from dataclasses import replace

import numpy as np
import pytest

from ml_opt.optimizers import MovingAsymptotes
from ml_opt.problem import Evaluation


def crossing_fields(design, sensitivities=True):
    x = float(design[0])
    values = {"eroded": (x - .2)**2 + .1, "nominal": (x - .8)**2 + .1,
              "dilated": (x - .5)**2 + .08}
    gradients = {"eroded": np.array([2 * (x - .2)]), "nominal": np.array([2 * (x - .8)]),
                 "dilated": np.array([2 * (x - .5)])} if sensitivities else None
    return Evaluation(max(values.values()), 0., -1., design.copy(), np.zeros(1), 0., 0,
        # Deliberately unusable max gradient: the epigraph must use every field Jacobian.
        objective_gradient=None, constraint_gradient=np.zeros(1) if sensitivities else None,
        field_compliances=values, field_gradients=gradients)


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_epigraph_converges_at_crossing_with_two_active_fields(method):
    design = np.array([.3])
    optimizer = MovingAsymptotes(design, method, robust=True)
    for _ in range(40):
        step = optimizer.step(design, crossing_fields(design), crossing_fields)
        assert np.max(np.abs(step.design - design)) <= .2 + 1e-12
        design = step.design
    # min max((x-.2)^2+.1, (x-.8)^2+.1) occurs at x=.5, t=.19.
    np.testing.assert_allclose(design, [.5], atol=2e-4)
    np.testing.assert_allclose(step.epigraph_compliance, .19, atol=2e-4)
    assert max(crossing_fields(design).field_compliances.values()) <= step.epigraph_compliance + 1e-5


def test_gcmma_checks_an_inactive_epigraph_row_before_accepting():
    design = np.array([.3])
    optimizer = MovingAsymptotes(design, "gcmma", max_inner_iterations=1, robust=True)
    trials=[]
    def bad_trial(design, sensitivities):
        trials.append(design.copy())
        trial = crossing_fields(design, sensitivities)
        # Keep the scalar reporting objective unchanged to detect a max-only check.
        return replace(trial, field_compliances={**trial.field_compliances, "dilated": 1e20})
    step=optimizer.step(design, crossing_fields(design), bad_trial)
    assert step.inner_limit_reached and step.inner_iterations==1
    assert len(trials)==2
    np.testing.assert_array_equal(step.design,trials[-1])
    assert np.isfinite(step.epigraph_compliance)
