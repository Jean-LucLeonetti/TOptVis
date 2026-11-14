"""Compare every MMA/GCMMA output against the pinned upstream routines."""

import mmapy.mma as upstream
import numpy as np
import pytest

from ml_opt.config import CantileverConfig
import ml_opt.optimizers.moving_asymptotes as methods
from ml_opt.runner import optimise


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_broadcast_arithmetic_matches_upstream_during_optimization(monkeypatch, method):
    name = "mmasub" if method == "mma" else "gcmmasub"
    fast, original = getattr(methods, name), getattr(upstream, name)
    original_diags = upstream.diags
    calls = []
    def compared(*args, **kwargs):
        def copy(value):
            return value.copy() if isinstance(value, np.ndarray) else value
        expected = original(*tuple(map(copy, args)), **{key: copy(value) for key, value in kwargs.items()})
        actual = fast(*args, **kwargs)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(right, left)
        calls.append(1)
        return actual
    monkeypatch.setattr(methods, name, compared)
    optimise(CantileverConfig(nx=12, ny=4, optimizer=method, max_iterations=20))
    assert len(calls) >= 20
    assert upstream.diags is original_diags
    assert upstream.mmasub.__globals__["subsolv"] is upstream.subsolv
