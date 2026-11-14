"""Deterministic timing boundaries, reset between runs, and cancellation."""

from dataclasses import replace

import numpy as np

from ml_opt.config import CantileverConfig
import ml_opt.runner as runner


def test_elapsed_time_includes_setup_resets_and_keeps_last_cancelled_sample(monkeypatch):
    clock = [100.0]
    original = runner.CantileverProblem

    class TimedProblem(original):
        def __init__(self, config):
            super().__init__(config)
            clock[0] += .3

        def evaluate(self, *args, **kwargs):
            evaluation = super().evaluate(*args, **kwargs)
            clock[0] += .2
            return evaluation

    monkeypatch.setattr(runner, "CantileverProblem", TimedProblem)
    monkeypatch.setattr(runner, "perf_counter", lambda: clock[0])
    samples = []

    def callback(entry):
        samples.append(entry.elapsed_seconds)
        clock[0] += .1  # Earlier callbacks count towards elapsed wall time.

    config = CantileverConfig(nx=4, ny=2, max_iterations=2)
    result = runner.optimise(config, callback=callback)
    np.testing.assert_allclose(samples, [.5, .8, 1.1], atol=1e-12)
    np.testing.assert_allclose([row.elapsed_seconds for row in result.history], samples)

    clock[0] += 60
    samples.clear()
    cancelled = runner.optimise(replace(config, max_iterations=10), callback=callback,
                                stop_requested=lambda: len(samples) >= 2)
    assert cancelled.stop_reason == "cancelled"
    assert cancelled.history[-1].iteration == 1
    np.testing.assert_allclose(samples, [.5, .8], atol=1e-12)
    np.testing.assert_allclose(cancelled.history[-1].elapsed_seconds, .8, atol=1e-12)
