"""Exact-state reuse, bounded worker lifetime, and independent solver accuracy."""

from dataclasses import replace
import json
import multiprocessing as mp
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml_opt.config import CantileverConfig
from ml_opt.fem.cache import StateCache
from ml_opt.fem.worker_pool import FieldWorkerPool
from ml_opt.problem import CantileverProblem
from ml_opt.runner import optimise
from ml_opt.web.app import create_app


def config(**options):
    return replace(CantileverConfig(nx=6, ny=3, robust=True, projection_beta=4, filter_radius=.12), **options)


@pytest.mark.parametrize("robust", [False, True])
def test_value_to_gradient_reuses_states_without_aliasing_and_invalidates_on_moduli(robust):
    settings = config(robust=robust)
    design = np.linspace(.25, .75, 18)
    count = 3 if robust else 1
    with CantileverProblem(settings) as problem:
        trial = problem.evaluate(design, False)
        expected = trial.compliance
        # Public result mutations must not alter private cached states.
        trial.displacement[:] = 100
        if robust:
            for state in problem.fields.last_states:
                state.displacement[:] = 100
                state.element_energy[:] = 100
        else:
            problem.solver.last_state.displacement[:] = 100
            problem.solver.last_state.element_energy[:] = 100
        current = problem.evaluate(design)
        assert problem.solve_count == count
        assert current.compliance == expected
        with CantileverProblem(settings) as reference:
            result = reference.evaluate(design)
        np.testing.assert_array_equal(current.displacement, result.displacement)
        np.testing.assert_array_equal(current.objective_gradient, result.objective_gradient)
        problem.projection_beta = 8
        problem.evaluate(design)
        assert problem.solve_count == 2 * count
        problem.config = replace(settings, material_model="ramp", penalisation=2)
        problem.evaluate(design)
        assert problem.solve_count == 3 * count
        perturbed = design.copy()
        perturbed[0] += 1e-10
        problem.evaluate(perturbed)
        assert problem.solve_count == 4 * count


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("material", ["simp", "ramp"])
@pytest.mark.parametrize("schedule", ["fixed", "geometric"])
def test_cached_gcmma_matches_uncached_trajectory_with_material_and_beta_changes(monkeypatch, robust, material, schedule):
    settings = config(robust=robust, material_model=material, optimizer="gcmma", max_iterations=8,
        projection_schedule=schedule, projection_beta_interval=3, projection_beta_max=8)
    cached = optimise(settings)
    with monkeypatch.context() as patch:
        patch.setattr(StateCache, "get", lambda self, moduli: None)
        baseline = optimise(settings)
    np.testing.assert_array_equal(cached.design, baseline.design)
    np.testing.assert_array_equal(cached.evaluation.objective_gradient, baseline.evaluation.objective_gradient)
    assert cached.evaluation.compliance == baseline.evaluation.compliance
    assert baseline.history[-1].state_solves - cached.history[-1].state_solves == 8 * (3 if robust else 1)


def test_pool_reuses_only_compatible_workers_and_resets_problem_caches_and_counts():
    first = config(field_execution="parallel")
    second = replace(first, projection_beta=8, material_model="ramp", penalisation=2, volume_fraction=.5)
    with FieldWorkerPool() as pool:
        with CantileverProblem(first, field_pool=pool) as problem:
            problem.evaluate(problem.initial_design())
            pids = [p.pid for p in problem.fields._workers]
        with CantileverProblem(second, field_pool=pool) as problem:
            result = problem.evaluate(problem.initial_design())
            assert problem.solve_count == 3
            assert [p.pid for p in problem.fields._workers] == pids
        with CantileverProblem(replace(second, field_execution="sequential")) as reference:
            expected = reference.evaluate(reference.initial_design())
        np.testing.assert_array_equal(result.objective_gradient, expected.objective_gradient)
        with CantileverProblem(replace(second, load_angle=-45), field_pool=pool) as problem:
            problem.evaluate(problem.initial_design())
            assert not set(pids) & {p.pid for p in problem.fields._workers}
            assert not set(pids) & {p.pid for p in mp.active_children()}


def test_pool_rejects_a_second_active_problem_and_survives_callback_failure():
    settings = config(field_execution="parallel", max_iterations=1)
    with FieldWorkerPool() as pool:
        with CantileverProblem(settings, field_pool=pool) as problem:
            with pytest.raises(RuntimeError, match="already in use"):
                CantileverProblem(settings, field_pool=pool)
            assert problem.evaluate(problem.initial_design()).residual < 1e-7
            pids = [worker.pid for worker in problem.fields._workers]
        def failed_callback(row):
            raise RuntimeError("callback failed")
        with pytest.raises(RuntimeError, match="callback failed"):
            optimise(settings, callback=failed_callback, field_pool=pool)
        assert [worker.pid for worker in pool._group.workers] == pids
        assert optimise(settings, field_pool=pool).evaluation.residual < 1e-7


@pytest.mark.parametrize("change", [{"nx": 8}, {"ny": 4}, {"length": 4}, {"height": 2},
    {"poisson_ratio": .2}, {"thickness": 2}, {"load": 2}, {"load_angle": -45},
    {"load_node_x": 3, "load_node_y": 1}, {"linear_tolerance": 1e-12}, {"linear_solver": "cholesky"},
    {"state_solver": "rom", "rom_basis_path": "model.npz"}])
def test_pool_compatibility_key_covers_state_solver_inputs(change):
    base = config()
    assert FieldWorkerPool.configuration_key(base) != FieldWorkerPool.configuration_key(replace(base, **change))


def test_pool_expires_idle_workers_and_does_not_evict_an_active_lease():
    with FieldWorkerPool() as pool:
        with CantileverProblem(config(field_execution="parallel"), field_pool=pool) as problem:
            problem.evaluate(problem.initial_design())
            pids = [p.pid for p in problem.fields._workers]
            pool._last_used = time.monotonic() - 61
            pool.evict_idle()
            assert pool._group.alive()
        pool._last_used = time.monotonic() - 61
        pool.evict_idle()
        assert pool._group is None
        assert not set(pids) & {p.pid for p in mp.active_children()}


def test_pool_recovers_after_native_worker_failure():
    settings = config(field_execution="parallel")
    with FieldWorkerPool() as pool:
        with CantileverProblem(settings, field_pool=pool) as problem:
            problem.evaluate(problem.initial_design())
            old_pids = [p.pid for p in problem.fields._workers]
            with pytest.raises(RuntimeError, match="failed|exited"):
                problem.fields.solve(np.zeros((3, 18)))
        assert pool._group is None
        with CantileverProblem(settings, field_pool=pool) as problem:
            assert problem.evaluate(problem.initial_design()).residual < 1e-7
            assert not set(old_pids) & {p.pid for p in problem.fields._workers}


def test_pool_recovers_after_worker_process_crash():
    settings = config(field_execution="parallel")
    with FieldWorkerPool() as pool:
        with CantileverProblem(settings, field_pool=pool) as problem:
            problem.evaluate(problem.initial_design())
        pool._group.workers[0].terminate()
        pool._group.workers[0].join(timeout=5)
        with CantileverProblem(settings, field_pool=pool) as problem:
            assert problem.evaluate(problem.initial_design()).residual < 1e-7


@pytest.mark.parametrize("method", ["cholesky", "cg-icc", "cg-gamg"])
def test_solver_compliance_and_gradients_match_lu_and_finite_differences(method):
    settings = config(robust=False, linear_solver=method)
    design = np.random.default_rng(32).uniform(.25, .75, 18)
    with CantileverProblem(replace(settings, linear_solver="lu")) as reference:
        expected = reference.evaluate(design)
    with CantileverProblem(settings) as problem:
        result = problem.evaluate(design)
        assert result.residual < 1e-8
        np.testing.assert_allclose(result.compliance, expected.compliance, rtol=1e-8)
        np.testing.assert_allclose(result.objective_gradient, expected.objective_gradient, rtol=2e-7)
        for index in (0, 8, 17):
            step = np.zeros(18)
            step[index] = 1e-5
            plus = problem.evaluate(design + step, False)
            minus = problem.evaluate(design - step, False)
            np.testing.assert_allclose(result.objective_gradient[index], (plus.compliance-minus.compliance)/2e-5, rtol=5e-5)


@pytest.mark.parametrize("method", ["cholesky", "cg-icc", "cg-gamg"])
def test_high_contrast_state_matches_lu(method):
    settings = config(nx=12, ny=4, linear_solver=method)
    moduli = np.random.default_rng(5).choice([1e-6, 1.], 48)
    with CantileverProblem(replace(settings, linear_solver="lu")) as reference:
        expected = reference.solver.solve(moduli)
    with CantileverProblem(settings) as problem:
        actual = problem.solver.solve(moduli)
    np.testing.assert_allclose(actual.compliance, expected.compliance, rtol=1e-7)
    assert np.linalg.norm(actual.element_energy-expected.element_energy)/np.linalg.norm(expected.element_energy) < 1e-6
    assert actual.relative_residual < 1e-7


@pytest.mark.parametrize("method", ["cholesky", "cg-icc", "cg-gamg"])
def test_alternative_solvers_match_in_parallel_fields(method):
    settings = config(linear_solver=method)
    design = np.linspace(.25, .75, 18)
    with CantileverProblem(settings) as problem:
        expected = problem.evaluate(design)
    with CantileverProblem(replace(settings, field_execution="parallel")) as problem:
        actual = problem.evaluate(design)
    for name in ("eroded", "nominal", "dilated"):
        np.testing.assert_allclose(actual.field_compliances[name], expected.field_compliances[name], rtol=1e-8)
        np.testing.assert_allclose(actual.field_gradients[name], expected.field_gradients[name], rtol=1e-7)


def test_invalid_linear_solver_is_rejected():
    with pytest.raises(ValueError, match="linear_solver"):
        config(linear_solver="invalid")


def test_app_selects_alternative_solvers_and_reuses_parallel_run_service():
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json()["linear_solver_supported"]
        assert client.post("/api/run", json={"linear_solver": "unknown"}).status_code == 422
        for method, execution in (("cholesky", "parallel"), ("cholesky", "parallel"),
                                  ("cg-icc", "sequential"), ("cg-gamg", "sequential")):
            response = client.post("/api/run", json={"nx": 6, "ny": 3, "load_node_x": 6,
                "load_node_y": 1, "projection_beta": 4, "robust": True,
                "max_iterations": 2, "optimizer": "gcmma", "field_execution": execution,
                "linear_solver": method})
            assert response.status_code == 200
            events = [json.loads(line) for line in response.text.splitlines()]
            assert events[0]["config"]["linear_solver"] == method
            assert events[-1]["event"] == "complete", events[-1]
            frames = [event for event in events if event["event"] == "iteration"]
            assert [event["history"]["iteration"] for event in frames] == [0, 1, 2]
            assert max(event["history"]["linear_residual"] for event in frames) < 1e-7
            if method == "cholesky":
                assert frames[0]["history"]["linear_iterations"] == 3
            else:
                assert frames[0]["history"]["linear_iterations"] > 3
