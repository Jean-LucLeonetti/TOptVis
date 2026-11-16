"""Manual boundary conditions reach streamed solves and reject invalid layouts."""

from dataclasses import asdict
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from ml_opt.boundaries import BoundaryCondition, MeshSelection
from ml_opt.config import CantileverConfig
from ml_opt.problem import CantileverProblem
from ml_opt.web.app import RunRequest, create_app


CONDITIONS = [
    {"kind": "fixed", "selection": {"kind": "line", "start": [0, 0], "end": [1, 0]}},
    {"kind": "force", "selection": {"kind": "line", "start": [.25, 1], "end": [.75, 1]},
     "magnitude": 2, "angle": -90},
]


def test_api_uses_selected_supports_and_distributed_force():
    with TestClient(create_app()) as client:
        assert client.get('/api/health').json()['boundary_conditions_supported']
        response = client.post('/api/run', json={'nx': 8, 'ny': 4, 'max_iterations': 2,
                                               'boundary_conditions': CONDITIONS})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]['event'] == 'complete', events[-1]
        assert len(events[0]['config']['boundary_conditions']) == 2
        row = [event for event in events if event['event'] == 'iteration'][0]
        reference_config = RunRequest(nx=8, ny=4, max_iterations=2, boundary_conditions=CONDITIONS).configuration()
        with CantileverProblem(reference_config) as problem:
            expected = problem.evaluate(problem.initial_design())
            np.testing.assert_array_equal(problem.solver.mesh.fixed_dofs, np.arange(18))
            np.testing.assert_allclose(problem.solver.force.reshape(-1, 2).sum(axis=0), [0, -2])
            np.testing.assert_allclose(row['history']['compliance'], expected.compliance, rtol=1e-10)
            np.testing.assert_array_equal(expected.displacement[problem.solver.mesh.fixed_dofs], 0)


@pytest.mark.parametrize('conditions', [
    [], CONDITIONS[1:], CONDITIONS[:1],
    [{'kind': 'fixed', 'selection': {'kind': 'surface', 'start': [0, 0], 'end': [1, 1]}}, CONDITIONS[1]],
    [CONDITIONS[0], {'kind': 'force', 'selection': {'kind': 'line', 'start': [0, 0], 'end': [1, 1]}}],
    [CONDITIONS[0], {'kind': 'force', 'selection': {'kind': 'point', 'start': [-.1, 1]}}],
])
def test_request_rejects_unstable_empty_and_invalid_selections(conditions):
    with pytest.raises(ValueError):
        RunRequest(nx=8, ny=4, boundary_conditions=conditions)


def test_custom_conditions_survive_worker_serialisation():
    config = RunRequest(nx=8, ny=4, boundary_conditions=CONDITIONS).configuration()
    assert CantileverConfig(**json.loads(json.dumps(asdict(config)))) == config


def test_parallel_worker_key_changes_when_boundaries_change():
    from ml_opt.fem.worker_pool import FieldWorkerPool
    config = CantileverConfig(nx=8, ny=4, boundary_conditions=CONDITIONS)
    alternative = CantileverConfig(nx=8, ny=4, boundary_conditions=(
        BoundaryCondition('fixed', MeshSelection('line', (0, 0), (0, 1))),
        BoundaryCondition('force', MeshSelection('point', (1, 1))),
    ))
    assert FieldWorkerPool.configuration_key(config) != FieldWorkerPool.configuration_key(alternative)
