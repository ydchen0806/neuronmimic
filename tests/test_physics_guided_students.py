from __future__ import annotations

import numpy as np
import torch

from neuronmimic.physics_guided import CableTeacherConfig, GraphDynamics, MLPDynamics, PhysicsResidualDynamics


def _chain(n: int) -> torch.Tensor:
    a = torch.zeros(n, n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def test_students_have_common_step_and_rollout_contract():
    cfg = CableTeacherConfig(n_compartments=5, duration_ms=2.0)
    adjacency = _chain(cfg.n_compartments)
    models = [MLPDynamics(cfg.n_compartments), GraphDynamics(adjacency), PhysicsResidualDynamics(adjacency, cfg)]
    state = torch.zeros(2, 2 * cfg.n_compartments)
    event = torch.zeros(2, cfg.n_compartments)
    for model in models:
        one = model.step(state, event)
        rollout = model.rollout(state[:1], torch.zeros(3, cfg.n_compartments))
        assert one.shape == state.shape
        assert rollout.shape == (4, 1, 2 * cfg.n_compartments)
        assert torch.isfinite(rollout).all()


def test_graph_uses_morphology_buffer():
    model = GraphDynamics(_chain(5))
    assert "adjacency" in dict(model.named_buffers())
    assert model.adjacency.shape == (5, 5)


def test_physics_residual_is_finite_for_longer_rollout():
    cfg = CableTeacherConfig(n_compartments=8)
    model = PhysicsResidualDynamics(_chain(cfg.n_compartments), cfg)
    output = model.rollout(torch.zeros(1, 2 * cfg.n_compartments), torch.ones(20, cfg.n_compartments) * 0.01)
    assert np.isfinite(output.detach().numpy()).all()
