from __future__ import annotations

import numpy as np

from neuronmimic.physics_guided.teacher import CableTeacherConfig, make_stimulus_panel, simulate_teacher


def _chain(n: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        out[i, i + 1] = out[i + 1, i] = 1.0
    return out


def test_teacher_is_deterministic_and_has_aligned_event_contract():
    cfg = CableTeacherConfig()
    stimulus = make_stimulus_panel(cfg)[-1]
    adjacency = _chain(cfg.n_compartments)
    a = simulate_teacher(cfg, stimulus, adjacency=adjacency)
    b = simulate_teacher(cfg, stimulus, adjacency=adjacency)
    assert a["voltage"].shape == (int(cfg.duration_ms / cfg.dt_ms) + 1, cfg.n_compartments)
    assert a["event"].shape == (int(cfg.duration_ms / cfg.dt_ms), cfg.n_compartments)
    assert np.array_equal(a["voltage"], b["voltage"])
    assert np.array_equal(a["event"], b["event"])
    assert np.allclose(adjacency, adjacency.T)


def test_panel_has_explicit_unseen_combination_and_timing_splits():
    panel = make_stimulus_panel(CableTeacherConfig())
    train = {x.name for x in panel if x.split == "train"}
    test = {x.name for x in panel if x.split == "test"}
    assert train and test and train.isdisjoint(test)
    assert any(x.family == "unseen_combo" for x in panel if x.split == "test")
    assert any(x.family == "unseen_timing" for x in panel if x.split == "test")
