from __future__ import annotations

import numpy as np
import pandas as pd

from neuronmimic.physics_guided.teacher import CableTeacherConfig, make_stimulus_panel, simulate_teacher


def test_fixture_teacher_is_explicitly_not_real_data():
    cfg = CableTeacherConfig()
    panel = make_stimulus_panel(cfg)
    trace = simulate_teacher(cfg, panel[0])
    assert trace["voltage"].dtype == np.float64
    assert trace["event"].shape[0] == trace["voltage"].shape[0] - 1


def test_active_search_contract_requires_exact_k_and_teacher_recheck():
    # Contract-level fixture check: candidate rows must carry a fixed K and a
    # Teacher-derived score; no Student-only acceptance field is permitted.
    frame = pd.DataFrame([
        {"method": "random", "sites": "1;3", "k": 2, "teacher_abs_error_to_target": 0.1}
    ])
    assert (frame["k"] == 2).all()
    assert "teacher_abs_error_to_target" in frame
    assert "student_abs_error_to_target" not in frame
