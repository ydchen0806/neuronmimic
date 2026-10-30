"""Deterministic passive cable Teacher used when the real Arbor Teacher is blocked.

The implementation is deliberately small and explicit. It is a fixture for
testing the data contract and Student methods, not a replacement for Arbor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CableTeacherConfig:
    n_compartments: int = 12
    dt_ms: float = 0.05
    duration_ms: float = 80.0
    tau_mem_ms: float = 12.0
    tau_syn_ms: float = 4.0
    leak_conductance: float = 1.0
    axial_conductance: float = 0.35
    capacitance: float = 1.0
    resting_voltage: float = -65.0
    synapse_reversal: float = 0.0
    synapse_increment: float = 0.8
    spike_threshold: float = -45.0


@dataclass(frozen=True)
class Stimulus:
    name: str
    synapse_compartments: tuple[int, ...]
    onset_ms: tuple[float, ...]
    amplitude: tuple[float, ...]
    split: str
    family: str

    def __post_init__(self) -> None:
        if not (len(self.synapse_compartments) == len(self.onset_ms) == len(self.amplitude)):
            raise ValueError("stimulus fields must have equal lengths")
        if len(set(self.synapse_compartments)) != len(self.synapse_compartments):
            raise ValueError("synapse compartments must be unique")


def simulate_teacher(config: CableTeacherConfig, stimulus: Stimulus, *, adjacency: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Run the deterministic passive cable Teacher and return full states."""
    n = int(config.n_compartments)
    if any(int(i) < 0 or int(i) >= n for i in stimulus.synapse_compartments):
        raise ValueError("stimulus compartment outside teacher morphology")
    steps = int(round(config.duration_ms / config.dt_ms))
    if steps <= 0:
        raise ValueError("duration must produce at least one step")
    if adjacency is None:
        adjacency = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.shape != (n, n) or not np.allclose(adjacency, adjacency.T):
        raise ValueError("Teacher adjacency must be a symmetric morphology matrix")
    v = np.full(n, config.resting_voltage, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    voltage = np.zeros((steps + 1, n), dtype=np.float64)
    conductance = np.zeros_like(voltage)
    # Events are inputs applied during each Euler step, hence there are
    # exactly ``steps`` rows.  Voltage/conductance contain the initial state
    # plus one row per update (``steps + 1`` rows).
    event = np.zeros((steps, n), dtype=np.float64)
    voltage[0] = v
    onsets = np.asarray(stimulus.onset_ms, dtype=float)
    for t in range(steps):
        now = t * config.dt_ms
        g *= np.exp(-config.dt_ms / config.tau_syn_ms)
        for compartment, onset, amplitude in zip(stimulus.synapse_compartments, onsets, stimulus.amplitude):
            if abs(now - onset) <= 0.5 * config.dt_ms:
                g[int(compartment)] += config.synapse_increment * float(amplitude)
                event[t, int(compartment)] += float(amplitude)
        axial = config.axial_conductance * (adjacency @ v - adjacency.sum(axis=1) * v)
        # ``tau_mem_ms`` defines the passive leak scale when capacitance is
        # expressed in the same fixture units; ``leak_conductance`` is a
        # dimensionless calibration multiplier.
        effective_leak = config.leak_conductance * config.capacitance / config.tau_mem_ms
        leak = effective_leak * (config.resting_voltage - v)
        syn = g * (config.synapse_reversal - v)
        v = v + config.dt_ms / config.capacitance * (leak + axial + syn)
        voltage[t + 1] = v
        conductance[t + 1] = g
    return {"voltage": voltage, "conductance": conductance, "event": event, "time_ms": np.arange(steps + 1) * config.dt_ms}


def make_stimulus_panel(config: CableTeacherConfig, *, seed: int = 20261030) -> list[Stimulus]:
    """Create fixed train/validation/test stimulus families.

    Held-out rows use unseen compartment combinations rather than merely new
    random seeds, which tests spatial compositional generalization.
    """
    n = int(config.n_compartments)
    rng = np.random.default_rng(seed)
    panel: list[Stimulus] = []
    panel.append(Stimulus("single_02", (2,), (10.0,), (1.0,), "train", "single"))
    panel.append(Stimulus("single_07", (7,), (18.0,), (0.8,), "train", "single"))
    panel.append(Stimulus("double_01_05", (1, 5), (12.0, 22.0), (1.0, 0.7), "train", "double"))
    panel.append(Stimulus("double_03_09", (3, 9), (14.0, 25.0), (0.7, 1.0), "train", "double"))
    panel.append(Stimulus("sparse_00_04_08", (0, 4, 8), (10.0, 17.0, 28.0), (0.8, 0.5, 0.7), "train", "sparse_multi"))
    panel.append(Stimulus("jitter_02_07", (2, 7), (float(rng.choice([11.0, 13.0])), float(rng.choice([21.0, 24.0]))), (0.9, 0.6), "train", "jittered"))
    panel.append(Stimulus("double_02_10_unseen", (2, min(10, n - 1)), (16.0, 31.0), (1.0, 0.9), "test", "unseen_combo"))
    panel.append(Stimulus("sparse_01_06_11_unseen", (1, min(6, n - 1), n - 1), (9.0, 20.0, 34.0), (0.6, 1.0, 0.8), "test", "unseen_combo"))
    panel.append(Stimulus("错峰_05_10", (5, min(10, n - 1)), (11.0, 37.0), (1.0, 0.5), "test", "unseen_timing"))
    return panel


def config_dict(config: CableTeacherConfig) -> dict[str, float | int]:
    return asdict(config)
