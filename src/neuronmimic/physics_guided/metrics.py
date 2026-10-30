"""Evaluation metrics for Teacher/Student voltage and graded responses."""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch


def latency_ms(trace: np.ndarray, time_ms: np.ndarray, threshold: float, *, readout_compartment: int | None = None) -> float:
    """Latency of a graded response at a declared readout.

    No spike is inferred here.  If ``readout_compartment`` is omitted, the
    earliest threshold crossing over compartments is used as a descriptive
    aggregate and is reported as such by the caller.
    """
    signal = trace[:, int(readout_compartment)] if readout_compartment is not None else np.max(trace, axis=-1)
    idx = np.flatnonzero(signal > threshold)
    return float(time_ms[idx[0]]) if len(idx) else float("nan")


def evaluate_student(model, examples: Iterable[dict[str, np.ndarray]], *, threshold: float = -45.0, readout_compartment: int | None = None) -> dict[str, float]:
    one_step, multi, latency, finite = [], [], [], []
    elapsed = 0.0
    for ex in examples:
        state = torch.from_numpy(ex["state"]).float()
        event = torch.from_numpy(ex["event"]).float()
        with torch.no_grad():
            start = time.perf_counter()
            pred = model.rollout(state[0:1], event)
            elapsed += time.perf_counter() - start
        target = state
        one_step.append(float(torch.mean((pred[1] - target[1]) ** 2).sqrt()))
        multi.append(float(torch.mean((pred - target) ** 2).sqrt()))
        t = ex["time_ms"]
        pred_voltage = pred[:, 0, : model.n].numpy()
        target_voltage = target[:, : model.n].numpy()
        pred_latency = latency_ms(pred_voltage, t, threshold, readout_compartment=readout_compartment)
        target_latency = latency_ms(target_voltage, t, threshold, readout_compartment=readout_compartment)
        latency.append(abs(pred_latency - target_latency) if np.isfinite(target_latency) and np.isfinite(pred_latency) else np.nan)
        finite.append(bool(torch.isfinite(pred).all()))
    return {"one_step_voltage_rmse": float(np.nanmean(one_step)), "multi_step_voltage_rmse": float(np.nanmean(multi)), "latency_error_ms": float(np.nanmean(latency)) if np.isfinite(latency).any() else float("nan"), "finite_fraction": float(np.mean(finite)), "wall_clock_s": float(elapsed)}
