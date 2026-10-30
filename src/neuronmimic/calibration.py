"""Held-out trajectory calibration for :mod:`neuronmimic`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .model import NeuronMimic


@dataclass
class CalibrationResult:
    train_loss_initial: float
    train_loss_final: float
    steps: int
    seed: int
    parameter_summary: dict[str, float]


def calibrate(model: NeuronMimic, external_current: Tensor, *, target_voltage: Optional[Tensor] = None, target_spike: Optional[Tensor] = None, steps: int = 200, learning_rate: float = 5e-3, seed: int = 0, voltage_weight: float = 1.0, spike_weight: float = 1.0) -> CalibrationResult:
    """Fit declared calibration parameters on one training trajectory only.

    Evaluation must use untouched drives or perturbations supplied by the
    caller; this helper intentionally never selects parameters on test data.
    """
    if target_voltage is None and target_spike is None:
        raise ValueError("target_voltage or target_spike is required")
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial = final = float("nan")
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(external_current)
        loss = model.prior_penalty()
        if target_voltage is not None:
            loss = loss + voltage_weight * torch.mean((out["voltage_mv"] - target_voltage) ** 2)
        if target_spike is not None:
            loss = loss + spike_weight * torch.mean((out["spike_soft"] - target_spike.to(out["spike_soft"].dtype)) ** 2)
        if step == 0:
            initial = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        final = float(loss.detach())
    params = model.effective_parameters()
    summary = {name + "_mean": float(value.detach().mean()) for name, value in params.items()}
    return CalibrationResult(initial, final, steps, seed, summary)
