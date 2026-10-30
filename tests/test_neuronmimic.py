from __future__ import annotations

import torch

from neuronmimic import NeuronMimic, NeuronMimicConfig, calibrate


def _model(rank: int = 0) -> NeuronMimic:
    return NeuronMimic(3, torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0.7, 0.5]), NeuronMimicConfig(surrogate_beta=12.0, residual_rank=rank))


def test_directed_rollout_and_schema():
    m = _model()
    x = torch.zeros(30, 3)
    x[2:15, 0] = 7.0
    out = m(x)
    assert out["voltage_mv"].shape == (30, 1, 3)
    assert out["spike_soft"].shape == (30, 1, 3)
    assert torch.isfinite(out["voltage_mv"]).all()
    assert float(out["spike_soft"][:, :, 0].sum().detach()) >= float(out["spike_soft"][:, :, 2].sum().detach())


def test_calibration_reduces_training_loss_and_keeps_parameters_bounded():
    truth = _model(rank=1)
    x = torch.zeros(40, 3)
    x[3:24, 0] = 8.0
    with torch.no_grad():
        target = truth(x)["voltage_mv"].detach()
    fitted = _model(rank=1)
    result = calibrate(fitted, x, target_voltage=target, steps=30, learning_rate=1e-2, seed=4)
    assert result.train_loss_final < result.train_loss_initial
    p = fitted.effective_parameters()
    assert float(p["tau_mem_ms"].min().detach()) > 0
    assert float(p["tau_syn_ms"].min().detach()) > 0
    assert float((p["threshold_mv"] + 45.0).abs().max().detach()) <= 2.0 + 1e-5


def test_same_seed_replay_is_exact():
    x = torch.zeros(12, 3)
    x[1:8, 0] = 6.0
    a, b = _model(), _model()
    ra = calibrate(a, x, target_spike=torch.zeros(12, 1, 3), steps=5, seed=9)
    rb = calibrate(b, x, target_spike=torch.zeros(12, 1, 3), steps=5, seed=9)
    assert ra.train_loss_final == rb.train_loss_final
    assert torch.equal(a.log_tau_mem_delta, b.log_tau_mem_delta)
