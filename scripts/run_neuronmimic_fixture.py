"""Run the reproducible NeuronMimic fixture calibration/evaluation.

Outputs are explicitly FIXTURE_TEST_ONLY and must never enter a scientific
figure or be labelled as FlyWire dynamics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from neuronmimic import NeuronMimic, NeuronMimicConfig, calibrate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "neuronmimic_fixture_v1"


def run() -> dict:
    torch.manual_seed(4)
    cfg = NeuronMimicConfig(surrogate_beta=12.0, residual_rank=1)
    src = torch.tensor([0, 1], dtype=torch.long)
    dst = torch.tensor([1, 2], dtype=torch.long)
    weights = torch.tensor([0.7, 0.5])
    truth = NeuronMimic(3, src, dst, weights, cfg)
    with torch.no_grad():
        truth.log_tau_mem_delta[:] = torch.tensor([0.25, -0.12, 0.08])
        truth.log_tau_syn_delta[:] = torch.tensor([-0.10, 0.15, 0.05])
        truth.threshold_delta[:] = torch.tensor([0.10, -0.08, 0.04])
    train = torch.zeros(50, 3)
    train[4:30, 0] = 8.0
    train[12:35, 1] = 2.0
    heldout = torch.zeros(50, 3)
    heldout[5:22, 0] = 5.5
    heldout[28:44, 2] = 3.0
    with torch.no_grad():
        train_target = truth(train)["voltage_mv"].detach()
        heldout_target = truth(heldout)["voltage_mv"].detach()
    fitted = NeuronMimic(3, src, dst, weights, cfg)
    with torch.no_grad():
        before = fitted(heldout)["voltage_mv"]
    result = calibrate(fitted, train, target_voltage=train_target, steps=80, learning_rate=1e-2, seed=4)
    with torch.no_grad():
        after_train = fitted(train)["voltage_mv"]
        after = fitted(heldout)["voltage_mv"]
    rows = [
        {"split": "train", "stage": "before", "voltage_mse": float(torch.mean((fitted.baseline_state(1)[0][None] - train_target) ** 2))},
        {"split": "train", "stage": "after", "voltage_mse": float(torch.mean((after_train - train_target) ** 2))},
        {"split": "heldout", "stage": "before", "voltage_mse": float(torch.mean((before - heldout_target) ** 2))},
        {"split": "heldout", "stage": "after", "voltage_mse": float(torch.mean((after - heldout_target) ** 2))},
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    protocol = {
        "name": "neuronmimic_fixture_v1",
        "status": "FIXTURE_TEST_ONLY",
        "seed": 4,
        "n_neurons": 3,
        "directed_edges": [[0, 1, 0.7], [1, 2, 0.5]],
        "train_steps": len(train),
        "heldout_steps": len(heldout),
        "calibration_steps": result.steps,
        "model": "directed soft-surrogate LIF + bounded per-neuron calibration + rank-1 residual",
        "biology_claim": "none",
        "source_file_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    protocol["calibration"] = result.__dict__
    (OUT / "provenance.json").write_text(json.dumps(protocol, indent=2))
    return {"status": "PASS", "metrics": rows, "output": str(OUT)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
