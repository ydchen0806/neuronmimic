"""Build the fixture Teacher dataset, train three Students, evaluate and search.

All outputs are explicitly FIXTURE_TEST_ONLY until an Arbor-backed Teacher is
available and its data provenance is registered.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuronmimic.physics_guided import CableTeacherConfig, GraphDynamics, MLPDynamics, PhysicsResidualDynamics, make_stimulus_panel, simulate_teacher

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "wu_20261030_physics_guided_baseline"
TEACHER = OUT / "teacher"
RESULTS = OUT / "results"
SEED = 20261030


def _configure_determinism() -> None:
    """Make the small CPU baseline replayable; timing remains non-deterministic."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _adjacency(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def build_teacher_dataset(config: CableTeacherConfig) -> list[dict[str, np.ndarray | str]]:
    TEACHER.mkdir(parents=True, exist_ok=True)
    adj = _adjacency(config.n_compartments)
    panel = make_stimulus_panel(config, seed=SEED)
    examples = []
    rows = []
    packed = {}
    for idx, stimulus in enumerate(panel):
        trace = simulate_teacher(config, stimulus, adjacency=adj)
        state = np.concatenate([trace["voltage"], trace["conductance"]], axis=1).astype(np.float32)
        # event is one row per Euler step; state[1:] is the corresponding target.
        examples.append({"name": stimulus.name, "split": stimulus.split, "family": stimulus.family, "state": state, "event": trace["event"].astype(np.float32), "time_ms": trace["time_ms"].astype(np.float32)})
        packed[f"{stimulus.name}__state"] = state
        packed[f"{stimulus.name}__event"] = trace["event"].astype(np.float32)
        packed[f"{stimulus.name}__time_ms"] = trace["time_ms"].astype(np.float32)
        rows.append({"index": idx, "name": stimulus.name, "split": stimulus.split, "family": stimulus.family, "synapse_compartments": ";".join(map(str, stimulus.synapse_compartments)), "onset_ms": ";".join(map(str, stimulus.onset_ms)), "amplitude": ";".join(map(str, stimulus.amplitude)), "n_steps": len(trace["event"])})
    np.savez_compressed(TEACHER / "teacher_dataset.npz", **packed)
    pd.DataFrame(rows).to_csv(TEACHER / "stimulus_manifest.csv", index=False)
    (TEACHER / "teacher_protocol.json").write_text(json.dumps({"status": "FIXTURE_TEST_ONLY", "seed": SEED, "config_source": "simulation/configs/physics_guided_baseline_fixture.yaml", "config": config.__dict__, "morphology": "12-node symmetric chain fixture", "event_semantics": "conductance increment at declared compartment and onset", "train_stimuli": [x["name"] for x in rows if x["split"] == "train"], "heldout_stimuli": [x["name"] for x in rows if x["split"] == "test"]}, indent=2, ensure_ascii=False))
    return examples


def _fit(model: torch.nn.Module, examples: list[dict], *, epochs: int = 120, seed: int = SEED) -> dict[str, float | int]:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    train = [x for x in examples if x["split"] == "train"]
    initial = None
    final = None
    start = time.perf_counter()
    for epoch in range(epochs):
        total = torch.zeros(())
        optimizer.zero_grad(set_to_none=True)
        for ex in train:
            state = torch.from_numpy(ex["state"]).float()
            event = torch.from_numpy(ex["event"]).float()
            pred = model.step(state[:-1], event)
            loss = torch.mean((pred - state[1:]) ** 2)
            total = total + loss
        total = total / max(1, len(train))
        if initial is None: initial = float(total.detach())
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final = float(total.detach())
    device = str(next(model.parameters()).device)
    peak_mb = float(torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else 0.0
    return {"epochs": epochs, "train_loss_initial": initial, "train_loss_final": final, "wall_clock_s": time.perf_counter() - start, "parameter_count": sum(int(p.numel()) for p in model.parameters()), "gpu_peak_memory_mb": peak_mb, "teacher_equivalent_queries": len(train), "device": device}


def _eval(model: torch.nn.Module, examples: list[dict], config: CableTeacherConfig) -> dict[str, float]:
    from neuronmimic.physics_guided.metrics import evaluate_student
    return evaluate_student(model, [x for x in examples if x["split"] == "test"], threshold=-64.5, readout_compartment=config.n_compartments - 1)


def _student_models(config: CableTeacherConfig) -> dict[str, torch.nn.Module]:
    # Seed model construction as well as optimization so the full baseline is
    # replayable across processes.
    torch.manual_seed(SEED)
    adj = torch.from_numpy(_adjacency(config.n_compartments)).float()
    return {"MLP": MLPDynamics(config.n_compartments), "Graph": GraphDynamics(adj), "PhysicsResidual": PhysicsResidualDynamics(adj, config)}


def gradient_check(config: CableTeacherConfig, model: torch.nn.Module) -> dict[str, float | str]:
    """Compare Student event gradient with a finite-difference Teacher slope."""
    stimulus = make_stimulus_panel(config, seed=SEED)[-1]
    target_step = int(round(35.0 / config.dt_ms))
    base_trace = simulate_teacher(config, stimulus, adjacency=_adjacency(config.n_compartments))
    base = base_trace["voltage"][target_step, -1]
    eps = 1e-3
    fd_values = []
    for j in range(len(stimulus.amplitude)):
        plus_amp = list(stimulus.amplitude); minus_amp = list(stimulus.amplitude)
        plus_amp[j] += eps; minus_amp[j] -= eps
        plus = stimulus.__class__(stimulus.name, stimulus.synapse_compartments, stimulus.onset_ms, tuple(plus_amp), stimulus.split, stimulus.family)
        minus = stimulus.__class__(stimulus.name, stimulus.synapse_compartments, stimulus.onset_ms, tuple(minus_amp), stimulus.split, stimulus.family)
        fd_values.append((simulate_teacher(config, plus, adjacency=_adjacency(config.n_compartments))["voltage"][target_step, -1] - simulate_teacher(config, minus, adjacency=_adjacency(config.n_compartments))["voltage"][target_step, -1]) / (2 * eps))
    fd = float(fd_values[0])
    state = torch.from_numpy(np.concatenate([base_trace["voltage"], base_trace["conductance"]], axis=1)).float()
    event = torch.from_numpy(base_trace["event"].astype(np.float32)).float()
    event.requires_grad_(True)
    pred = model.rollout(state[0:1], event)
    objective = pred[target_step, 0, model.n - 1]
    objective.backward()
    grad_values = [float(event.grad[:, int(site)].sum().detach()) for site in stimulus.synapse_compartments]
    grad = float(grad_values[0])
    fd_vec = np.asarray(fd_values, dtype=float)
    grad_vec = np.asarray(grad_values, dtype=float)
    denom = float(np.linalg.norm(fd_vec) * np.linalg.norm(grad_vec))
    cosine = float(np.dot(fd_vec, grad_vec) / denom) if denom > 0 else float("nan")
    return {"model": type(model).__name__, "teacher_finite_difference": fd, "student_gradient": grad, "gradient_cosine": cosine, "same_sign": bool(np.sign(fd) == np.sign(grad) or abs(fd) < 1e-9), "base_target": float(base), "gradient_time_ms": target_step * config.dt_ms, "readout_compartment": model.n - 1}


def active_search(config: CableTeacherConfig, model: torch.nn.Module, *, k: int = 2, seed: int = SEED) -> pd.DataFrame:
    """Compare random, structural and relaxed-gradient Top-K proposals in Teacher."""
    rng = np.random.default_rng(seed)
    n = config.n_compartments
    target = make_stimulus_panel(config, seed=seed)[-1]
    target_step = int(round(25.0 / config.dt_ms))
    target_value = float(simulate_teacher(config, target, adjacency=_adjacency(n))["voltage"][target_step, -1])
    candidates = []
    # Single-site structural heuristic, scored by Teacher single-site response.
    site_scores = []
    for site in range(n):
        s = target.__class__(f"single_{site}", (site,), (16.0,), (1.0,), "search", "single")
        value = float(simulate_teacher(config, s, adjacency=_adjacency(n))["voltage"][target_step, -1])
        site_scores.append(value)
    heuristic_sites = tuple(np.argsort(site_scores)[-k:].tolist())
    proposals = [("heuristic", heuristic_sites)]
    for i in range(25): proposals.append(("random", tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))))
    # Gradient relaxation is proposal generation only; every proposal below is
    # evaluated by the Teacher, never accepted from Student output alone.
    logits = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    for _ in range(40):
        weights = torch.softmax(logits, dim=0)
        event = torch.zeros(int(round(config.duration_ms / config.dt_ms)), n)
        event[round(16.0 / config.dt_ms)] = weights
        initial = torch.full((1, 2 * n), config.resting_voltage)
        initial[:, n:] = 0.0
        pred = model.rollout(initial, event)
        loss = -pred[-1, 0, -1] + 0.01 * (weights.square().sum())
        opt.zero_grad(); loss.backward(); opt.step()
    grad_sites = tuple(torch.topk(logits.detach(), k).indices.sort().values.tolist())
    proposals.append(("gradient_topk", grad_sites))
    for method, sites in proposals:
        s = target.__class__(f"search_{method}_{sites}", tuple(map(int, sites)), (16.0,) * k, (1.0,) * k, "search", method)
        value = float(simulate_teacher(config, s, adjacency=_adjacency(n))["voltage"][target_step, -1])
        candidates.append({"method": method, "sites": ";".join(map(str, sites)), "k": k, "teacher_target_voltage": target_value, "teacher_candidate_voltage": value, "teacher_abs_error_to_target": abs(value - target_value)})
    return pd.DataFrame(candidates)


def run() -> dict:
    _configure_determinism()
    OUT.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    config = CableTeacherConfig()
    examples = build_teacher_dataset(config)
    models = _student_models(config)
    rows = []
    gradient_rows = []
    active = None
    for name, model in models.items():
        fit = _fit(model, examples)
        metrics = _eval(model, examples, config)
        rows.append({"model": name, **fit, **metrics, "test_examples": sum(x["split"] == "test" for x in examples), "status": "FIXTURE_TEST_ONLY"})
        gradient_rows.append(gradient_check(config, model))
        if name == "PhysicsResidual": active = active_search(config, model)
    pd.DataFrame(rows).to_csv(RESULTS / "student_comparison.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(RESULTS / "gradient_check.csv", index=False)
    active.to_csv(RESULTS / "active_search_teacher_rechecked.csv", index=False)
    comparison = pd.DataFrame(rows)
    stable_cols = [c for c in comparison.columns if c not in {"wall_clock_s"}]
    stable_csv = comparison[stable_cols].to_csv(index=False, lineterminator="\n")
    manifest = {"status": "FIXTURE_TEST_ONLY", "seed": SEED, "teacher": "deterministic passive cable fixture", "student_training": "same six train stimuli, 120 epochs, Adam lr=0.002, same parameter budget record", "outputs": [str(p.relative_to(ROOT)) for p in [TEACHER / "teacher_dataset.npz", RESULTS / "student_comparison.csv", RESULTS / "gradient_check.csv", RESULTS / "active_search_teacher_rechecked.csv"]], "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "stable_metric_columns": stable_cols, "stable_metric_sha256": hashlib.sha256(stable_csv.encode()).hexdigest(), "timing_note": "wall_clock_s is intentionally excluded from stable metric hash"}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    run()
