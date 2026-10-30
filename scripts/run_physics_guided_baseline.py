"""Build the fixture Teacher dataset, train three Students, evaluate and search.

All outputs are explicitly FIXTURE_TEST_ONLY until an Arbor-backed Teacher is
available and its data provenance is registered.
"""
from __future__ import annotations

import argparse
import dataclasses
import subprocess
import yaml
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuronmimic.physics_guided.experiments import gradient_check, active_search
from neuronmimic.physics_guided import CableTeacherConfig, GraphDynamics, MLPDynamics, PhysicsResidualDynamics, PhysicsOnlyDynamics, make_stimulus_panel, simulate_teacher

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/physics_guided_p0_v2"
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
    (TEACHER / "teacher_protocol.json").write_text(json.dumps({"status": "FIXTURE_TEST_ONLY", "seed": SEED, "config_source": "config_snapshot.yaml", "config": config.__dict__, "morphology": "12-node symmetric chain fixture", "event_semantics": "conductance increment at declared compartment and onset", "train_stimuli": [x["name"] for x in rows if x["split"] == "train"], "heldout_stimuli": [x["name"] for x in rows if x["split"] == "test"]}, indent=2, ensure_ascii=False))
    return examples


def _fit(model: torch.nn.Module, examples: list[dict], *, epochs: int = 120, seed: int = SEED, learning_rate: float = .002) -> dict[str, float | int]:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
    return {"epochs": epochs, "train_loss_initial": initial, "train_loss_final": final, "training_wall_clock_s": time.perf_counter() - start, "parameter_count": sum(int(p.numel()) for p in model.parameters()), "gpu_peak_memory_mb": peak_mb, "teacher_training_trajectories_shared": len(train), "device": device}


def _eval(model: torch.nn.Module, examples: list[dict], config: CableTeacherConfig, *, threshold=-64.5, readout=None) -> dict[str, float]:
    from neuronmimic.physics_guided.metrics import evaluate_student
    return evaluate_student(model, [x for x in examples if x["split"] == "test"], threshold=threshold, readout_compartment=config.n_compartments - 1 if readout is None else readout)


def _student_models(config: CableTeacherConfig) -> dict[str, torch.nn.Module]:
    # Seed model construction as well as optimization so the full baseline is
    # replayable across processes.
    torch.manual_seed(SEED)
    adj = torch.from_numpy(_adjacency(config.n_compartments)).float()
    return {"MLP": MLPDynamics(config.n_compartments), "Graph": GraphDynamics(adj), "PhysicsResidual": PhysicsResidualDynamics(adj, config), "PhysicsOnly": PhysicsOnlyDynamics(adj, config)}


def run(output_dir=None, config_path=None) -> dict:
    global OUT, TEACHER, RESULTS
    if output_dir is not None: OUT = Path(output_dir).resolve()
    if (OUT / 'run_manifest.json').exists():
        raise FileExistsError('Completed output exists; choose a new --output-dir')
    TEACHER=OUT/'teacher'; RESULTS=OUT/'results'
    cfgpath=Path(config_path or ROOT/'configs/physics_guided_baseline_fixture.yaml')
    settings=yaml.safe_load(cfgpath.read_text())
    if settings['seed'] != SEED or settings['status'] != 'FIXTURE_TEST_ONLY':
        raise ValueError('P0 fixture seed/status mismatch')
    _configure_determinism()
    OUT.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(exist_ok=True)
    (OUT/'config_snapshot.yaml').write_text(cfgpath.read_text())
    config=CableTeacherConfig(**settings['teacher'])
    train_settings=settings['students']
    start=time.perf_counter();examples=build_teacher_dataset(config)
    teacher_dataset_wall=time.perf_counter()-start
    models=_student_models(config)
    rows=[];gradients=[]
    (OUT/'models').mkdir(exist_ok=True)
    for name,model in models.items():
        if name=='PhysicsOnly':
            fit={'epochs':0,'parameter_count':0,'training_wall_clock_s':0.,'device':'cpu',
                 'teacher_training_trajectories_shared':0,'gpu_peak_memory_mb':0.}
        else:
            fit=_fit(model,examples,epochs=train_settings['epochs'],learning_rate=train_settings['learning_rate'])
        metrics=_eval(model,examples,config,threshold=train_settings['latency_threshold_mv'],
                      readout=train_settings['rollout_readout_compartment'])
        rows.append({'model':name,**fit,**metrics,'status':'FIXTURE_TEST_ONLY'})
        gradients.append(gradient_check(config,model))
        torch.save(model.state_dict(),OUT/'models'/f'{name}.pt')
        print(name, metrics['one_step_voltage_rmse'],metrics['multi_step_voltage_rmse'],flush=True)
    search=active_search(config,models['PhysicsResidual'],k=settings['active_search']['k'],
                         time_ms=settings['active_search']['target_time_ms'])
    pd.DataFrame(rows).to_csv(RESULTS/'student_comparison.csv',index=False)
    pd.DataFrame(gradients).to_csv(RESULTS/'gradient_check.csv',index=False)
    search.to_csv(RESULTS/'active_search_teacher_rechecked.csv',index=False)
    comparison=pd.DataFrame(rows)
    stable_cols=[c for c in comparison.columns if not c.endswith('wall_clock_s')]
    stable_csv=comparison[stable_cols].to_csv(index=False,lineterminator='\n')
    source_paths=list((ROOT/'src/neuronmimic/physics_guided').glob('*.py'))+[Path(__file__),cfgpath]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'status':'FIXTURE_TEST_ONLY','protocol_version':'nature_kc_p0_metrics_v2',
              'dataset_version':'synthetic_chain12_fixture_v1','seed':SEED,'model_version':'same_architectures_corrected_evaluation_v2',
              'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source_paths},
              'teacher_dataset_wall_clock_s':teacher_dataset_wall,
              'teacher_query_counts':{'dataset_shared':len(examples),'gradient_checks':sum(r['teacher_queries'] for r in gradients),
                                      'search_shared_target':1,'search_candidate_evaluations':len(search),'search_proposals':0},
              'stable_metric_columns':stable_cols,'stable_metric_sha256':hashlib.sha256(stable_csv.encode()).hexdigest(),
              'limitations':['one morphology and seed','mixed-state legacy training loss retained; evaluation channels separated',
                            'same-resolution exact-physics reference; no coarse-grid speedup or biology claim'],
              'outputs':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'stable_metric_sha256':manifest['stable_metric_sha256']}))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/physics_guided_baseline_fixture.yaml')
    args=parser.parse_args()
    run(args.output_dir,args.config)
