# NeuronMimic

NeuronMimic is a small, reproducible baseline for **physics-guided differentiable neural dynamics**. It preserves directed edges, calibrates bounded LIF parameters, and exposes a fixture-only passive cable Teacher with three comparable Student models: MLP, graph message passing, and physics plus learned residual.

## What is complete

- Directed surrogate-LIF model and held-out trajectory calibration.
- Deterministic passive-cable Teacher fixture.
- MLP, morphology-graph, and physics-residual Student baselines.
- Explicit train/test stimulus families, reproducibility checks, and gradient and active-search hooks in the source tree.
- MaleCNS v1.0 structural adapter and audit are maintained in the parent BioFly repository; this release does not package restricted raw connectome data.

## What is not complete

This release is **not** a calibrated multi-compartment neural twin. The cable Teacher is `FIXTURE_TEST_ONLY`. A real Arbor Teacher requires complete compartment geometry, contact-to-branch mapping, conductance parameters, and held-out biological recordings. No fixture result should be described as physiology or independent biological validation.

## Reproduce

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest -q
python scripts/run_neuronmimic_fixture.py
python scripts/run_physics_guided_baseline.py
```

## Releases and data

- Code repository: https://github.com/ydchen0806/neuronmimic
- Reusable compact data and receipts: https://huggingface.co/datasets/ydchen/neuronmimic-data
- FlyWire KC structural-entry package (separate, source-bound release): `KC_fine_unit_structural_entries_20261030_v2_final4_clean.zip`, SHA-256 `5446c1df6419c1afddcdefe88d2c71c0481c7e18a6ec6e772222d049bce2e718`

The GitHub repository contains code and protocols. The Hugging Face dataset contains only compact, source-bound tables, fixture outputs, schemas, and provenance receipts. Raw or license-restricted connectome files are excluded. The two locations will be cross-linked after remote creation; upload scripts must receive credentials through the environment and never write tokens to files or history.

## Status labels

`VERIFIED_REAL_DATA` means a source table was audited and hashed. `COMPUTED_MODEL_RESULT` means a result from the declared model. `FIXTURE_TEST_ONLY` means synthetic or passive-cable test data. `BLOCKED_MISSING_DATA` means the required biological input is unavailable or not yet mapped.
