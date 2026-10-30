# NeuronMimic

NeuronMimic is a small, reproducible baseline for **physics-guided differentiable neural dynamics**. It preserves directed edges, calibrates bounded LIF parameters, and exposes a fixture-only passive cable Teacher with three comparable Student models: MLP, graph message passing, and physics plus learned residual.

## Corrected P0 release (2026-10-30)

Earlier physics-guided evaluation metrics are **INVALIDATED**: trajectory broadcasting,
voltage/conductance mixing and mismatched gradient/search objectives were corrected.
Historical outputs remain for traceability; use `outputs/physics_guided_p0_v2` for new runs.
See [audit and invalidation ledger](docs/p0_audit/P0_REPORT.md).

Corrected held-out voltage RMSE (mV): MLP 5.82327, Graph 33.85993,
PhysicsResidual 0.0146632, PhysicsOnly 0.00000815265.
These are **FIXTURE_TEST_ONLY** results on one synthetic 12-compartment chain,
six training and three held-out stimuli, seed 20261030. The known-equation
solver outperforms every Student; this establishes no biological discovery
or surrogate advantage. Negative-conductance and missed-response failures are retained.

## What is complete

- Directed surrogate-LIF model and held-out trajectory calibration.
- Deterministic passive-cable Teacher fixture.
- MLP, morphology-graph, and physics-residual Student baselines.
- Explicit train/test stimulus families, reproducibility checks, and gradient and active-search hooks in the source tree.
- MaleCNS v1.0 structural adapter and audit are maintained in the parent BioFly repository; this release does not package restricted raw connectome data.

## What is not complete

This release is **not** a calibrated multi-compartment neural twin. The cable Teacher is `FIXTURE_TEST_ONLY`. A real Arbor Teacher requires complete compartment geometry, contact-to-branch mapping, conductance parameters, and held-out biological recordings. No fixture result should be described as physiology or independent biological validation.

The parent BioFly project records the current Wu-lab requirement matrix in
`artifacts/wu_20261030_science_gate/WU_REQUIREMENTS_STATUS_20261030.md`.
The stable8 dynamic gate currently uses a frozen model panel (8 input sources,
20 trials/source): four candidates show sparse model responses and four show
none. An earlier source-verified pilot contains three ALPN inputs with 100
stimulation trials plus 100 no-stim controls per source. These are model
results, not functional validation; broader inputs, complete-parent controls
and a calibrated multi-compartment backend remain required.

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
- Reusable compact data and receipts: https://huggingface.co/datasets/cyd0806/neuronmimic-data
- FlyWire KC structural-entry package (separate, source-bound release): `KC_fine_unit_structural_entries_20261030_v2_final4_clean.zip`, SHA-256 `5446c1df6419c1afddcdefe88d2c71c0481c7e18a6ec6e772222d049bce2e718`

The GitHub repository contains code and protocols. The Hugging Face dataset contains only compact, source-bound tables, fixture outputs, schemas, and provenance receipts. Raw or license-restricted connectome files are excluded. Remote upload status is recorded in PUBLISH_STATUS.md. Credentials are never included in releases.

## Status labels

`VERIFIED_REAL_DATA` means a source table was audited and hashed. `COMPUTED_MODEL_RESULT` means a result from the declared model. `FIXTURE_TEST_ONLY` means synthetic or passive-cable test data. `BLOCKED_MISSING_DATA` means the required biological input is unavailable or not yet mapped.

## Structural P1-v2.2 release

[Reproduce the KC structural experiment](docs/P1_REPRODUCE.md) from the
[public compact data](https://huggingface.co/datasets/cyd0806/neuronmimic-data/tree/main/p1_v2_fine_types).
It includes unordered-pair isolation, genuine soma/tree fusion, complete-parent splits
and a retrospective five-model descriptor comparison. Stable8 membership is unchanged;
new partitions and the 18-cell ap1 subset remain exploratory.
