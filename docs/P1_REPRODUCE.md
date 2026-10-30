# P1-v2.2 structural baseline

Code: https://github.com/ydchen0806/neuronmimic
Data: https://huggingface.co/datasets/cyd0806/neuronmimic-data

This is a retrospective FlyWire-v783 analysis, not a blind test or biological validation.
The full 5,177 KC registry and frozen stable8 (965 members) remain unchanged.
CM, SU, and new P1 exploratory labels are separate namespaces.

## Run from the compact public inputs

```bash
pip install -e '.[test,structure]'
pytest -q
hf download cyd0806/neuronmimic-data --repo-type dataset --revision c8e36f59da204e52a94fd7a1a88df8b113b32807 --include 'p1_v2_fine_types/*' --local-dir data
python scripts/run_kc_p1_v2.py --bundle-root data/p1_v2_fine_types --output-dir outputs/p1_replay
python scripts/finalize_kc_p1_v2.py --output-dir outputs/p1_replay
```

For a pinned release, pass `--revision <HF commit>` to `hf download`; see PUBLISH_STATUS.md.
A completed output directory is protected against overwrite. Choose another output path to rerun.
The replay rebuilds the directed graph from the published pair table, redoes unordered-pair
splitting, train-only embedding, six geometry-discovery ablations, full-parent splits,
and the five model comparisons. It also regenerates the 5177×5177 soma-distance matrix
(about 107 MB uncompressed), which is omitted from the compact download.

The compact replay **imports** the original MaleCNS QC receipt and contact-free geometry
feature table. It does not repeat raw-source annotation or skeleton extraction audits.
Original input file hashes, original runner snapshots and environment versions are included.
All FlyWire root IDs remain strings. No live credentials or private physiology data are needed.

## Methods and scope

Unordered neuron pairs share train/validation/test assignment (70/15/15).
Neither direction of held-out pairs enters the embedding or training degree features.
Contact-derived labels/features are excluded from this pilot.
The normalized discovery objective combines directed graph embedding, actual soma
Euclidean distance, and contact-free skeleton descriptors. Global K=30 is a fixed
ablation resolution; it is not a claim of 30 stable biological units.

Local K=2/3/4 tests include complete fine-type parents and residuals on both sides.
The initial coarse-label pilot omitted KCab-s; the corrected v2.2 uses both coarse
and known fine labels in M0 and the frozen fine labels to define full parents.

All five prediction models use the same 120,000 pair sample, split and two-setting
validation budget. M2_descriptor is not the complete regional arbor/contact model.
M4 is a continuous **training-graph** embedding and has more features, so its lower
loss does not establish a continuous-geometric biological principle. Test probabilities
are case-control scores, not natural-prevalence connection or transmission probabilities.

## Actual results

Test log loss: M0 0.260694; M1 0.259567; M2_descriptor 0.243393;
M3_descriptor_discrete 0.236744; M4_descriptor_continuous 0.221216.
Independent process replay reproduced model scores, assignments and parent-child
members exactly on the recorded environment. Source-data audit was not rerun in the replay.

An exploratory 18-cell left ap1 subset contains all 16 CM019 residuals at K=3 and K=4.
It is not promoted to stable. Stronger soma weighting reduces spatial radius but
fails to improve dropout stability; visual compactness alone is inadequate.

MaleCNS still has 31,355 unmatched edge endpoints even using the complete source id
column. Cross-connectome replication and functional/causal claims remain blocked.
See the dataset P1_REPORT.md, MODEL_COMPARISON.csv, P1_STATUS.json and failure logs.
