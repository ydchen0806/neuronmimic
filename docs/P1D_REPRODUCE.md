# P1D-v3 and bilateral regional-tree experiment (2026-11-01)

Code: https://github.com/ydchen0806/neuronmimic
Data: https://huggingface.co/datasets/cyd0806/neuronmimic-data

## Result and scope

Within the known FlyWire-v783 ap1 fine type, regional tree proximity improves
connection prediction over soma coordinates/distance and cell-level tree descriptors.
Left 148 and right 133 cells give 21,756 and 17,556 directed nonself pairs, respectively.
All within-parent nonedges are included; there is no case-control sampling.

| Evaluation | Left-trained M1 → M2 log loss | Right-trained M1 → M2 log loss |
| --- | --- | --- |
| Unordered-pair holdout | .57461 → .52815 | .54515 → .51274 |
| Five soma-block holdouts pooled | .60146 → .56618 | .59826 → .57348 |
| Transfer to opposite hemisphere | .56142 → .54163 | .60188 → .55497 |

All ten soma folds improve. Predictions use no candidate labels, graph degrees,
graph embeddings or observed contact features. These are retrospective tests in
**one animal**, not independent-specimen or functional validation. The two brain
sides are not independent animals. Spatial blocks have no buffer (minimum soma
separation 2.474 μm); tree arbors can overlap. Cross-side registration is a single
reflection, not cellular homology or a nonrigid anatomical registration.

Each model uses 128 random Fourier features plus an intercept, L2 logistic
regression, four C/gamma settings selected only on source validation data, and
seeds 20261101–03. Input dimensions differ: identical nominal coefficient counts
do not ensure identical effective model capacity. M2 adds regional proximity,
regional lengths and missingness together, so the experiment does not isolate a
single causal geometric mechanism. Fixed-model 1,000-draw vertex-Poisson intervals
use seed 2026110109 and omit model-fitting, tuning and animal-level uncertainty.

## Reproduce the published bilateral model experiment

```bash
pip install -e '.[test,structure]'
DATA_REV=c5054f9c96b259c6ff6e5d1fd3556894eae76fad
hf download cyd0806/neuronmimic-data --repo-type dataset --revision "$DATA_REV" \
  --include 'p1d_v3_bilateral/*' --local-dir data
python scripts/run_kc_p1d_bilateral.py models \
  --bundle-root data/p1d_v3_bilateral --output-dir outputs/bilateral_replay
python scripts/finalize_kc_p1d_bilateral.py --output-dir outputs/bilateral_replay
python scripts/verify_p1d_replay.py --reference data/p1d_v3_bilateral \
  --replay outputs/bilateral_replay --receipt outputs/bilateral_replay/replay_check.json
P1D_BILATERAL_DATA=data/p1d_v3_bilateral pytest -q tests/test_p1d_bilateral.py
```

The replay refits 108 final models (four settings each), evaluates 126 model/task
rows, and regenerates predictions, contrasts, portable weights and plots. Completed
run directories are protected from overwrite. Tested source-model runtime was
23.35 seconds using four CPU threads; this is a local measurement, not a general
performance promise. No GPU or dynamics simulation is needed. Reproduction
versions: Python 3.11.11, NumPy 1.26.4, SciPy 1.16.3, pandas 2.3.3,
scikit-learn 1.8.0. Other versions may change numerical results.

The compact replay starts from published design matrices and outcome/split tables.
It **imports** original geometry/source receipts; it does not redownload or rebuild
raw skeletons, regional meshes or source connectomes. `audit/provenance.json` in
replay output identifies this boundary and hashes the imported inputs. Source
experiment figures/data are published separately from replay output.

Full source-mode code is preserved verbatim in `research/p1d_source_snapshot/`.
The audit manifests list exact original input paths and hashes. Raw-source geometry
extraction requires the original BioFly inputs and existing trimesh/rtree setup;
the original source files are not claimed to be standalone. The published geometry
is length-weighted midpoint integration of point-to-segment distances within
FULL/CA/PED/ML/VL mesh regions at 1/2/5 μm kernel scales. Four sorted right-side IDs
were checked at 0.5 vs 1 μm quadrature; exact point-to-segment reference also passed.
Mesh regions approximate anatomy; proximity is not an observed synaptic contact.

## Earlier P1D-v3 results and failures retained

Download `p1d_v3/*` for source QC, frozen members, all predictions, matched-null
failures, sensitivity tables, protocols and reports. This earlier stage remains a
source-mode experiment; the bilateral compact replay does not rerun its perturbation,
matching or crossbrain audit stages. It contains:

- A frozen 18-cell subset: 16 historical CM030 cells plus 2 CM019, **not a new subtype**.
- Mean Jaccard .936 under 20 random 20%-pair deletions; removing single-synapse
  edges reduces K=3 overlap to .409. Equally sparse random controls also destabilize.
- Shared-geometry nested label increment .002192 log loss, with exploratory
  interval [-.00254, .00716]; nonzero residual information is not established.
- 0/1,000 matched groups pass the frozen geometry/degree balance rule. Matched
  PN/MBON significance is NA; 72 observed routes remain descriptive only.
- MaleCNS: all 31,355 annotation-missing endpoints exist in the same release's
  original neurotransmitter table. The annotation gap is real; a release mismatch
  was not found. Do not manufacture missing cell types.
- BANC: 4,552 KC; root_888 agrees with banc_888_id throughout. Current root IDs
  differ in 240 cases; 4,532 side annotations are unknown. No external geometric
  mechanism test or homologous-unit validation was completed.

Stable8 remains 8/965 and unchanged by SHA-256. Newly validated functional units:
**zero**. The next scientific gate is independent-connectome region/type harmonization
and prospective input/output prediction, not adding more clusters or optimizing
NeuronMimic Students. Recent literature supplied by the user is attributed in the
literature ledger; this release does not claim successful retrieval of blocked papers.
