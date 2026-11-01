# Original BioFly P1D source snapshot

These are exact source files used for the P1D-v3 and bilateral experiments.
`source_manifest.json` hashes the unmodified files; the source parent commit alone
is insufficient because the scripts were uncommitted when run.
Run them within the original BioFly tree with the input paths recorded in HF
`p1d_v3/audit/` and `p1d_v3_bilateral/audit/`. This is an auditable source snapshot,
not a self-contained raw-connectome distribution. Real skeletons, ROI meshes and
original MaleCNS/BANC downloads are not bundled. Original tree-geometry extraction
needs trimesh and rtree; reported environments and numerical checks are retained.

For a tested standalone model replay use `scripts/run_kc_p1d_bilateral.py` and
`docs/P1D_REPRODUCE.md`. Its import/path changes do not alter model mathematics.
