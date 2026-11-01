# Publication status — P0 corrected release

Code: https://github.com/ydchen0806/neuronmimic (master).
Data: https://huggingface.co/datasets/cyd0806/neuronmimic-data .

HF P0 commit: 22b892d184736f1f7c7c98a8a18eb409d603ccd0.
The previously documented ydchen HF namespace was incorrect for the supplied account.
The available HTTP CONNECT proxy successfully uploaded the data.

Validation: 19 standalone tests passed; complete CPU baseline rerun reproduced
stable metric hash d6ee192e77ebfd593b7845643b2dfe811db060ec28413f5c1e4abeacb8e48f91.
Historical evaluation metrics are invalidated, not deleted. Updated metrics remain
FIXTURE_TEST_ONLY. This release establishes no new biological functional units.

## P1-v2.2 release

HF dataset revision: `c8e36f59da204e52a94fd7a1a88df8b113b32807`. All 83 manifest-listed files were
downloaded from this exact revision and SHA-256 verified.
Standalone suite: 23 tests passed. Compact-bundle replay exactly reproduced
all five model scores, global assignments and parent-child memberships.
Ten real-data structural checks passed. Imported source-audit receipts are
explicitly distinguished from audits rerun by the compact replay.

P1A/B/C are implemented with the MaleCNS mapping blocker preserved. P1D is a
retrospective descriptor pilot; P1E independent-connectome confirmation is blocked.
No new candidate was promoted to frozen stable or biologically functional.

## P1D-v3 and bilateral geometry release — 2026-11-01

HF immutable revision: `c5054f9c96b259c6ff6e5d1fd3556894eae76fad`. All 313 manifest-listed
files plus the manifest itself (314 total) were downloaded from that exact revision
and SHA-256 verified, including historical files. No prior results were deleted.

Standalone suite: 28 tests passed. Independent-process compact model refit exactly
reproduced 475,722 predictions (maximum absolute difference 0), 108 portable model
parameter sets, split tables, model scores and contrasts. Parent repository focused
P1D/bilateral tests: 13 passed. Original raw-source/geometry audits are imported by
the public replay and are explicitly not claimed as rerun.

The release includes source snapshots with hashes, frozen protocols, retained failed
matched controls, low-count-edge sensitivity, MaleCNS/BANC source QC and bilateral
soma-block/cross-side prediction results. MaleCNS endpoint provenance is resolved
as an annotation-coverage gap; independent-connectome mechanism validation remains
open. The geometry results come from one FlyWire animal. Stable8 is unchanged;
newly validated functional units remain zero.

See [replay commands](docs/P1D_REPRODUCE.md), [verification receipt](docs/p1d_v3/HF_RELEASE.json)
and [independent replay check](docs/p1d_v3/independent_replay_check.json).

## Full KC anatomy release — 2026-11-01

HF immutable revision: `b4143c83be3df42a95983d2be34af4354a25a84c`. All591 manifest files plus
the manifest itself were downloaded from the exact remote revision and SHA256
verified (592files,918,426,875bytes including historical releases). The new
structure_handoff_v1 namespace contains278files /836,683,041bytes.

FlyWire:5,177 KC geometries,1,881,678 incident physical contact pairs, full
pre/post positions, branch projections/quality, known-type and residual context.
MaleCNS v1.0:4,064 original same-version public coarse SWCs,4,027,967 incident
physical contacts and990,665 presynaptic site transmitter records. Source MD5
and SHA256 checks, all skeleton coverage/topology and aggregate-edge conservation
passed. No v0.9 skeleton was silently joined to v1.0.

Software suite31passed; actual group/cell export tests passed. CM019132members
and SU019104members remain distinct (104overlap). Query commands and exact data
revision are in [STRUCTURE_HANDOFF.md](docs/STRUCTURE_HANDOFF.md). No new stable
or biological functional unit was declared. Full-brain EM/segmentation volumes
and non-KC partner skeletons are outside this KC-scope handoff.
