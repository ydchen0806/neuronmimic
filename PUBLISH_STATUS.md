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
