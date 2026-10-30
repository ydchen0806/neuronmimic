# Publication status

The code and compact data are prepared as two clean local git repositories.
The release tests pass, and no credential is stored in either repository.

GitHub publication is complete at
`https://github.com/ydchen0806/neuronmimic` (branch `master`, head
`fcbdebe`). Run
`scripts/publish_hf.sh PATH_TO_DATA_REPOSITORY` with `HF_TOKEN` and, if needed,
`HF_REPOSITORY` to publish the data side. The scripts fail closed when
credentials are absent.

The Hugging Face repository creation/upload remains pending because direct
access to `huggingface.co` is blocked in this environment and the configured
local proxy at `127.0.0.1:18181` is not listening. No token is stored in the
repository or its history.

## Science-gate status

The current structural/dynamic gate is recorded at
`artifacts/wu_20261030_science_gate/WU_REQUIREMENTS_STATUS_20261030.md`.
The stable8 dynamic audit used a frozen model panel (8 input sources, 20 trials
per source): four candidates showed sparse model response and four showed none;
this is a model result and is not independent biological validation. An earlier
pilot already contains three source-verified ALPN inputs with 100 stimulation
trials and 100 no-stim controls per source. The remaining gap is broader source
coverage, complete-parent matched controls, and a calibrated multi-compartment
backend.
