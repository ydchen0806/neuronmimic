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
