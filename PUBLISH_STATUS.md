# Publication status

The code and compact data are prepared as two clean local git repositories.
The release tests pass, and no credential is stored in either repository.

Remote publication is a separate operation because it requires valid GitHub
and Hugging Face credentials plus working network/proxy access. Run
`scripts/publish_github.sh` from this repository with `GITHUB_TOKEN` and run
`scripts/publish_hf.sh PATH_TO_DATA_REPOSITORY` with `HF_TOKEN` and, if needed,
`HF_REPOSITORY`. The scripts fail closed when credentials are absent.

At the time this release was prepared, the local GitHub CLI credential was
expired and the configured proxy refused connections. Therefore the commit
hashes are available locally, but no remote push is claimed here.
