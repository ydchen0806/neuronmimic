#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN in the environment}"
REPOSITORY="${HF_REPOSITORY:-ydchen/neuronmimic-data}"
DATA_DIR="${1:?usage: publish_hf.sh DATA_DIR}"
hf repo create "$REPOSITORY" --repo-type dataset --exist-ok --token "$HF_TOKEN"
hf upload "$REPOSITORY" "$DATA_DIR" . --repo-type dataset --token "$HF_TOKEN"
