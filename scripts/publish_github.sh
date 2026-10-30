#!/usr/bin/env bash
set -euo pipefail

# Credentials are intentionally read only from the environment. Never put a
# token in a remote URL, README, shell history, or git config.
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN in the environment}"
REPOSITORY="${GITHUB_REPOSITORY:-ydchen0806/neuronmimic}"
REMOTE="https://github.com/${REPOSITORY}.git"
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REMOTE"
git branch -M main
git push -u origin main
