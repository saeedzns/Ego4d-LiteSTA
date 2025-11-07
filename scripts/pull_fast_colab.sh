#!/usr/bin/env bash
set -euo pipefail

# Fast pull for Colab: perform a rebase pull inside the Drive-mounted repo
# to update notebooks/code, with autostash support to preserve local edits.
#
# Usage:
#   export WORKDIR=/content/drive/MyDrive/Ego4d-LiteSTA
#   bash scripts/pull_fast_colab.sh
#
# Optional env:
#   FAST_PULL_AUTOSTASH=1   # stash local changes before pulling
#   FAST_PULL_INCLUDE_HEAVY=1  # unused here (kept for symmetry with fast push)

if [[ -z "${WORKDIR:-}" ]]; then
  echo "Set WORKDIR env to your Drive repo path." >&2
  echo "Example: export WORKDIR=/content/drive/MyDrive/Ego4d-LiteSTA" >&2
  exit 1
fi

if [[ ! -d "$WORKDIR/.git" ]]; then
  echo "WORKDIR does not look like a git repo: $WORKDIR" >&2
  exit 1
fi

# Keep SSH alive; improve pack reliability
export GIT_SSH_COMMAND="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
git config --global core.compression 9 >/dev/null 2>&1 || true
git config --global pack.threads 2 >/dev/null 2>&1 || true

BRANCH=$(git -C "$WORKDIR" rev-parse --abbrev-ref HEAD || echo main)

echo "[fast-pull] Repo: $WORKDIR (branch $BRANCH)"

# Optionally stash local changes
if [[ "${FAST_PULL_AUTOSTASH:-}" == "1" ]]; then
  if ! git -C "$WORKDIR" diff --quiet || ! git -C "$WORKDIR" diff --cached --quiet; then
    echo "[fast-pull] Autostashing local changes..."
    git -C "$WORKDIR" stash push -u -m "fast-pull autostash $(date -u +'%Y-%m-%d %H:%M:%SZ')" >/dev/null || true
  fi
fi

echo "[fast-pull] Fetching and rebasing onto origin/$BRANCH ..."
git -C "$WORKDIR" fetch --prune origin
git -C "$WORKDIR" pull --rebase --autostash origin "$BRANCH" || true

echo "[fast-pull] Latest commit:"
git -C "$WORKDIR" --no-pager log -1 --oneline

echo "[fast-pull] Changed notebooks (if any):"
git -C "$WORKDIR" --no-pager diff --name-only HEAD@{1}...HEAD -- '*.ipynb' || true

echo "[fast-pull] Done. Note: Colab notebook UI does not auto-reload file contents."
echo "[fast-pull] Close and reopen the notebook (or refresh the tab) to see changes."
