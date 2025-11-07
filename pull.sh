#!/usr/bin/env bash
set -euo pipefail

# Simple daily pull script

# If on Colab and FAST_PULL is set, use fast path script
if [[ "${FAST_PULL:-}" == "1" && -d "/content" && -f "scripts/pull_fast_colab.sh" ]]; then
  echo "[pull] Using fast Colab path via scripts/pull_fast_colab.sh"
  WORKDIR="${WORKDIR:-$PWD}" bash scripts/pull_fast_colab.sh
  exit 0
fi

# Keep SSH alive on flaky networks (e.g., Colab)
export GIT_SSH_COMMAND="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10"

# Optional: slightly better pack/compression defaults
git config --global core.compression 9 >/dev/null 2>&1 || true
git config --global pack.threads 2 >/dev/null 2>&1 || true

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Not a git repository." >&2
  exit 1
fi

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" = "HEAD" ]; then
  echo "Detached HEAD; please checkout a branch." >&2
  exit 1
fi

echo "[pull] Fetching and pulling latest for '$branch'..."
git fetch --prune origin
git pull --rebase --autostash origin "$branch"
echo "[pull] Done."
