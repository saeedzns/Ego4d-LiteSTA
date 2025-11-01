#!/usr/bin/env bash
set -euo pipefail

# Simple daily pull script

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

