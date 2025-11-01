#!/usr/bin/env bash
set -euo pipefail

# Simple daily push script

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Not a git repository." >&2
  exit 1
fi

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" = "HEAD" ]; then
  echo "Detached HEAD; please checkout a branch." >&2
  exit 1
fi

echo "[push] Staging all changes..."
git add -A

if git diff --cached --quiet; then
  echo "[push] No changes to commit."
  exit 0
fi

msg="chore: daily sync $(date -u +'%Y-%m-%d %H:%M:%SZ')"
echo "[push] Committing with message: $msg"
git commit -m "$msg" || true

echo "[push] Updating branch with latest remote (rebase)..."
git fetch --prune origin
git pull --rebase --autostash origin "$branch" || true

echo "[push] Pushing to origin/$branch..."
git push -u origin "$branch"

echo "[push] Done."

