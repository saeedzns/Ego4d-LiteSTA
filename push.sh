#!/usr/bin/env bash
set -euo pipefail

# Simple daily push script
# - Accepts optional commit message as $1
# - On Colab, you can enable a faster path by setting FAST_PUSH=1 which
#   clones to /content and rsyncs code-only before pushing.

MSG=${1:-}

# If user requested fast path (e.g., on Colab), delegate to scripts/push_fast_colab.sh
if [[ "${FAST_PUSH:-}" == "1" && -d "/content" && -f "scripts/push_fast_colab.sh" ]]; then
  echo "[push] Using fast Colab path via scripts/push_fast_colab.sh"
  WORKDIR="${WORKDIR:-$PWD}" bash scripts/push_fast_colab.sh "${MSG}"
  exit 0
fi

# Keep SSH alive on flaky networks (e.g., Colab)
export GIT_SSH_COMMAND="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10"

# Optional: slightly better pack/compression defaults for large pushes
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

echo "[push] Staging all changes..."
git add -A

if git diff --cached --quiet; then
  echo "[push] No changes to commit."
  exit 0
fi

msg=${MSG:-"chore: daily sync $(date -u +'%Y-%m-%d %H:%M:%SZ')"}
echo "[push] Committing with message: $msg"
git commit -m "$msg" || true

echo "[push] Updating branch with latest remote (rebase)..."
git fetch --prune origin
git pull --rebase --autostash origin "$branch" || true

echo "[push] Pushing to origin/$branch..."
git push -u origin "$branch"

echo "[push] Done."
