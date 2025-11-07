#!/usr/bin/env bash
set -euo pipefail

# Simple daily pull script

# If on Colab and FAST_PULL is set, use fast path script
if [[ "${FAST_PULL:-}" == "1" && -d "/content" && -f "scripts/pull_fast_colab.sh" ]]; then
  echo "[pull] Using fast Colab path via scripts/pull_fast_colab.sh"
  WORKDIR="${WORKDIR:-$PWD}" bash scripts/pull_fast_colab.sh
  exit 0
fi

# Keep SSH alive on flaky networks (e.g., Colab) and avoid interactive prompts
export GIT_SSH_COMMAND="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
export GIT_EDITOR=true
export GIT_TERMINAL_PROMPT=0

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

echo "[pull] Fetching latest for '$branch'..."
git fetch --prune origin

# Prefer fast-forward when HEAD has no unique commits; else rebase non-interactively
set +e
ahead_behind=$(git rev-list --left-right --count origin/"$branch"...HEAD 2>/dev/null)
set -e
left=${ahead_behind%% *}; right=${ahead_behind##* }
if [ "${right:-1}" = "0" ]; then
  echo "[pull] Fast-forward merge from origin/$branch (behind ${left:-?})"
  git merge --ff-only origin/"$branch"
else
  echo "[pull] Rebase onto origin/$branch (ours ahead ${right:-?})"
  git pull --rebase --autostash origin "$branch"
fi

echo "[pull] Latest commit:" && git --no-pager log -1 --oneline
echo "[pull] Done."
