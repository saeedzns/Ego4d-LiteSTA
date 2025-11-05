#!/usr/bin/env bash
set -euo pipefail

# Fast push from Colab by cloning into /content (fast local disk),
# syncing code/config files from Drive, then committing & pushing.
#
# Usage:
#   export WORKDIR=/content/drive/MyDrive/Ego4d-LiteSTA
#   bash scripts/push_fast_colab.sh "optional commit message"
#
# Notes:
# - By default excludes heavy artifacts (runs/, logs/, outputs/, frames/, videos/, *.pt, *.mp4, *.jpg, etc.)
# - To include heavy artifacts too, set FAST_PUSH_INCLUDE_HEAVY=1
# - Requires that $WORKDIR points at your Drive repo with a configured origin remote.

MSG=${1:-}
if [[ -z "${WORKDIR:-}" ]]; then
  echo "Set WORKDIR env to your Drive repo path or pass it as first arg." >&2
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

ORIGIN_URL=$(git -C "$WORKDIR" config --get remote.origin.url || true)
BRANCH=$(git -C "$WORKDIR" rev-parse --abbrev-ref HEAD || echo main)
if [[ -z "$ORIGIN_URL" ]]; then
  echo "Could not read origin URL from $WORKDIR/.git/config" >&2
  exit 1
fi

TMP=/content/tmp_push_repo
rm -rf "$TMP"
echo "[fast-push] Cloning $ORIGIN_URL (branch $BRANCH) into $TMP ..."
git clone --depth 1 -b "$BRANCH" "$ORIGIN_URL" "$TMP" >/dev/null 2>&1 || git clone --depth 1 "$ORIGIN_URL" "$TMP"

if [[ "${FAST_PUSH_INCLUDE_HEAVY:-}" == "1" ]]; then
  echo "[fast-push] Rsync ALL from Drive → /content (including heavy artifacts) ..."
  rsync -a --delete \
    --exclude='.git/' \
    --exclude='.ipynb_checkpoints/' \
    "$WORKDIR"/ "$TMP"/
else
  echo "[fast-push] Rsync code from Drive → /content (excluding heavy artifacts) ..."
  rsync -a --delete \
    --exclude='.git/' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='runs/' --exclude='logs/' --exclude='outputs/' --exclude='checkpoints/' \
    --exclude='data/' --exclude='frames/' --exclude='videos/' \
    --exclude='*.mp4' --exclude='*.webm' --exclude='*.mkv' \
    --exclude='*.jpg' --exclude='*.jpeg' --exclude='*.png' --exclude='*.bmp' --exclude='*.tif' --exclude='*.tiff' \
    --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.onnx' --exclude='*.npz' --exclude='*.npy' \
    "$WORKDIR"/ "$TMP"/
fi

cd "$TMP"
echo "[fast-push] Staging changes ..."
git add -A
if git diff --cached --quiet; then
  echo "[fast-push] No changes to commit."
  exit 0
fi

COMMIT_MSG=${MSG:-"chore: colab fast push $(date -u +'%Y-%m-%d %H:%M:%SZ')"}
echo "[fast-push] Committing: $COMMIT_MSG"
git commit -m "$COMMIT_MSG" >/dev/null 2>&1 || true

echo "[fast-push] Pushing to origin/$BRANCH ..."
git push origin "$BRANCH"
echo "[fast-push] Done."
