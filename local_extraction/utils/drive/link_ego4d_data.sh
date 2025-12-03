#!/usr/bin/env bash
set -euo pipefail

# Local-only symlink helper (copy of scripts/link_ego4d_data.sh)
# Creates a link in the repo to the Drive ego4d_data folder.

TARGET="${1:-${EGO4D_DATA:-}}"

if [[ -z "${TARGET}" ]]; then
  for cand in \
    "/content/drive/MyDrive/ego4d_data" \
    "/drive/MyDrive/ego4d_data" \
    "/content/drive/Shareddrives/ego4d_data"; do
    if [[ -d "$cand" ]]; then TARGET="$cand"; break; fi
  done
fi

if [[ -z "${TARGET}" || ! -d "${TARGET}" ]]; then
  echo "[link] Could not locate ego4d_data. Pass the path as an argument or set EGO4D_DATA." >&2
  echo "       Example: bash local_extraction/link_ego4d_data.sh /content/drive/MyDrive/ego4d_data" >&2
  exit 1
fi

LINK_NAME="${LINK_NAME:-ego4d_data}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

if [[ -L "$LINK_NAME" || -e "$LINK_NAME" ]]; then
  if [[ -L "$LINK_NAME" ]]; then
    CURRENT="$(readlink "$LINK_NAME" || true)"
    if [[ "$CURRENT" == "$TARGET" ]]; then
      echo "[link] Already linked: $LINK_NAME -> $TARGET"
      ls -la "$LINK_NAME" | head -n 20 || true
      exit 0
    fi
  fi
  rm -rf -- "$LINK_NAME"
fi

ln -s "$TARGET" "$LINK_NAME"
echo "[link] Created: $REPO_ROOT/$LINK_NAME -> $TARGET"
ls -la "$LINK_NAME" | head -n 20 || true
