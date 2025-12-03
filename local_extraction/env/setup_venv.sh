#!/usr/bin/env bash
set -euo pipefail

# Creates a local Python venv under local_extraction/.venv and installs requirements.

repo_root=$(cd "$(dirname "$0")"/.. && pwd)
cd "$repo_root"

py_cmd="python3"
if ! command -v python3 >/dev/null 2>&1; then
  py_cmd="python"
fi

echo "[env] Using Python: $py_cmd"

venv_path="$repo_root/local_extraction/.venv"
if [[ -d "$venv_path" ]]; then
  echo "[env] Reusing venv at $venv_path"
else
  echo "[env] Creating venv at $venv_path"
  "$py_cmd" -m venv "$venv_path"
fi

if [[ -f "$venv_path/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$venv_path/bin/activate"
else
  # Windows git-bash path
  # shellcheck disable=SC1090
  source "$venv_path/Scripts/activate" || true
fi

python -m pip install --upgrade pip
pip install -r local_extraction/requirements.txt

echo "[env] Done. Activate later with:"
echo "  source local_extraction/.venv/bin/activate   # (bash)"
echo "or on Windows PowerShell:"
echo "  . local_extraction/.venv/Scripts/Activate.ps1"
echo "Then run:"
echo "  python local_extraction/ego4d_resume_fast_extract.py"
