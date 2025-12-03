#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Creates a local Python venv under local_extraction/.venv and installs requirements.

function Find-Python {
  $candidates = @('py -3','py','python','python3')
  foreach ($cmd in $candidates) {
    try { & $cmd -c "import sys; print(sys.version)" | Out-Null; return $cmd } catch { }
  }
  throw "Python not found. Install Python 3.9+ and retry."
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $repoRoot

$py = Find-Python
Write-Host "[env] Using Python: $py"

$venvPath = Join-Path $repoRoot 'local_extraction/.venv'
if (Test-Path $venvPath) { Write-Host "[env] Reusing venv at $venvPath" } else {
  Write-Host "[env] Creating venv at $venvPath"
  & $py -m venv $venvPath
}

$activate = Join-Path $venvPath 'Scripts/Activate.ps1'
if (-not (Test-Path $activate)) { throw "Activate script not found: $activate" }
Write-Host "[env] Activating venv"
. $activate

python -m pip install --upgrade pip
pip install -r local_extraction/requirements.txt

Write-Host "[env] Done. Activate later with:`n  `n  . $activate`n  `nThen run:`n  python local_extraction/ego4d_resume_fast_extract.py"
