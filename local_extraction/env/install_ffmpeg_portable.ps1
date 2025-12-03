<#
Downloads a portable Windows build of FFmpeg and installs ffmpeg.exe and ffprobe.exe
into your local virtualenv so they’re on PATH when the venv is activated.

Usage (Windows PowerShell 5.1 or PowerShell 7):
  # From the repo root
  powershell -NoProfile -ExecutionPolicy Bypass -File local_extraction\install_ffmpeg_portable.ps1

Optional parameters:
  -VenvScripts   Path to the venv Scripts folder (default: local_extraction\.venv\Scripts)
  -IncludeFFplay Also copy ffplay.exe (optional GUI player)

Notes:
  - Requires internet access.
  - You should create/activate the venv first (see local_extraction/setup_venv.ps1).
#>

[CmdletBinding()]
param(
  [string]$VenvScripts = "local_extraction\\.venv\\Scripts",
  [switch]$IncludeFFplay
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-RepoRoot {
  param([string]$Start)
  $p = (Resolve-Path $Start).Path
  while ($true) {
    if (Test-Path (Join-Path $p '.git')) { return $p }
    $parent = Split-Path -Parent $p
    if (-not $parent -or $parent -eq $p) { return (Resolve-Path $Start).Path }
    $p = $parent
  }
}

$repoRoot = Resolve-RepoRoot -Start "."
Set-Location $repoRoot

$scriptsDir = Resolve-Path -LiteralPath $VenvScripts -ErrorAction SilentlyContinue
if (-not $scriptsDir) {
  Write-Host "[ffmpeg] venv Scripts folder not found: $VenvScripts" -ForegroundColor Yellow
  Write-Host "         Run local_extraction/setup_venv.ps1 first, or pass -VenvScripts." -ForegroundColor Yellow
  throw "Venv Scripts not found"
}
$scriptsDir = $scriptsDir.Path
Write-Host "[ffmpeg] Installing into: $scriptsDir"

# Ensure TLS 1.2 for Invoke-WebRequest on older PowerShell
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }

$uri = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"
$tmp = Join-Path $env:TEMP ("ffmpeg_dl_" + [Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$zip = Join-Path $tmp 'ffmpeg.zip'

Write-Host "[ffmpeg] Downloading: $uri"
Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile $zip

Write-Host "[ffmpeg] Extracting archive..."
Expand-Archive -Path $zip -DestinationPath $tmp -Force

$root = Get-ChildItem -Path $tmp -Directory | Select-Object -First 1
if (-not $root) { throw "Could not locate extracted folder under $tmp" }
$bin = Join-Path $($root.FullName) 'bin'

foreach ($exe in @('ffmpeg.exe','ffprobe.exe')) {
  $src = Join-Path $bin $exe
  if (-not (Test-Path $src)) { throw "Missing $exe in $bin" }
  Copy-Item -Force $src $scriptsDir
  Write-Host "[ffmpeg] Installed $exe"
}

if ($IncludeFFplay) {
  $ffplay = Join-Path $bin 'ffplay.exe'
  if (Test-Path $ffplay) {
    Copy-Item -Force $ffplay $scriptsDir
    Write-Host "[ffmpeg] Installed ffplay.exe"
  } else {
    Write-Host "[ffmpeg] ffplay.exe not present in this build" -ForegroundColor Yellow
  }
}

Write-Host "[ffmpeg] Verifying..."
& (Join-Path $scriptsDir 'ffmpeg.exe') -version | Select-Object -First 1 | Out-Host
& (Join-Path $scriptsDir 'ffprobe.exe') -version | Select-Object -First 1 | Out-Host

Write-Host "[ffmpeg] Done. Ensure your venv is activated so these are on PATH:" -ForegroundColor Green
Write-Host "  . $scriptsDir/Activate.ps1"
