Local Python Environment (Windows/WSL/Colab)

Overview
- Creates an isolated venv under `local_extraction/.venv` and installs minimal deps for the extractor.
- Nothing in `local_extraction/` is pushed to Git (local-only).
- You still need the `ffmpeg` binary installed on your system (see below).

Quick Start — Windows PowerShell
- From repo root:
  - `pwsh -File local_extraction/setup_venv.ps1`
  - Activate: `. local_extraction/.venv/Scripts/Activate.ps1`
  - Run: `python local_extraction/ego4d_resume_fast_extract.py`

Quick Start — WSL/Git Bash
- From repo root:
  - `bash local_extraction/setup_venv.sh`
  - Activate: `source local_extraction/.venv/bin/activate`
  - Run: `python local_extraction/ego4d_resume_fast_extract.py`

ffmpeg install tips
- Windows (winget): `winget install Gyan.FFmpeg` (or search: `winget search ffmpeg`)
- Windows (choco): `choco install ffmpeg`
- WSL (Ubuntu/Debian): `sudo apt-get update && sudo apt-get install -y ffmpeg`
- Verify: `ffmpeg -version` and `ffprobe -version`

Notes
- The extractor auto-detects your Drive root; set `EGO4D_ROOT` to override.
- `decord` is optional; the script falls back to `ffmpeg` if not present.
- On Windows, you may need to unblock scripts: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
