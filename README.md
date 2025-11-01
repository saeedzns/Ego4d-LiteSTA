# Ego4d-LiteSTA
Egocentric Lite Short‑Term Anticipation (LiteSTA)

## Daily Pull/Push Scripts
- One-time setup: `chmod +x pull.sh push.sh`
- Pull latest: `./pull.sh`
- Push changes: `./push.sh`

Notes
- Both scripts use rebase with autostash to avoid merge commits.
- `push.sh` stages everything, commits with a timestamped message, rebases, then pushes to `origin`.

Colab example
- In a cell: `!bash pull.sh` then `!bash push.sh`
