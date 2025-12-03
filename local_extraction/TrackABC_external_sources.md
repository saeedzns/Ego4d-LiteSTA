# Track ABC External Data References (Outside Workspace)

Based on scanning the Track A/B/C Python scripts, no paths or assets are referenced outside this workspace (i.e., no absolute paths to other drives or storage locations). All JSON/JSONL/CSV/pt assets are under the project’s own `local_extraction/` and `runs/` directories.

If you add any external paths (e.g., `D:\\data\\...` or `/mnt/shared/...`), list them here with the script and purpose.
