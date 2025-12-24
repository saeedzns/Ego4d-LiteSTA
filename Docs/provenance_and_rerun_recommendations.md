# Provenance vs. Exact Rerun — Recommendations (Ego4D-LiteSTA)

Date: 2025-12-19

This note summarizes what the current logging/provenance approach guarantees today, what it does **not** guarantee, and small future upgrades if we want “one-command exact reruns” later.

## Current approach (status: sufficient)

Current behavior is designed to answer:

- **"What config did this checkpoint/metric come from?"**

It is **not** yet designed to answer (automatically):

- **"Can I rerun using exactly that config with one command?"** (without manually copying values)

### What we capture today

- **Track B training checkpoints** store the full resolved YAML config snapshot inside the `.pt` payload:
  - `config_name`
  - `yaml_config` (resolved dict)
  - `yaml_config_flat` (flattened resolved dict)

- **Track B eval runs** save a `resolved_config.json` in the RunLogger run directory and also record `config_name` and eval settings in the run log.

- **Track A Stage A / Stage B runs** write `resolved_config.json` inside the run folder and also include `config_name` + `yaml_config_flat` inside `summary.json`.

- **Track C runs** write `resolved_config.json` inside the run folder and include `config_name` + `yaml_config_flat` in the per-run summary JSON.

**Result:** Even if YAML files change later, we can still recover the exact resolved config used at the time of training/evaluation.

## What “exact rerun” really requires

Even with perfect config capture, exact rerun depends on more than YAML:

- **Code version**: the same repo commit / code state.
- **Data invariants**: manifests, extracted frames, and any derived files must still exist and match.
- **Weights/artifacts**: YOLO/VideoMAE weights paths must still resolve; checkpoints must still exist.
- **Determinism**: GPU ops and dataloader behavior can be nondeterministic unless explicitly forced.

So “exact rerun” is best treated as:

- “Rerun with the same config snapshot + same code + same data/weights + deterministic flags.”

## Recommendation (keep current approach)

For now, the current approach is **enough** because it provides strong provenance:

- You can always trace any metric/checkpoint to the full resolved config used.
- You can manually rerun by loading the preset config name (or by inspecting `resolved_config.json` / checkpoint payload).

## Future improvements (optional; implement only if needed)

If later we want **one-command exact reruns**, the best upgrades are:

### Option A — Add `--config_path` / `--resolved_config`

Add CLI support to load a saved config snapshot (dict) from a JSON file:

- Example: `--config_path local_extraction/runs/Track_B/trackB_eval_*/resolved_config.json`

Benefits:
- Makes old runs reproducible even if YAML on disk changed.
- Minimal new UX surface: one extra CLI arg.

Notes:
- This requires a small loader helper that can construct a `Config`-like object (or a minimal adapter) from a dict.

### Option B — Add a “rerun helper script”

Create a utility that reads a run directory or checkpoint and relaunches the relevant command:

- Example:
  - `python local_extraction/utils/rerun_from_run_dir.py --run_dir ...`
  - `python local_extraction/utils/rerun_from_checkpoint.py --checkpoint ... --task trackB_eval`

Benefits:
- Centralizes the logic for mapping run artifacts → CLI invocations.
- Avoids modifying each track script repeatedly.

### Option C — Add a config hash (lightweight portability)

If we keep `resolved_config.json` as the source of truth, we can also embed a small provenance stub into summaries:

- `config_name`
- `yaml_config_hash` = SHA256(`yaml_config_flat` serialized)

Benefits:
- Lets you verify if two metrics were produced with the exact same resolved config, without embedding the entire config blob everywhere.

## Practical guidance (how to use provenance today)

- If you have a **Track B checkpoint**: read the checkpoint keys `config_name`, `yaml_config_flat` to recover what was used.
- If you have a **run folder**: use the run’s `resolved_config.json` as the authoritative snapshot.
- If you only have a **summary JSON** (Track A / Track C): `config_name` + `yaml_config_flat` is already embedded for portability.

## Decision

- Current approach is sufficient for now.
- If later “exact rerun” becomes a frequent workflow need, implement **Option A** (recommended) or **Option B**.
