#!/usr/bin/env python3
"""
Summarize best Track B and Track C runs into a markdown table.

Default behavior:
  - Track B: select best run by "mAP" from metrics_val_*_summary.json
  - Track C: select best run by "All_top5_mAP" from trackC_val_rate*_summary.json

Outputs a markdown file with:
  - Best Track B run
  - Best Track C run (overall)
  - Best Track C run per pruning rate
  - Full tables with main config fields
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_nested(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in d:
        return d.get(key, default)
    if "." not in key:
        return default
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _fmt(v: Any, ndigits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{v:.{ndigits}f}"
    return str(v)


def _basename(path_str: Optional[str]) -> str:
    if not path_str:
        return ""
    return Path(path_str).name


def _select_best(rows: List[Dict[str, Any]], metric_key: str) -> Optional[Dict[str, Any]]:
    best = None
    best_val = None
    for r in rows:
        val = r.get(metric_key)
        if val is None:
            continue
        if best_val is None or val > best_val:
            best_val = val
            best = r
    return best


def _write_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]]) -> List[str]:
    lines = []
    header = "| " + " | ".join(c[0] for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        line = "| " + " | ".join(_fmt(r.get(key)) for _, key in columns) + " |"
        lines.append(line)
    return lines


def _parse_weights(spec: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            weights[part] = 1.0
            continue
        key, val = part.split("=", 1)
        try:
            weights[key.strip()] = float(val.strip())
        except Exception:
            weights[key.strip()] = 1.0
    return weights


def _score_row(row: Dict[str, Any], keys: Iterable[str], weights: Dict[str, float]) -> Optional[float]:
    total = 0.0
    total_w = 0.0
    for k in keys:
        w = float(weights.get(k, 1.0))
        v = row.get(k)
        if v is None:
            v = 0.0
        try:
            total += w * float(v)
            total_w += w
        except Exception:
            continue
    return (total / total_w) if total_w > 0 else None


def _load_trackb_rows(metrics_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("metrics_val_*_summary.json")):
        data = _load_json(path)
        if not data:
            continue
        metrics = data.get("metrics", {})
        train = metrics.get("train_config", {})
        flat = data.get("yaml_config_flat", {})
        row = {
            "run": path.name,
            "timestamp": metrics.get("timestamp"),
            "checkpoint": _basename(metrics.get("checkpoint")),
            "mAP": metrics.get("mAP"),
            "N_top5_mAP": metrics.get("N_top5_mAP"),
            "Nv_top5_mAP": metrics.get("Nv_top5_mAP"),
            "N_delta_top5_mAP": metrics.get("N_delta_top5_mAP"),
            "All_top5_mAP": metrics.get("All_top5_mAP"),
            "ttc_mae_seconds": metrics.get("ttc_mae_seconds"),
            "config_name": data.get("config_name") or train.get("config_name"),
            "video_backbone": train.get("video_backbone") or flat.get("model.tokenizer.video_backbone"),
            "tokens_root": train.get("tokens_root") or flat.get("data.tokens_root"),
            "loss_w_noun": train.get("loss_w_noun"),
            "loss_w_verb": train.get("loss_w_verb"),
            "class_weight_alpha": train.get("class_weight_alpha"),
            "use_class_weights": train.get("use_class_weights"),
        }
        rows.append(row)
    return rows


def _load_trackc_rows(metrics_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("trackC_val_rate*_summary.json")):
        data = _load_json(path)
        if not data:
            continue
        metrics = data.get("metrics", {})
        train = metrics.get("train_config", {})
        eval_cfg = data.get("eval_config", {})
        rgtp_cfg = data.get("rgtp_config", {})
        flat = data.get("yaml_config_flat", {})
        row = {
            "run": path.name,
            "timestamp": metrics.get("timestamp"),
            "checkpoint": _basename(metrics.get("checkpoint")),
            "mAP": metrics.get("mAP"),
            "N_top5_mAP": metrics.get("N_top5_mAP"),
            "Nv_top5_mAP": metrics.get("Nv_top5_mAP"),
            "N_delta_top5_mAP": metrics.get("N_delta_top5_mAP"),
            "All_top5_mAP": metrics.get("All_top5_mAP"),
            "ttc_mae_seconds": metrics.get("ttc_mae_seconds"),
            "rgtp_rate_request": metrics.get("rgtp_rate_request"),
            "rgtp_mean_fraction_pruned": metrics.get("rgtp_mean_fraction_pruned"),
            "pruning_enabled": metrics.get("rgtp_enabled"),
            "config_name": data.get("config_name"),
            "video_backbone": flat.get("model.tokenizer.video_backbone"),
            "tokens_root": _get_nested(eval_cfg, "tokens_root"),
            "use_class_weights": train.get("use_class_weights"),
            "class_weight_alpha": train.get("class_weight_alpha"),
        }
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize best Track B/Track C runs to markdown.")
    ap.add_argument("--trackb-metrics-dir", default="local_extraction/runs/Track_B/metrics")
    ap.add_argument("--trackc-metrics-dir", default="local_extraction/runs/Track_C/metrics")
    ap.add_argument("--trackb-metric", default="mAP")
    ap.add_argument("--trackc-metric", default="All_top5_mAP")
    ap.add_argument(
        "--score-weights",
        default="mAP=1,N_top5_mAP=1,Nv_top5_mAP=1,N_delta_top5_mAP=1,All_top5_mAP=1",
        help="Comma-separated weights for composite score (e.g., mAP=1,N_top5_mAP=1,All_top5_mAP=2).",
    )
    default_out = Path(__file__).resolve().parent / "best_runs_summary.md"
    ap.add_argument("--output", default=str(default_out))
    args = ap.parse_args()

    score_keys = ["mAP", "N_top5_mAP", "Nv_top5_mAP", "N_delta_top5_mAP", "All_top5_mAP"]
    score_weights = _parse_weights(args.score_weights)

    trackb_rows = [r for r in _load_trackb_rows(Path(args.trackb_metrics_dir)) if r.get("use_class_weights")]
    trackc_rows = [r for r in _load_trackc_rows(Path(args.trackc_metrics_dir)) if r.get("use_class_weights")]

    for r in trackb_rows:
        r["score"] = _score_row(r, score_keys, score_weights)
    for r in trackc_rows:
        r["score"] = _score_row(r, score_keys, score_weights)

    best_trackb = _select_best(trackb_rows, "score")
    best_trackc = _select_best(trackc_rows, "score")

    # Best Track C per rate
    best_trackc_per_rate: List[Dict[str, Any]] = []
    by_rate: Dict[str, List[Dict[str, Any]]] = {}
    for r in trackc_rows:
        rate = r.get("rgtp_rate_request")
        key = f"{rate:.3f}" if isinstance(rate, (int, float)) else "unknown"
        by_rate.setdefault(key, []).append(r)
    for rate_key in sorted(by_rate.keys()):
        best = _select_best(by_rate[rate_key], args.trackc_metric)
        if best:
            best_trackc_per_rate.append(best)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Best Track B/Track C Runs")
    lines.append("")
    lines.append("Selection metrics (weighted-class runs only):")
    lines.append("- Track B: composite score (mAP + N_top5 + Nv_top5 + N_delta_top5 + All_top5)")
    lines.append("- Track C: composite score (mAP + N_top5 + Nv_top5 + N_delta_top5 + All_top5)")
    lines.append(f"- Weights: `{args.score_weights}` (missing metrics count as 0)")
    lines.append("")

    lines.append("## Best Track B run")
    if best_trackb:
        lines.extend(_write_table([best_trackb], [
            ("run", "run"),
            ("checkpoint", "checkpoint"),
            ("score", "score"),
            ("mAP", "mAP"),
            ("N_top5_mAP", "N_top5_mAP"),
            ("Nv_top5_mAP", "Nv_top5_mAP"),
            ("N_delta_top5_mAP", "N_delta_top5_mAP"),
            ("All_top5_mAP", "All_top5_mAP"),
            ("ttc_mae_seconds", "ttc_mae_seconds"),
            ("video_backbone", "video_backbone"),
            ("tokens_root", "tokens_root"),
            ("loss_w_noun", "loss_w_noun"),
            ("loss_w_verb", "loss_w_verb"),
            ("class_weight_alpha", "class_weight_alpha"),
            ("use_class_weights", "use_class_weights"),
        ]))
    else:
        lines.append("No Track B metrics found.")
    lines.append("")

    lines.append("## Best Track C run (overall)")
    if best_trackc:
        lines.extend(_write_table([best_trackc], [
            ("run", "run"),
            ("checkpoint", "checkpoint"),
            ("score", "score"),
            ("mAP", "mAP"),
            ("N_top5_mAP", "N_top5_mAP"),
            ("Nv_top5_mAP", "Nv_top5_mAP"),
            ("N_delta_top5_mAP", "N_delta_top5_mAP"),
            ("All_top5_mAP", "All_top5_mAP"),
            ("ttc_mae_seconds", "ttc_mae_seconds"),
            ("rgtp_rate_request", "rgtp_rate_request"),
            ("rgtp_mean_fraction_pruned", "rgtp_mean_fraction_pruned"),
            ("pruning_enabled", "pruning_enabled"),
            ("video_backbone", "video_backbone"),
        ]))
    else:
        lines.append("No Track C metrics found.")
    lines.append("")

    lines.append("## Best Track C runs by pruning rate")
    if best_trackc_per_rate:
        lines.extend(_write_table(best_trackc_per_rate, [
            ("run", "run"),
            ("checkpoint", "checkpoint"),
            ("score", "score"),
            ("All_top5_mAP", "All_top5_mAP"),
            ("rgtp_rate_request", "rgtp_rate_request"),
            ("rgtp_mean_fraction_pruned", "rgtp_mean_fraction_pruned"),
            ("mAP", "mAP"),
            ("ttc_mae_seconds", "ttc_mae_seconds"),
        ]))
    else:
        lines.append("No Track C metrics found.")
    lines.append("")

    lines.append("## All Track B runs (summary)")
    if trackb_rows:
        lines.extend(_write_table(trackb_rows, [
            ("run", "run"),
            ("checkpoint", "checkpoint"),
            ("score", "score"),
            ("mAP", "mAP"),
            ("N_top5_mAP", "N_top5_mAP"),
            ("Nv_top5_mAP", "Nv_top5_mAP"),
            ("N_delta_top5_mAP", "N_delta_top5_mAP"),
            ("All_top5_mAP", "All_top5_mAP"),
            ("video_backbone", "video_backbone"),
            ("tokens_root", "tokens_root"),
        ]))
    else:
        lines.append("No Track B metrics found.")
    lines.append("")

    lines.append("## All Track C runs (summary)")
    if trackc_rows:
        lines.extend(_write_table(trackc_rows, [
            ("run", "run"),
            ("checkpoint", "checkpoint"),
            ("score", "score"),
            ("All_top5_mAP", "All_top5_mAP"),
            ("rgtp_rate_request", "rgtp_rate_request"),
            ("rgtp_mean_fraction_pruned", "rgtp_mean_fraction_pruned"),
            ("mAP", "mAP"),
            ("ttc_mae_seconds", "ttc_mae_seconds"),
        ]))
    else:
        lines.append("No Track C metrics found.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
