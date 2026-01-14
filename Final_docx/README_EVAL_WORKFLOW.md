# Thesis Evaluation Workflow (Track B/C + Error Analysis)

This file is a practical checklist to (1) re-run evaluation for a specific Track‑B checkpoint, (2) generate comparable error-analysis artifacts, and (3) decide whether the “weighted checkpoint baseline” claim in `Final_docx/thesis_fixed.md` is supported by evidence.

## What “evaluate it” means (for the checkpoint choice)

You can justify choosing the **weighted** checkpoint even if some benchmark metrics are lower, but only if you show evidence for the specific claim you make:

- **If you claim “best benchmark performance”** → compare top‑5 metrics (N, N+V, N+δ, All).
- **If you claim “less class collapse / less frequent-class bias”** → compare prediction diversity + per-class failures.
- **If you claim “helps rare classes”** → compare per-class stats (zero-accuracy classes, worst/best classes, confusions).

This workflow produces all 3 kinds of evidence.

---

## 1) Run Track‑B evaluation for a specific checkpoint

### Option A (recommended): evaluate a known checkpoint directly

**PowerShell (Windows venv):**
```powershell
cd D:\Thesis\Ego4d-LiteSTA
.\local_extraction\.venv\Scripts\Activate.ps1
python local_extraction/trackB/trackB_eval.py --config trackB --checkpoint "local_extraction/runs/Track_B/checkpoints/trackB_best_mAP_0.3708_20251225_224220.pt"
```

**Bash (WSL / Linux):**
```bash
cd /mnt/d/Thesis/Ego4d-LiteSTA
python3 local_extraction/trackB/trackB_eval.py --config trackB --checkpoint "local_extraction/runs/Track_B/checkpoints/trackB_best_mAP_0.3708_20251225_224220.pt"
```

Outputs are written to:
- `local_extraction/runs/Track_B/metrics/metrics_val_<timestamp>.json` and `_summary.json`
- `local_extraction/runs/Track_B/predictions/predictions_val_<timestamp>.csv` and `.jsonl`
- `local_extraction/runs/Track_B/overlays/val/` (if enabled)

### Option B: evaluate “latest/final”

```bash
python3 local_extraction/trackB/trackB_eval.py --config trackB
```

This uses the latest `trackB_final_*.pt` if `--checkpoint` is not provided.

---

## 2) Run error analysis for a chosen eval (metrics + predictions)

Run error analysis into a *separate* folder per checkpoint so outputs don’t overwrite each other.

**Unweighted example (0.3904):**
```bash
python3 local_extraction/trackB/error_analysis.py \
  --metrics local_extraction/runs/Track_B/metrics/metrics_val_20251222_175251.json \
  --predictions local_extraction/runs/Track_B/predictions/predictions_val_20251222_175251.csv \
  --output_dir local_extraction/runs/Track_B/error_analysis_unweighted
```

**Weighted example (0.3708):**
```bash
python3 local_extraction/trackB/error_analysis.py \
  --metrics local_extraction/runs/Track_B/metrics/metrics_val_20251226_231700.json \
  --predictions local_extraction/runs/Track_B/predictions/predictions_val_20251226_231700.csv \
  --output_dir local_extraction/runs/Track_B/error_analysis_weighted
```

Key outputs to compare:
- `noun_worst_classes_data.json`, `verb_worst_classes_data.json`
- `noun_confusions_data.json`
- `failure_gallery/` + `failure_summary.txt`
- `success_gallery/` + `success_summary.txt`

---

## 3) Check “class collapse / frequent-class bias” quantitatively

This measures how concentrated the predictions are (how much the top classes dominate).

```bash
python3 - <<'PY'
import csv
from collections import Counter, defaultdict

def analyze(pred_csv):
    rows=[]
    with open(pred_csv, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    groups=defaultdict(list)
    for r in rows:
        groups[(r['uid'], r['frame_path'])].append(r)

    top1_n, top1_v, top5_n, top5_v = [], [], [], []
    for _, items in groups.items():
        items.sort(key=lambda x: float(x.get('score_final') or 0), reverse=True)
        top1=items[0]; top=items[:5]
        top1_n.append(int(top1['pred_noun_id'])); top1_v.append(int(top1['pred_verb_id']))
        for it in top:
            top5_n.append(int(it['pred_noun_id'])); top5_v.append(int(it['pred_verb_id']))

    def summ(lst):
        c=Counter(lst)
        total=len(lst)
        return dict(unique=len(c),
                    share_top5=sum(v for _,v in c.most_common(5))/total,
                    share_top10=sum(v for _,v in c.most_common(10))/total)

    return len(groups), summ(top1_n), summ(top1_v), summ(top5_n), summ(top5_v)

cases = [
    ("unweighted","local_extraction/runs/Track_B/predictions/predictions_val_20251222_175251.csv"),
    ("weighted","local_extraction/runs/Track_B/predictions/predictions_val_20251226_231700.csv"),
]

for name, path in cases:
    frames, t1n, t1v, t5n, t5v = analyze(path)
    print("\n",name,"frames",frames)
    print(" top1 noun:", t1n)
    print(" top1 verb:", t1v)
    print(" top5 noun:", t5n)
    print(" top5 verb:", t5v)
PY
```

How to interpret:
- **Higher `unique`** and **lower `share_top5/share_top10`** → less collapse (model is not always predicting the same few classes).
- If this improves for weighted while benchmark metrics drop, your thesis can argue “more robust semantics / less frequent-class domination”, but you must state the trade-off clearly.

---

## 4) “Connect the dots” inside `Final_docx/thesis_fixed.md`

When you write the selection rationale, it should link to concrete evidence:

- **Metrics evidence:** cite the exact `_summary.json` numbers for top‑5 metrics (N/N+V/N+δ/All).
- **Collapse evidence:** cite the diversity stats (unique predictions + concentration).
- **Error-analysis evidence:** cite the worst/best classes and top confusions from the error-analysis outputs.

If your thesis claims “weighted baseline used throughout”, then tables in Chapter 8 must use the weighted numbers (or explicitly state where unweighted numbers are reported for reference).

---

## TODO (next actions)

### A) Decide your thesis baseline story (pick one)
- **Option 1 (performance baseline):** use unweighted `0.3904` for headline Track‑B results; use weighted only for “methodology improvement” discussion.
- **Option 2 (methodology baseline):** use weighted `0.3708` everywhere for Track‑B/Track‑C comparisons; explicitly state benchmark trade-off.

### B) Fix internal consistency (must-do)
- Update any Track‑B result tables so the numbers match the checkpoint you claim they come from.
- Replace vague wording like “slightly lower” if the drop is large (e.g., All top‑5).

### C) Add one “proof paragraph” for the selection rationale
- 1–2 sentences citing the diversity/concentration stats and 1–2 sentences citing the error analysis (top confusions + zero-accuracy classes).

### D) Re-run artifacts if needed
- Re-run `trackB_eval.py` for the exact checkpoint(s) you want to cite in the final PDF.
- Re-run `error_analysis.py` into separate output folders (weighted vs unweighted) so you can cite consistent file paths.

### E) Final doc build
- Rebuild docx from `Final_docx/thesis_fixed.md` after edits (see `Final_docx/README.md`).

