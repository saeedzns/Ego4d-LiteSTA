import json
from pathlib import Path

metrics_dir = Path("local_extraction/runs/Track_B/metrics")

# VideoMAE weighted
f1 = metrics_dir / "metrics_val_20260102_125317_summary.json"
if f1.exists():
    with open(f1) as f:
        d = json.load(f)
    print(f"VideoMAE Weighted (20260102_125317):")
    print(f"  mAP: {d['metrics']['mAP']*100:.2f}%")
    print(f"  N_top5_mAP: {d['metrics']['N_top5_mAP']*100:.2f}%")
    print(f"  use_class_weights: {d['metrics']['train_config'].get('use_class_weights', 'N/A')}")

# VideoMAE unweighted  
f2 = metrics_dir / "metrics_val_20251207_235859_summary.json"
if f2.exists():
    with open(f2) as f:
        d2 = json.load(f)
    print(f"\nVideoMAE Unweighted (20251207_235859):")
    print(f"  mAP: {d2['metrics']['mAP']*100:.2f}%")
    print(f"  N_top5_mAP: {d2['metrics']['N_top5_mAP']*100:.2f}%")
    print(f"  use_class_weights: {d2['metrics']['train_config'].get('use_class_weights', 'N/A')}")
