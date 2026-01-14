"""
Run class collapse analysis for both ResNet18 and VideoMAE
Compares weighted vs unweighted for each backbone separately.
"""

import subprocess
import sys
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
PREDICTIONS_DIR = REPO_ROOT / "local_extraction/runs/Track_B/predictions"
SCRIPT = REPO_ROOT / "local_extraction/trackB/class_collapse_analysis.py"

# ResNet18 predictions (already identified)
RESNET18_WEIGHTED = PREDICTIONS_DIR / "predictions_val_20251226_231700.csv"
RESNET18_UNWEIGHTED = PREDICTIONS_DIR / "predictions_val_20251222_175251.csv"

# VideoMAE predictions
VIDEOMAE_WEIGHTED = PREDICTIONS_DIR / "predictions_val_20260102_125317.csv"  # Jan 2, 2026, with class weights
# Need to find unweighted VideoMAE - checking available predictions

print("=" * 60)
print("Running Collapse Analysis Comparisons")
print("=" * 60)

# Check which prediction files exist
print("\nChecking prediction files:")
print(f"ResNet18 Weighted: {RESNET18_WEIGHTED.exists()} - {RESNET18_WEIGHTED.name}")
print(f"ResNet18 Unweighted: {RESNET18_UNWEIGHTED.exists()} - {RESNET18_UNWEIGHTED.name}")
print(f"VideoMAE Weighted: {VIDEOMAE_WEIGHTED.exists()} - {VIDEOMAE_WEIGHTED.name}")

# Find VideoMAE unweighted predictions by checking metadata
# From the analysis doc, VideoMAE unweighted had 33.46% mAP (Dec 23)
# Let's check the run_categorization results
import json

backbone_data_file = REPO_ROOT / "local_extraction/final_scripts/comparison_results/backbone_metrics_comparison_data.json"
if backbone_data_file.exists():
    with open(backbone_data_file) as f:
        data = json.load(f)
    print(f"\nVideoMAE mAP from comparison: {data['videomae']['metrics']['mAP']:.2f}%")
    videomae_map = data['videomae']['metrics']['mAP']
    
# Search for VideoMAE predictions by checking which ones have ~31% mAP
# The analysis says the Dec 23 unweighted run achieved 33.46% mAP
# But the thesis uses 31.19% - let me find which prediction file that corresponds to

# For now, let's try to identify VideoMAE predictions by looking at recent runs
# VideoMAE runs would be from Dec 2025 or Jan 2026

# Let me check the metrics files to find the corresponding predictions
metrics_dir = REPO_ROOT / "local_extraction/runs/Track_B/metrics"
print(f"\nSearching metrics files for VideoMAE runs...")

videomae_runs = []
for metrics_file in sorted(metrics_dir.glob("*_summary.json")):
    try:
        with open(metrics_file) as f:
            data = json.load(f)
        
        train_config = data.get("metrics", {}).get("train_config", {})
        video_backbone = train_config.get("video_backbone", "")
        use_class_weights = train_config.get("use_class_weights", False)
        
        if "videomae" in video_backbone.lower():
            map_val = data.get("metrics", {}).get("mAP", 0)
            timestamp = metrics_file.stem.replace("metrics_val_", "").replace("_summary", "")
            
            videomae_runs.append({
                "file": metrics_file.name,
                "timestamp": timestamp,
                "mAP": map_val,
                "weighted": use_class_weights,
                "predictions": f"predictions_val_{timestamp}.csv"
            })
    except Exception as e:
        continue

print(f"\nFound {len(videomae_runs)} VideoMAE runs:")
for run in sorted(videomae_runs, key=lambda x: x["mAP"], reverse=True):
    weight_str = "WEIGHTED" if run["weighted"] else "unweighted"
    print(f"  {run['timestamp']}: mAP={run['mAP']:.2f}% ({weight_str}) - {run['predictions']}")

if len(videomae_runs) >= 2:
    # Find best weighted and best unweighted
    weighted_runs = [r for r in videomae_runs if r["weighted"]]
    unweighted_runs = [r for r in videomae_runs if not r["weighted"]]
    
    if weighted_runs and unweighted_runs:
        best_weighted = max(weighted_runs, key=lambda x: x["mAP"])
        best_unweighted = max(unweighted_runs, key=lambda x: x["mAP"])
        
        VIDEOMAE_WEIGHTED = PREDICTIONS_DIR / best_weighted["predictions"]
        VIDEOMAE_UNWEIGHTED = PREDICTIONS_DIR / best_unweighted["predictions"]
        
        print(f"\nSelected VideoMAE runs:")
        print(f"  Weighted: {best_weighted['timestamp']} (mAP={best_weighted['mAP']:.2f}%)")
        print(f"  Unweighted: {best_unweighted['timestamp']} (mAP={best_unweighted['mAP']:.2f}%)")
    else:
        print("\n⚠️  Could not find both weighted and unweighted VideoMAE runs!")
        VIDEOMAE_UNWEIGHTED = None
else:
    print("\n⚠️  Not enough VideoMAE runs found!")
    VIDEOMAE_UNWEIGHTED = None

# Run analyses
print("\n" + "=" * 60)
print("Running Analysis 1: ResNet18 (Weighted vs Unweighted)")
print("=" * 60)

cmd1 = [
    sys.executable, str(SCRIPT),
    "--weighted_csv", str(RESNET18_WEIGHTED),
    "--unweighted_csv", str(RESNET18_UNWEIGHTED),
    "--output_dir", str(REPO_ROOT / "local_extraction/runs/Track_B/collapse_analysis"),
    "--prefix", "resnet18"
]
subprocess.run(cmd1)

if VIDEOMAE_UNWEIGHTED and VIDEOMAE_UNWEIGHTED.exists():
    print("\n" + "=" * 60)
    print("Running Analysis 2: VideoMAE (Weighted vs Unweighted)")
    print("=" * 60)
    
    cmd2 = [
        sys.executable, str(SCRIPT),
        "--weighted_csv", str(VIDEOMAE_WEIGHTED),
        "--unweighted_csv", str(VIDEOMAE_UNWEIGHTED),
        "--output_dir", str(REPO_ROOT / "local_extraction/runs/Track_B/collapse_analysis_videomae"),
        "--prefix", "videomae"
    ]
    subprocess.run(cmd2)
else:
    print("\n⚠️  Skipping VideoMAE analysis - unweighted predictions not found")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
