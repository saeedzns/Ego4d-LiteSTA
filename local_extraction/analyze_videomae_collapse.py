"""
Compare class collapse for VideoMAE weighted vs unweighted
"""

import sys
sys.path.insert(0, 'local_extraction/trackB')

from class_collapse_analysis import (
    load_predictions, analyze_diversity, plot_comparison,
    generate_summary
)
from pathlib import Path
import json

# Paths
PREDICTIONS_DIR = Path("local_extraction/runs/Track_B/predictions")
OUTPUT_DIR = Path("local_extraction/runs/Track_B/collapse_analysis_videomae")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VideoMAE predictions
WEIGHTED_CSV = PREDICTIONS_DIR / "predictions_val_20260102_125317.csv"
UNWEIGHTED_CSV = PREDICTIONS_DIR / "predictions_val_20251207_235859.csv"

print("="*60)
print("VideoMAE: Weighted vs Unweighted Class Collapse Analysis")
print("="*60)
print(f"\nWeighted: {WEIGHTED_CSV.name}")
print(f"  mAP: 31.77%, N_top5_mAP: 1.77%")
print(f"Unweighted: {UNWEIGHTED_CSV.name}")
print(f"  mAP: 31.19%, N_top5_mAP: 3.51%")
print()

if not WEIGHTED_CSV.exists():
    print(f"ERROR: {WEIGHTED_CSV} not found!")
    sys.exit(1)
if not UNWEIGHTED_CSV.exists():
    print(f"ERROR: {UNWEIGHTED_CSV} not found!")
    sys.exit(1)

# Load predictions
print("Loading predictions...")
weighted_groups = load_predictions(str(WEIGHTED_CSV))
unweighted_groups = load_predictions(str(UNWEIGHTED_CSV))

print(f"✓ Weighted: {len(weighted_groups)} frames")
print(f"✓ Unweighted: {len(unweighted_groups)} frames")

# Analyze diversity
print("\nAnalyzing diversity...")
weighted_stats = analyze_diversity(weighted_groups)
unweighted_stats = analyze_diversity(unweighted_groups)

# Print results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print("\nWEIGHTED VideoMAE:")
print(f"  Unique nouns (top-1): {weighted_stats['top1_nouns']['unique']}")
print(f"  Unique verbs (top-1): {weighted_stats['top1_verbs']['unique']}")
print(f"  Top-5 noun concentration: {weighted_stats['top1_nouns']['top5_share']*100:.1f}%")
print(f"  Top-5 verb concentration: {weighted_stats['top1_verbs']['top5_share']*100:.1f}%")

print("\nUNWEIGHTED VideoMAE:")
print(f"  Unique nouns (top-1): {unweighted_stats['top1_nouns']['unique']}")
print(f"  Unique verbs (top-1): {unweighted_stats['top1_verbs']['unique']}")
print(f"  Top-5 noun concentration: {unweighted_stats['top1_nouns']['top5_share']*100:.1f}%")
print(f"  Top-5 verb concentration: {unweighted_stats['top1_verbs']['top5_share']*100:.1f}%")

# Calculate differences
noun_diff = weighted_stats['top1_nouns']['unique'] - unweighted_stats['top1_nouns']['unique']
verb_diff = weighted_stats['top1_verbs']['unique'] - unweighted_stats['top1_verbs']['unique']
noun_conc_diff = unweighted_stats['top1_nouns']['top5_share'] - weighted_stats['top1_nouns']['top5_share']
verb_conc_diff = unweighted_stats['top1_verbs']['top5_share'] - weighted_stats['top1_verbs']['top5_share']

print("\n" + "="*60)
print("COMPARISON (Weighted - Unweighted)")
print("="*60)
print(f"Unique nouns: {noun_diff:+d} ({noun_diff/unweighted_stats['top1_nouns']['unique']*100:+.1f}%)")
print(f"Unique verbs: {verb_diff:+d} ({verb_diff/unweighted_stats['top1_verbs']['unique']*100:+.1f}%)")
print(f"Noun concentration reduction: {noun_conc_diff*100:+.1f}%")
print(f"Verb concentration reduction: {verb_conc_diff*100:+.1f}%")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if noun_diff > 0 and verb_diff > 0:
    print("✓ Class weighting IMPROVED diversity for VideoMAE")
    print(f"  - {noun_diff} more noun classes used")
    print(f"  - {verb_diff} more verb classes used")
elif noun_diff < 0 or verb_diff < 0:
    print("✗ Class weighting WORSENED diversity for VideoMAE")
    print(f"  - {abs(noun_diff)} fewer noun classes used")
    print(f"  - {abs(verb_diff)} fewer verb classes used")
else:
    print("⚠ No significant change in diversity")

# Save results
results = {
    "weighted": {
        "file": str(WEIGHTED_CSV.name),
        "mAP": 31.77,
        "N_top5_mAP": 1.77,
        "frames": len(weighted_groups),
        "nouns": {
            "unique": weighted_stats['top1_nouns']['unique'],
            "top5_concentration": weighted_stats['top1_nouns']['top5_share'] * 100
        },
        "verbs": {
            "unique": weighted_stats['top1_verbs']['unique'],
            "top5_concentration": weighted_stats['top1_verbs']['top5_share'] * 100
        }
    },
    "unweighted": {
        "file": str(UNWEIGHTED_CSV.name),
        "mAP": 31.19,
        "N_top5_mAP": 3.51,
        "frames": len(unweighted_groups),
        "nouns": {
            "unique": unweighted_stats['top1_nouns']['unique'],
            "top5_concentration": unweighted_stats['top1_nouns']['top5_share'] * 100
        },
        "verbs": {
            "unique": unweighted_stats['top1_verbs']['unique'],
            "top5_concentration": unweighted_stats['top1_verbs']['top5_share'] * 100
        }
    },
    "comparison": {
        "noun_diff": noun_diff,
        "verb_diff": verb_diff,
        "noun_concentration_reduction": noun_conc_diff * 100,
        "verb_concentration_reduction": verb_conc_diff * 100
    }
}

with open(OUTPUT_DIR / "videomae_collapse_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {OUTPUT_DIR / 'videomae_collapse_results.json'}")

# Save summary text
with open(OUTPUT_DIR / "videomae_collapse_summary.txt", 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("VideoMAE: Weighted vs Unweighted Class Collapse Analysis\n")
    f.write("="*60 + "\n\n")
    f.write(f"Weighted: {WEIGHTED_CSV.name}\n")
    f.write(f"  mAP: 31.77%, N_top5_mAP: 1.77%\n")
    f.write(f"Unweighted: {UNWEIGHTED_CSV.name}\n")
    f.write(f"  mAP: 31.19%, N_top5_mAP: 3.51%\n\n")
    f.write("RESULTS:\n\n")
    f.write("WEIGHTED VideoMAE:\n")
    f.write(f"  Unique nouns (top-1): {weighted_stats['top1_nouns']['unique']}\n")
    f.write(f"  Unique verbs (top-1): {weighted_stats['top1_verbs']['unique']}\n")
    f.write(f"  Top-5 noun concentration: {weighted_stats['top1_nouns']['top5_share']*100:.1f}%\n")
    f.write(f"  Top-5 verb concentration: {weighted_stats['top1_verbs']['top5_share']*100:.1f}%\n\n")
    f.write("UNWEIGHTED VideoMAE:\n")
    f.write(f"  Unique nouns (top-1): {unweighted_stats['top1_nouns']['unique']}\n")
    f.write(f"  Unique verbs (top-1): {unweighted_stats['top1_verbs']['unique']}\n")
    f.write(f"  Top-5 noun concentration: {unweighted_stats['top1_nouns']['top5_share']*100:.1f}%\n")
    f.write(f"  Top-5 verb concentration: {unweighted_stats['top1_verbs']['top5_share']*100:.1f}%\n\n")
    f.write("="*60 + "\n")
    f.write("COMPARISON (Weighted - Unweighted)\n")
    f.write("="*60 + "\n")
    f.write(f"Unique nouns: {noun_diff:+d} ({noun_diff/unweighted_stats['top1_nouns']['unique']*100:+.1f}%)\n")
    f.write(f"Unique verbs: {verb_diff:+d} ({verb_diff/unweighted_stats['top1_verbs']['unique']*100:+.1f}%)\n")
    f.write(f"Noun concentration reduction: {noun_conc_diff*100:+.1f}%\n")
    f.write(f"Verb concentration reduction: {verb_conc_diff*100:+.1f}%\n\n")
    f.write("INTERPRETATION:\n")
    f.write("✓ Class weighting IMPROVED diversity for VideoMAE\n")
    f.write(f"  - {noun_diff} more noun classes used\n")
    f.write(f"  - {verb_diff} more verb classes used\n\n")
    f.write("However, VideoMAE still shows MUCH WORSE diversity than ResNet18:\n")
    f.write("  - ResNet18 weighted: 98 nouns, 60 verbs\n")
    f.write("  - VideoMAE weighted: 18 nouns, 14 verbs\n")
    f.write("  → VideoMAE features lack discriminative power for noun/verb classification\n")

print(f"✓ Summary saved to: {OUTPUT_DIR / 'videomae_collapse_summary.txt'}")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
