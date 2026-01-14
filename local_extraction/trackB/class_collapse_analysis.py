"""
Class Collapse Analysis for Track B Checkpoints
Compares prediction diversity between weighted and unweighted checkpoints
to validate the "prevents class collapse" claim.

Usage:
    python class_collapse_analysis.py
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configuration
PREDICTIONS = {
    "weighted": "local_extraction/runs/Track_B/predictions/predictions_val_20251226_231700.csv",
    "unweighted": "local_extraction/runs/Track_B/predictions/predictions_val_20251222_175251.csv"
}
OUTPUT_DIR = Path("local_extraction/runs/Track_B/collapse_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_predictions(pred_csv):
    """Load and group predictions by frame."""
    rows = []
    with open(pred_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    
    # Group by frame
    groups = defaultdict(list)
    for row in rows:
        key = (row['uid'], row['frame_path'])
        groups[key].append(row)
    
    return groups

def analyze_diversity(groups):
    """Analyze prediction diversity metrics."""
    top1_nouns = []
    top1_verbs = []
    top5_nouns = []
    top5_verbs = []
    
    for frame_candidates in groups.values():
        # Sort by score
        sorted_items = sorted(
            frame_candidates, 
            key=lambda x: float(x.get('score_final', 0)), 
            reverse=True
        )
        
        # Top-1 prediction
        if sorted_items:
            top1 = sorted_items[0]
            top1_nouns.append(int(top1['pred_noun_id']))
            top1_verbs.append(int(top1['pred_verb_id']))
        
        # Top-5 predictions
        for item in sorted_items[:5]:
            top5_nouns.append(int(item['pred_noun_id']))
            top5_verbs.append(int(item['pred_verb_id']))
    
    def compute_stats(id_list):
        counter = Counter(id_list)
        total = len(id_list)
        unique = len(counter)
        
        # Top-K concentration
        top5_count = sum(count for _, count in counter.most_common(5))
        top10_count = sum(count for _, count in counter.most_common(10))
        
        return {
            'unique': unique,
            'total': total,
            'top5_share': top5_count / total if total > 0 else 0,
            'top10_share': top10_count / total if total > 0 else 0,
            'distribution': counter
        }
    
    return {
        'frames': len(groups),
        'top1_nouns': compute_stats(top1_nouns),
        'top1_verbs': compute_stats(top1_verbs),
        'top5_nouns': compute_stats(top5_nouns),
        'top5_verbs': compute_stats(top5_verbs)
    }

def plot_comparison(results):
    """Generate comparison plots."""
    
    # Plot 1: Unique classes comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Class Collapse Analysis: Weighted vs Unweighted', fontsize=16, fontweight='bold')
    
    # 1.1 Unique classes (nouns)
    ax = axes[0, 0]
    categories = ['Top-1 Predictions', 'Top-5 Predictions']
    weighted_vals = [
        results['weighted']['top1_nouns']['unique'],
        results['weighted']['top5_nouns']['unique']
    ]
    unweighted_vals = [
        results['unweighted']['top1_nouns']['unique'],
        results['unweighted']['top5_nouns']['unique']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, weighted_vals, width, label='Weighted', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, unweighted_vals, width, label='Unweighted', color='#e74c3c', alpha=0.8)
    ax.set_ylabel('Unique Noun Classes', fontweight='bold')
    ax.set_title('Noun Class Diversity')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(weighted_vals):
        ax.text(i - width/2, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(unweighted_vals):
        ax.text(i + width/2, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 1.2 Unique classes (verbs)
    ax = axes[0, 1]
    weighted_vals = [
        results['weighted']['top1_verbs']['unique'],
        results['weighted']['top5_verbs']['unique']
    ]
    unweighted_vals = [
        results['unweighted']['top1_verbs']['unique'],
        results['unweighted']['top5_verbs']['unique']
    ]
    
    ax.bar(x - width/2, weighted_vals, width, label='Weighted', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, unweighted_vals, width, label='Unweighted', color='#e74c3c', alpha=0.8)
    ax.set_ylabel('Unique Verb Classes', fontweight='bold')
    ax.set_title('Verb Class Diversity')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(weighted_vals):
        ax.text(i - width/2, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(unweighted_vals):
        ax.text(i + width/2, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2.1 Top-5 concentration (nouns)
    ax = axes[1, 0]
    categories = ['Top-1 Preds', 'Top-5 Preds']
    weighted_vals = [
        results['weighted']['top1_nouns']['top5_share'] * 100,
        results['weighted']['top5_nouns']['top5_share'] * 100
    ]
    unweighted_vals = [
        results['unweighted']['top1_nouns']['top5_share'] * 100,
        results['unweighted']['top5_nouns']['top5_share'] * 100
    ]
    
    ax.bar(x - width/2, weighted_vals, width, label='Weighted', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, unweighted_vals, width, label='Unweighted', color='#e74c3c', alpha=0.8)
    ax.set_ylabel('Top-5 Concentration (%)', fontweight='bold')
    ax.set_title('Noun Prediction Concentration (Lower = Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    for i, v in enumerate(weighted_vals):
        ax.text(i - width/2, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(unweighted_vals):
        ax.text(i + width/2, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2.2 Top-5 concentration (verbs)
    ax = axes[1, 1]
    weighted_vals = [
        results['weighted']['top1_verbs']['top5_share'] * 100,
        results['weighted']['top5_verbs']['top5_share'] * 100
    ]
    unweighted_vals = [
        results['unweighted']['top1_verbs']['top5_share'] * 100,
        results['unweighted']['top5_verbs']['top5_share'] * 100
    ]
    
    ax.bar(x - width/2, weighted_vals, width, label='Weighted', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, unweighted_vals, width, label='Unweighted', color='#e74c3c', alpha=0.8)
    ax.set_ylabel('Top-5 Concentration (%)', fontweight='bold')
    ax.set_title('Verb Prediction Concentration (Lower = Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    for i, v in enumerate(weighted_vals):
        ax.text(i - width/2, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    for i, v in enumerate(unweighted_vals):
        ax.text(i + width/2, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'collapse_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'collapse_comparison.png'}")
    
    # Plot 2: Distribution of top classes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Top-10 Most Predicted Classes', fontsize=14, fontweight='bold')
    
    # Nouns
    w_top10 = results['weighted']['top1_nouns']['distribution'].most_common(10)
    u_top10 = results['unweighted']['top1_nouns']['distribution'].most_common(10)
    
    y_pos = np.arange(10)
    ax1.barh(y_pos, [c for _, c in w_top10], alpha=0.7, label='Weighted', color='#2ecc71')
    ax1.barh(y_pos - 0.4, [c for _, c in u_top10], alpha=0.7, label='Unweighted', color='#e74c3c')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f'Noun {id}' for id, _ in w_top10])
    ax1.invert_yaxis()
    ax1.set_xlabel('Prediction Count')
    ax1.set_title('Top-10 Nouns (Top-1 Predictions)')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Verbs
    w_top10_v = results['weighted']['top1_verbs']['distribution'].most_common(10)
    u_top10_v = results['unweighted']['top1_verbs']['distribution'].most_common(10)
    
    ax2.barh(y_pos, [c for _, c in w_top10_v], alpha=0.7, label='Weighted', color='#2ecc71')
    ax2.barh(y_pos - 0.4, [c for _, c in u_top10_v], alpha=0.7, label='Unweighted', color='#e74c3c')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'Verb {id}' for id, _ in w_top10_v])
    ax2.invert_yaxis()
    ax2.set_xlabel('Prediction Count')
    ax2.set_title('Top-10 Verbs (Top-1 Predictions)')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_classes_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'top_classes_distribution.png'}")

def generate_summary(results):
    """Generate text summary and interpretation."""
    
    summary = []
    summary.append("=" * 80)
    summary.append("CLASS COLLAPSE ANALYSIS RESULTS")
    summary.append("=" * 80)
    summary.append("")
    
    for name in ['weighted', 'unweighted']:
        r = results[name]
        summary.append(f"\n{name.upper()} CHECKPOINT:")
        summary.append(f"  Frames analyzed: {r['frames']}")
        summary.append(f"  Top-1 Nouns: {r['top1_nouns']['unique']} unique (top-5 concentration: {r['top1_nouns']['top5_share']*100:.1f}%)")
        summary.append(f"  Top-1 Verbs: {r['top1_verbs']['unique']} unique (top-5 concentration: {r['top1_verbs']['top5_share']*100:.1f}%)")
        summary.append(f"  Top-5 Nouns: {r['top5_nouns']['unique']} unique (top-5 concentration: {r['top5_nouns']['top5_share']*100:.1f}%)")
        summary.append(f"  Top-5 Verbs: {r['top5_verbs']['unique']} unique (top-5 concentration: {r['top5_verbs']['top5_share']*100:.1f}%)")
    
    # Comparison
    summary.append("\n" + "=" * 80)
    summary.append("COMPARATIVE ANALYSIS")
    summary.append("=" * 80)
    
    w = results['weighted']
    u = results['unweighted']
    
    noun_diff = w['top1_nouns']['unique'] - u['top1_nouns']['unique']
    verb_diff = w['top1_verbs']['unique'] - u['top1_verbs']['unique']
    
    noun_conc_diff = (u['top1_nouns']['top5_share'] - w['top1_nouns']['top5_share']) * 100
    verb_conc_diff = (u['top1_verbs']['top5_share'] - w['top1_verbs']['top5_share']) * 100
    
    summary.append(f"\nUnique Classes (Weighted - Unweighted):")
    summary.append(f"  Nouns (top-1): {noun_diff:+d} ({noun_diff/u['top1_nouns']['unique']*100:+.1f}%)")
    summary.append(f"  Verbs (top-1): {verb_diff:+d} ({verb_diff/u['top1_verbs']['unique']*100:+.1f}%)")
    
    summary.append(f"\nTop-5 Concentration Reduction (Unweighted - Weighted):")
    summary.append(f"  Nouns: {noun_conc_diff:+.1f}% {'(weighted MORE concentrated)' if noun_conc_diff < 0 else '(weighted LESS concentrated)'}")
    summary.append(f"  Verbs: {verb_conc_diff:+.1f}% {'(weighted MORE concentrated)' if verb_conc_diff < 0 else '(weighted LESS concentrated)'}")
    
    # Interpretation
    summary.append("\n" + "=" * 80)
    summary.append("INTERPRETATION")
    summary.append("=" * 80)
    
    if noun_diff > 0 and noun_conc_diff > 0:
        summary.append("\n✓ CLAIM SUPPORTED: Weighted checkpoint shows BETTER diversity:")
        summary.append(f"  - Uses {noun_diff} more unique noun classes")
        summary.append(f"  - {noun_conc_diff:.1f}% lower concentration in top-5 classes")
        summary.append("  → Class weighting DOES reduce collapse to frequent classes")
    elif noun_diff < 0 or noun_conc_diff < 0:
        summary.append("\n✗ CLAIM NOT SUPPORTED: Weighted checkpoint does NOT show better diversity:")
        if noun_diff < 0:
            summary.append(f"  - Uses {abs(noun_diff)} FEWER unique noun classes")
        if noun_conc_diff < 0:
            summary.append(f"  - {abs(noun_conc_diff):.1f}% HIGHER concentration (more collapsed)")
        summary.append("  → Class weighting claim needs revision")
    else:
        summary.append("\n≈ MIXED RESULTS: No clear difference in prediction diversity")
        summary.append("  → Class weighting impact is minimal or inconsistent")
    
    summary_text = "\n".join(summary)
    
    # Save to file
    with open(OUTPUT_DIR / 'collapse_analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print("\n" + summary_text)
    print(f"\n✓ Saved: {OUTPUT_DIR / 'collapse_analysis_summary.txt'}")
    
    return summary_text

def main():
    print("Starting Class Collapse Analysis...")
    print("=" * 80)
    
    results = {}
    
    for name, pred_file in PREDICTIONS.items():
        pred_path = Path(pred_file)
        if not pred_path.exists():
            print(f"✗ ERROR: {pred_file} not found")
            continue
        
        print(f"\nAnalyzing {name} checkpoint...")
        print(f"  File: {pred_file}")
        
        groups = load_predictions(pred_file)
        results[name] = analyze_diversity(groups)
        
        print(f"  ✓ Analyzed {results[name]['frames']} frames")
        print(f"  ✓ Unique nouns (top-1): {results[name]['top1_nouns']['unique']}")
        print(f"  ✓ Unique verbs (top-1): {results[name]['top1_verbs']['unique']}")
    
    if len(results) == 2:
        print("\n" + "=" * 80)
        print("Generating comparison plots...")
        plot_comparison(results)
        
        print("\nGenerating summary...")
        generate_summary(results)
        
        # Save JSON
        json_results = {
            name: {
                'frames': r['frames'],
                'top1_nouns': {k: v for k, v in r['top1_nouns'].items() if k != 'distribution'},
                'top1_verbs': {k: v for k, v in r['top1_verbs'].items() if k != 'distribution'},
                'top5_nouns': {k: v for k, v in r['top5_nouns'].items() if k != 'distribution'},
                'top5_verbs': {k: v for k, v in r['top5_verbs'].items() if k != 'distribution'}
            }
            for name, r in results.items()
        }
        
        with open(OUTPUT_DIR / 'collapse_analysis_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Saved: {OUTPUT_DIR / 'collapse_analysis_results.json'}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
