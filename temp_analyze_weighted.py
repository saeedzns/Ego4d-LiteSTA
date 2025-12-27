import json
from pathlib import Path
import pandas as pd

# Load TSV
tsv_path = Path('local_extraction/runs/Track_B/plots/trackB_metrics_summary.tsv')
df = pd.read_csv(tsv_path, sep='\t')

# Load recent summary JSONs and find class-weighted runs
metrics_dir = Path('local_extraction/runs/Track_B/metrics')
files = sorted(metrics_dir.glob('*summary.json'), reverse=True)

class_weighted_runs = []
for f in files:
    data = json.loads(f.read_text())
    train_cfg = data.get('metrics', {}).get('train_config', data.get('train_config', {}))
    
    use_class_weights = train_cfg.get('use_class_weights', False)
    if not use_class_weights:
        continue
        
    # Extract checkpoint name from summary
    checkpoint = train_cfg.get('checkpoint_path', '').split('/')[-1].replace('.pt', '')
    if not checkpoint:
        # Try to infer from filename
        checkpoint = f.name.replace('_summary.json', '')
    
    class_weight_alpha = train_cfg.get('class_weight_alpha', 'N/A')
    loss_noun = train_cfg.get('loss_w_noun', 1.0)
    loss_verb = train_cfg.get('loss_w_verb', 1.0)
    
    # Get metrics
    metrics = data.get('metrics', data)
    mAP = metrics.get('mAP', 0) * 100
    accuracy = metrics.get('accuracy', 0) * 100
    
    class_weighted_runs.append({
        'summary_file': f.name,
        'checkpoint': checkpoint,
        'alpha': class_weight_alpha,
        'loss_noun': loss_noun,
        'loss_verb': loss_verb,
        'accuracy': accuracy,
        'mAP': mAP
    })

print('=== Class-Weighted Runs from Summary JSONs ===')
for run in class_weighted_runs:
    print(f"{run['summary_file']}")
    print(f"  Checkpoint: {run['checkpoint']}")
    print(f"  Alpha: {run['alpha']}, Loss: {run['loss_noun']}:{run['loss_verb']}")
    print(f"  Accuracy: {run['accuracy']:.2f}%, mAP: {run['mAP']:.2f}%")
    print()

# Map summary files to TSV by loading checkpoint from each summary
print('\n=== Mapping Class-Weighted Runs to TSV ===')
summary_to_checkpoint = {}
for f in files:
    data = json.loads(f.read_text())
    metrics = data.get('metrics', data)
    checkpoint_full = metrics.get('checkpoint', '')
    if checkpoint_full:
        checkpoint_name = checkpoint_full.split('\\')[-1].split('/')[-1]
        summary_to_checkpoint[f.name] = checkpoint_name

# Find these checkpoints in TSV
weighted_indices = []
print('\n=== Matching TSV Entries for Class-Weighted Runs ===')
for run in class_weighted_runs:
    checkpoint_name = summary_to_checkpoint.get(run['summary_file'], '')
    if checkpoint_name:
        matches = df[df['checkpoint'] == checkpoint_name]
        if len(matches) > 0:
            print(f"\nCheckpoint: {checkpoint_name}")
            print(f"  Config: Alpha={run['alpha']}, Loss={run['loss_noun']}:{run['loss_verb']}")
            for idx, row in matches.iterrows():
                weighted_indices.append(idx)
                print(f"  Row {idx}: timestamp={row['timestamp']}")
                print(f"    accuracy={row['accuracy']:.4f}, mAP={row['mAP']:.4f}")
                print(f"    N_mAP={row['N_mAP']:.4f}, Nv_mAP={row['Nv_mAP']:.4f}, N_delta_mAP={row['N_delta_mAP']:.4f}")
                print(f"    All_top5_mAP={row['All_top5_mAP']:.4f}")
                print(f"    N_top5_mAP={row['N_top5_mAP']:.4f}, Nv_top5_mAP={row['Nv_top5_mAP']:.4f}, N_delta_top5_mAP={row['N_delta_top5_mAP']:.4f}")
                print(f"    TTC MAE={row['ttc_mae_seconds']:.4f}s")

# Now calculate a balanced score for class-weighted runs
print('\n\n=== BEST CLASS-WEIGHTED RUN (Balanced Score) ===')

if weighted_indices:
    weighted_df = df.loc[weighted_indices].copy()
    
    # Calculate balanced score considering all metrics
    # Normalize each metric to 0-1 scale
    weighted_df['norm_N_mAP'] = weighted_df['N_mAP'] / weighted_df['N_mAP'].max()
    weighted_df['norm_Nv_mAP'] = weighted_df['Nv_mAP'] / weighted_df['Nv_mAP'].max()
    weighted_df['norm_N_delta_mAP'] = weighted_df['N_delta_mAP'] / weighted_df['N_delta_mAP'].max()
    weighted_df['norm_All_top5_mAP'] = weighted_df['All_top5_mAP'] / weighted_df['All_top5_mAP'].max()
    weighted_df['norm_N_top5_mAP'] = weighted_df['N_top5_mAP'] / weighted_df['N_top5_mAP'].max()
    weighted_df['norm_Nv_top5_mAP'] = weighted_df['Nv_top5_mAP'] / weighted_df['Nv_top5_mAP'].max()
    weighted_df['norm_N_delta_top5_mAP'] = weighted_df['N_delta_top5_mAP'] / weighted_df['N_delta_top5_mAP'].max()
    weighted_df['norm_accuracy'] = weighted_df['accuracy'] / weighted_df['accuracy'].max()
    weighted_df['norm_ttc'] = 1 - (weighted_df['ttc_mae_seconds'] / weighted_df['ttc_mae_seconds'].max())  # Lower is better
    
    # Balanced score: weight all top-5 metrics equally, plus accuracy and TTC
    weighted_df['balanced_score'] = (
        0.15 * weighted_df['norm_N_mAP'] +
        0.15 * weighted_df['norm_Nv_mAP'] +
        0.15 * weighted_df['norm_N_delta_mAP'] +
        0.15 * weighted_df['norm_All_top5_mAP'] +
        0.10 * weighted_df['norm_N_top5_mAP'] +
        0.10 * weighted_df['norm_Nv_top5_mAP'] +
        0.10 * weighted_df['norm_N_delta_top5_mAP'] +
        0.05 * weighted_df['norm_accuracy'] +
        0.05 * weighted_df['norm_ttc']
    )
    
    # Sort by balanced score
    best_runs = weighted_df.nlargest(3, 'balanced_score')
    
    print("\nTop 3 Class-Weighted Runs (by balanced score):\n")
    for idx, row in best_runs.iterrows():
        print(f"Rank {list(best_runs.index).index(idx) + 1}: Row {idx} - {row['checkpoint']}")
        print(f"  Timestamp: {row['timestamp']}")
        print(f"  Balanced Score: {row['balanced_score']:.4f}")
        print(f"  Accuracy: {row['accuracy']:.4f}, mAP: {row['mAP']:.4f}")
        print(f"  N_mAP: {row['N_mAP']:.4f}, Nv_mAP: {row['Nv_mAP']:.4f}, N_delta_mAP: {row['N_delta_mAP']:.4f}")
        print(f"  All_top5_mAP: {row['All_top5_mAP']:.4f}")
        print(f"  N_top5_mAP: {row['N_top5_mAP']:.4f}, Nv_top5_mAP: {row['Nv_top5_mAP']:.4f}, N_delta_top5_mAP: {row['N_delta_top5_mAP']:.4f}")
        print(f"  TTC MAE: {row['ttc_mae_seconds']:.4f}s")
        print()
