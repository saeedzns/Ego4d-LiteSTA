import json
from pathlib import Path
from collections import defaultdict

# Load recent Track C summaries
metrics_dir = Path('local_extraction/runs/Track_C/metrics')
files = sorted(metrics_dir.glob('*summary.json'), reverse=True)[:15]

print('=== Track C Pruning Results ===\n')
results = []

for f in files:
    data = json.loads(f.read_text())
    
    eval_cfg = data.get('eval_config', {})
    metrics = data.get('metrics', data)
    
    rate = eval_cfg.get('rgtp_rate', 0.0)
    checkpoint = metrics.get('checkpoint', '').split('\\')[-1].split('/')[-1]
    
    accuracy = metrics.get('accuracy', 0) * 100
    mAP = metrics.get('mAP', 0) * 100
    N_mAP = metrics.get('N_mAP', 0) * 100
    All_top5_mAP = metrics.get('All_top5_mAP', 0) * 100
    ttc_mae = metrics.get('ttc_mae_seconds', 0)
    
    # Get latency info if available
    avg_time = metrics.get('avg_time_per_sample', 0)
    
    results.append({
        'file': f.name,
        'checkpoint': checkpoint,
        'rate': rate,
        'accuracy': accuracy,
        'mAP': mAP,
        'N_mAP': N_mAP,
        'All_top5_mAP': All_top5_mAP,
        'ttc_mae': ttc_mae,
        'avg_time': avg_time
    })

# Group by checkpoint and rate
by_checkpoint = defaultdict(list)
for r in results:
    by_checkpoint[r['checkpoint']].append(r)

for checkpoint, runs in by_checkpoint.items():
    print(f'\n{checkpoint}:')
    # Sort by rate
    runs_sorted = sorted(runs, key=lambda x: x['rate'])
    for r in runs_sorted:
        print(f'  Rate={r["rate"]:.1f}: All_top5_mAP={r["All_top5_mAP"]:.2f}%, N_mAP={r["N_mAP"]:.2f}%, Acc={r["accuracy"]:.2f}%, TTC={r["ttc_mae"]:.4f}s')
        if r['avg_time'] > 0:
            print(f'    Avg time: {r["avg_time"]:.4f}s/sample')

# Show best Track C vs Track B baseline
print('\n\n=== Track C vs Track B Comparison ===')
print('Track B Baseline (no pruning, with CLIP):')
print('  All_top5_mAP: 4.81%')
print('  N_mAP: 20.14%, Accuracy: 64.23%')
print('  TTC MAE: 0.1996s')
print()
print('Track C Best Runs (sorted by All_top5_mAP):')
for r in sorted(results, key=lambda x: x['All_top5_mAP'], reverse=True)[:5]:
    print(f'  Rate={r["rate"]:.1f}: All_top5_mAP={r["All_top5_mAP"]:.2f}%, N_mAP={r["N_mAP"]:.2f}%, Acc={r["accuracy"]:.2f}%')

# Calculate metric degradation
print('\n=== Metric Degradation Analysis ===')
baseline_all_top5 = 4.81
baseline_n_map = 20.14

for r in sorted(results, key=lambda x: x['rate']):
    if r['rate'] > 0:
        deg_all = ((baseline_all_top5 - r['All_top5_mAP']) / baseline_all_top5) * 100
        deg_n = ((baseline_n_map - r['N_mAP']) / baseline_n_map) * 100
        print(f'Rate={r["rate"]:.1f}: All_top5 degradation={deg_all:.1f}%, N_mAP degradation={deg_n:.1f}%')
        print(f'  Tokens pruned: {r["rate"]*100:.0f}% → Speedup potential: ~{1/(1-r["rate"]):.1f}x')
