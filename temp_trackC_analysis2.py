import pandas as pd

# Load Track C TSV
df = pd.read_csv('local_extraction/runs/Track_C/plots/trackC_metrics_summary.tsv', sep='\t')

# Filter for class-weighted checkpoint runs
weighted_checkpoint = 'trackB_best_mAP_0.3708_20251225_224220.pt'
df_weighted = df[df['checkpoint'] == weighted_checkpoint].copy()

print(f'=== Track C Pruning Analysis: {weighted_checkpoint} ===\n')

# Group by rate and show best metrics
rates = sorted(df_weighted['rgtp_rate_request'].unique())
for rate in rates:
    rate_df = df_weighted[df_weighted['rgtp_rate_request'] == rate]
    
    # Get best run for this rate
    best = rate_df.nlargest(1, 'All_top5_mAP').iloc[0]
    
    print(f'Pruning Rate: {rate:.1f} ({rate*100:.0f}% tokens pruned)')
    print(f'  All_top5_mAP: {best["All_top5_mAP"]*100:.2f}%')
    print(f'  N_mAP: {best["N_mAP"]*100:.2f}%, Accuracy: {best["accuracy"]*100:.2f}%')
    print(f'  TTC MAE: {best["ttc_mae_seconds"]:.4f}s')
    print(f'  Latency: {best["latency_ms_mean"]:.2f}ms (median: {best["latency_ms_median"]:.2f}ms)')
    print(f'  Throughput: {best["throughput_samples_per_s"]:.2f} samples/s')
    print()

# Compare with Track B baseline
print('=== Track B Baseline (no pruning, with CLIP) ===')
print('  All_top5_mAP: 4.81%')
print('  N_mAP: 20.14%, Accuracy: 64.23%')
print('  TTC MAE: 0.1996s')
print()

# Calculate trade-offs
print('=== Performance vs Efficiency Trade-off ===')
baseline_all = 4.81
baseline_latency = df_weighted[df_weighted['rgtp_rate_request'] == 0.0]['latency_ms_mean'].mean()
print(f'Baseline latency: {baseline_latency:.2f}ms\n')

for rate in rates:
    if rate > 0:
        rate_df = df_weighted[df_weighted['rgtp_rate_request'] == rate]
        best = rate_df.nlargest(1, 'All_top5_mAP').iloc[0]
        
        all_top5 = best['All_top5_mAP'] * 100
        latency = best['latency_ms_mean']
        
        metric_loss = ((baseline_all - all_top5) / baseline_all) * 100
        speedup = baseline_latency / latency
        
        print(f'Rate {rate:.1f}: Metric loss={metric_loss:.1f}%, Speedup={speedup:.2f}x')
        print(f'  → For {speedup:.1f}x faster inference, you lose {metric_loss:.1f}% of official metric')
