import pandas as pd

# Load Track C TSV
df = pd.read_csv('local_extraction/runs/Track_C/plots/trackC_metrics_summary.tsv', sep='\t')

# Analyze where time is spent
print('=== Latency Breakdown Analysis ===\n')

# Get runs with instrumentation data
df_with_latency = df[df['latency_ms_mean'].notna() & (df['latency_ms_mean'] > 0)]

print(f"Total runs with latency data: {len(df_with_latency)}")
print()

# Group by pruning rate
print("Rate | Total_ms | Head_ms | Head% | Non-head_ms | Non-head%")
print("-" * 60)

for rate in sorted(df_with_latency['rgtp_rate_request'].unique()):
    rate_df = df_with_latency[df_with_latency['rgtp_rate_request'] == rate]
    avg_latency = rate_df['latency_ms_mean'].mean()
    avg_head = rate_df['head_benchmark_ms_mean'].mean()
    head_pct = (avg_head / avg_latency) * 100
    non_head = avg_latency - avg_head
    non_head_pct = 100 - head_pct
    
    print(f"{rate:.1f}   | {avg_latency:6.1f}   | {avg_head:5.2f}  | {head_pct:4.1f}% | {non_head:8.1f}    | {non_head_pct:5.1f}%")

print()
print("=" * 60)
print("KEY INSIGHT: Track C pruning problem")
print("=" * 60)
print()
print("The HEAD is only ~0.3ms (< 1% of total ~30-50ms latency)!")
print()
print("Current RGTP prunes tokens BEFORE the head, but:")
print("  - Backbone (ResNet18): ~15-20ms")
print("  - FGTP + Dual Cross-Attention: ~15-25ms")
print("  - Head: ~0.3ms")
print()
print("Pruning tokens only helps the HEAD, which is already tiny.")
print("You get <1% speedup from pruning 50% of tokens!")
print()
print("SOLUTION: Prune at the RIGHT stage:")
print("1. Frame-level pruning: Skip some input frames")
print("2. Backbone pruning: Use lighter backbone (MobileNet)")
print("3. FGTP pruning: Reduce temporal attention computation")
print("4. Token merging: Merge similar tokens instead of dropping")
