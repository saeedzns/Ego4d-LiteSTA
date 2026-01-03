#!/usr/bin/env python3
"""
Example: Custom Plot from JSON Data
====================================
Demonstrates how to create custom visualizations using the saved JSON data files.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Example 1: Simple scatter plot from efficiency configs
print("Example 1: Latency vs mAP Trade-off")
print("-" * 50)

data_file = Path("local_extraction/final_scripts/comparison_results/efficiency_configs_data.json")
with open(data_file, 'r') as f:
    data = json.load(f)

configs = data['configs']
names = [c['name'] for c in configs]
mAPs = [c['mAP'] for c in configs]
latencies = [c['latency_ms'] for c in configs]

# Create custom scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(latencies, mAPs, s=200, alpha=0.6, c=range(len(names)), cmap='viridis', edgecolors='black')

# Annotate each point
for name, lat, mAP in zip(names, latencies, mAPs):
    ax.annotate(name, (lat, mAP), xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel('Latency (ms)', fontsize=12)
ax.set_ylabel('mAP (%)', fontsize=12)
ax.set_title('Custom: Efficiency Trade-off Analysis', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')
plt.colorbar(scatter, label='Config Index')
plt.tight_layout()
plt.savefig('local_extraction/final_scripts/comparison_results/custom_example1.png', dpi=150)
print(f"✓ Saved: custom_example1.png")
plt.close()

# Example 2: Delta visualization
print("\nExample 2: Weighted vs Unweighted Delta Bars")
print("-" * 50)

data_file = Path("local_extraction/final_scripts/comparison_results/weighted_vs_unweighted_data.json")
with open(data_file, 'r') as f:
    data = json.load(f)

weighted = data['weighted']['metrics']
unweighted = data['unweighted']['metrics']

# Calculate deltas
metrics = list(weighted.keys())
deltas = [weighted[m] - unweighted[m] for m in metrics]

# Create delta bar chart
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if d > 0 else 'red' for d in deltas]
bars = ax.barh(metrics, deltas, color=colors, alpha=0.7, edgecolor='black')

# Add zero line
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Add value labels
for i, (bar, delta) in enumerate(zip(bars, deltas)):
    x_pos = delta + (0.5 if delta > 0 else -0.5)
    ax.text(x_pos, i, f'{delta:+.2f}%', va='center', ha='left' if delta > 0 else 'right', fontweight='bold')

ax.set_xlabel('Delta (Weighted - Unweighted) %', fontsize=12)
ax.set_title('Custom: Performance Delta Analysis\n(Green = Weighted wins, Red = Unweighted wins)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('local_extraction/final_scripts/comparison_results/custom_example2.png', dpi=150)
print(f"✓ Saved: custom_example2.png")
plt.close()

# Example 3: Extract data to CSV
print("\nExample 3: Export to CSV for Excel")
print("-" * 50)

import csv

csv_file = Path("local_extraction/final_scripts/comparison_results/efficiency_configs_export.csv")
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Header
    header = ['Configuration', 'mAP (%)', 'N_top5 (%)', 'Latency (ms)', 'Score']
    writer.writerow(header)
    
    # Data rows
    for c in configs:
        row = [
            c['name'],
            f"{c['mAP']:.2f}",
            f"{c['N_top5_mAP']:.2f}",
            f"{c['latency_ms']:.2f}",
            f"{c['score']:.4f}"
        ]
        writer.writerow(row)

print(f"✓ Saved: {csv_file.name}")
print(f"  Rows: {len(configs) + 1} (including header)")

# Example 4: Summary statistics
print("\nExample 4: Quick Statistics from JSON")
print("-" * 50)

print(f"\nEfficiency Configurations Summary:")
print(f"  Total configs: {len(configs)}")
print(f"  Best mAP: {max(mAPs):.2f}% ({names[mAPs.index(max(mAPs))]})")
print(f"  Lowest latency: {min(latencies):.2f}ms ({names[latencies.index(min(latencies))]})")
print(f"  mAP range: {min(mAPs):.2f}% - {max(mAPs):.2f}%")
print(f"  Latency range: {min(latencies):.2f}ms - {max(latencies):.2f}ms")

print("\n" + "=" * 50)
print("DONE! Check the comparison_results directory for output files.")
print("=" * 50)
