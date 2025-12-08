#!/usr/bin/env python3
"""Analyze best Track B and C runs by thesis metrics."""

import json
from pathlib import Path
import re

b_dir = Path('local_extraction/runs/Track_B/metrics')
c_dir = Path('local_extraction/runs/Track_C/metrics')

b_files = sorted(b_dir.glob('metrics_val_*.json'))
c_files = sorted(c_dir.glob('trackC_val_rate*.json'))

key_metrics = ['N_top5_mAP', 'All_top5_mAP', 'mAP', 'N_mAP', 'accuracy']

print('='*90)
print('TRACK B - Best runs by thesis metrics')
print('='*90)

b_data = []
for p in b_files:
    if '_summary' in p.name:
        continue
    try:
        d = json.loads(p.read_text())
        if d.get('N_top5_mAP') is not None:
            d['_file'] = p.name
            d['_ckpt'] = Path(d.get('checkpoint', '')).stem if d.get('checkpoint') else '-'
            b_data.append(d)
    except:
        pass

b_sorted = sorted(b_data, key=lambda x: x.get('N_top5_mAP', 0), reverse=True)
print('Top 5 by N_top5_mAP:')
for i, d in enumerate(b_sorted[:5], 1):
    ckpt = d['_ckpt'][:40]
    n_top5 = d.get('N_top5_mAP', 0)
    all_top5 = d.get('All_top5_mAP', 0)
    mAP = d.get('mAP', 0)
    print(f"{i}. N_top5={n_top5:.4f} All_top5={all_top5:.4f} mAP={mAP:.4f} ckpt={ckpt}")

print()
print('='*90)
print('TRACK C - Best runs by thesis metrics')
print('='*90)

c_data = []
for p in c_files:
    if '_summary' in p.name:
        continue
    try:
        d = json.loads(p.read_text())
        if d.get('N_top5_mAP') is not None:
            d['_file'] = p.name
            d['_ckpt'] = Path(d.get('checkpoint', '')).stem if d.get('checkpoint') else '-'
            rate_match = re.search(r'rate(\d+)', p.name)
            d['_rate'] = int(rate_match.group(1)) if rate_match else 0
            c_data.append(d)
    except:
        pass

c_sorted = sorted(c_data, key=lambda x: x.get('N_top5_mAP', 0), reverse=True)
print('Top 5 by N_top5_mAP:')
for i, d in enumerate(c_sorted[:5], 1):
    ckpt = d['_ckpt'][:35]
    rate = d['_rate']
    n_top5 = d.get('N_top5_mAP', 0)
    all_top5 = d.get('All_top5_mAP', 0)
    mAP = d.get('mAP', 0)
    print(f"{i}. rate={rate:02d} N_top5={n_top5:.4f} All_top5={all_top5:.4f} mAP={mAP:.4f} ckpt={ckpt}")

print()
print('='*90)
print('BEST COMPARISON (Thesis Goal: High N_top5_mAP + All_top5_mAP)')
print('='*90)
best_b = b_sorted[0]
best_c = c_sorted[0]
print(f"Best Track B: {best_b['_ckpt']}")
print(f"Best Track C: {best_c['_ckpt']} (rate={best_c['_rate']})")
print()
print('Metric          Track_B    Track_C    Delta')
print('-'*50)
for m in key_metrics:
    bv, cv = best_b.get(m, 0), best_c.get(m, 0)
    delta = cv - bv
    sign = '+' if delta >= 0 else ''
    print(f"{m:<15} {bv:>8.4f}   {cv:>8.4f}   {sign}{delta:.4f}")

print()
print('='*90)
print('THESIS RECOMMENDATION')
print('='*90)
print()
print("For THESIS GOAL (practical STA with efficient inference):")
print()
print(f"  ** BEST Track B: {best_b['_ckpt']}")
print(f"     - N_top5_mAP: {best_b.get('N_top5_mAP', 0):.4f}")
print(f"     - All_top5_mAP: {best_b.get('All_top5_mAP', 0):.4f}")
print(f"     - mAP: {best_b.get('mAP', 0):.4f}")
print()
print(f"  ** BEST Track C: {best_c['_ckpt']} (rate={best_c['_rate']})")
print(f"     - N_top5_mAP: {best_c.get('N_top5_mAP', 0):.4f}")
print(f"     - All_top5_mAP: {best_c.get('All_top5_mAP', 0):.4f}")
print(f"     - mAP: {best_c.get('mAP', 0):.4f}")
print(f"     - Pruning rate: {best_c['_rate']}%")
