#!/usr/bin/env python3
"""Analyze December runs from JSON files to find best Track B and C."""

import json
from pathlib import Path

RUNS_DIR = Path(__file__).parent.parent / "runs"

def load_december_metrics():
    """Load December metrics directly from JSON files."""
    dec_b = []
    dec_c = []
    
    # Track B metrics
    b_metrics_dir = RUNS_DIR / "Track_B" / "metrics"
    for f in b_metrics_dir.glob("metrics_val_202512*.json"):
        if "_summary" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            if data.get('N_top5_mAP') is not None:
                data['file'] = f.name
                dec_b.append(data)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    # Track C metrics
    c_metrics_dir = RUNS_DIR / "Track_C" / "metrics"
    for f in c_metrics_dir.glob("trackC_val_*202512*.json"):
        if "_summary" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            if data.get('N_top5_mAP') is not None:
                data['file'] = f.name
                # Extract rate from filename
                if '_rate' in f.name:
                    data['rate'] = f.name.split('_rate')[1].split('_')[0]
                dec_c.append(data)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    return dec_b, dec_c

def main():
    dec_b, dec_c = load_december_metrics()

    print('='*90)
    print('DECEMBER TRACK B RUNS (sorted by N_top5_mAP)')
    print('='*90)
    if dec_b:
        dec_b_sorted = sorted(dec_b, key=lambda x: x.get('N_top5_mAP', 0), reverse=True)
        print(f'Found {len(dec_b)} runs\n')
        for r in dec_b_sorted[:5]:
            n = r.get('N_top5_mAP', 0)
            nv = r.get('Nv_top5_mAP', 0)
            nd = r.get('N_delta_top5_mAP', 0)
            all_ = r.get('All_top5_mAP', 0)
            print(f"N={n:.4f} N+V={nv:.4f} N+d={nd:.4f} All={all_:.4f}")
            print(f"  checkpoint: {r.get('checkpoint', 'unknown')}")
            print(f"  file: {r['file']}")
            print()
    else:
        print('No December Track B runs found')

    print('='*90)
    print('DECEMBER TRACK C RUNS (sorted by N_top5_mAP)')
    print('='*90)
    if dec_c:
        dec_c_sorted = sorted(dec_c, key=lambda x: x.get('N_top5_mAP', 0), reverse=True)
        print(f'Found {len(dec_c)} runs\n')
        for r in dec_c_sorted[:5]:
            rate = r.get('rate', '?')
            n = r.get('N_top5_mAP', 0)
            nv = r.get('Nv_top5_mAP', 0)
            nd = r.get('N_delta_top5_mAP', 0)
            all_ = r.get('All_top5_mAP', 0)
            print(f"rate={rate}% N={n:.4f} N+V={nv:.4f} N+d={nd:.4f} All={all_:.4f}")
            print(f"  checkpoint: {r.get('checkpoint', 'unknown')}")
            print(f"  file: {r['file']}")
            print()
    else:
        print('No December Track C runs found')

    # Best comparison
    if dec_b and dec_c:
        best_b = sorted(dec_b, key=lambda x: x.get('N_top5_mAP', 0), reverse=True)[0]
        best_c = sorted(dec_c, key=lambda x: x.get('N_top5_mAP', 0), reverse=True)[0]
        
        print('='*90)
        print('DECEMBER BEST - THESIS GOAL METRICS')
        print('='*90)
        print(f"\nBEST Track B: {best_b.get('checkpoint', 'unknown')}")
        print(f"  N_top5_mAP:       {best_b.get('N_top5_mAP', 0):.4f}")
        print(f"  Nv_top5_mAP:      {best_b.get('Nv_top5_mAP', 0):.4f}")
        print(f"  N_delta_top5_mAP: {best_b.get('N_delta_top5_mAP', 0):.4f}")
        print(f"  All_top5_mAP:     {best_b.get('All_top5_mAP', 0):.4f}")
        
        print(f"\nBEST Track C: {best_c.get('checkpoint', 'unknown')} (rate={best_c.get('rate', '?')}%)")
        print(f"  N_top5_mAP:       {best_c.get('N_top5_mAP', 0):.4f}")
        print(f"  Nv_top5_mAP:      {best_c.get('Nv_top5_mAP', 0):.4f}")
        print(f"  N_delta_top5_mAP: {best_c.get('N_delta_top5_mAP', 0):.4f}")
        print(f"  All_top5_mAP:     {best_c.get('All_top5_mAP', 0):.4f}")

if __name__ == '__main__':
    main()
