import json
from pathlib import Path

# Compare Track B vs Track C evaluations on SAME checkpoint
checkpoint = 'trackB_best_mAP_0.3708_20251225_224220.pt'

# Track B with CLIP
trackB_file = Path('local_extraction/runs/Track_B/metrics/metrics_val_20251226_231700_summary.json')
trackB_data = json.loads(trackB_file.read_text())

# Track C baseline (rate=0.0)
trackC_files = sorted(Path('local_extraction/runs/Track_C/metrics').glob('*summary.json'), reverse=True)
trackC_baseline = None
for f in trackC_files:
    data = json.loads(f.read_text())
    if checkpoint in data.get('metrics', {}).get('checkpoint', ''):
        rate = data.get('eval_config', {}).get('rgtp_rate', -1)
        if abs(rate - 0.0) < 0.01:
            trackC_baseline = data
            print(f'Found Track C baseline: {f.name}')
            break

if not trackC_baseline:
    print('Track C baseline not found!')
    exit(1)

print('\n=== Configuration Comparison ===')
print('Track B (CLIP):')
trackB_eval = trackB_data.get('eval_config', {})
print(f'  use_clip_rerank: {trackB_eval.get("use_clip_rerank", False)}')
print(f'  use_hotspot_priors: {trackB_eval.get("use_hotspot_priors", False)}')

print('\nTrack C (baseline):')
trackC_eval = trackC_baseline.get('eval_config', {})
print(f'  use_clip_rerank: {trackC_eval.get("use_clip_rerank", False)}')
print(f'  use_hotspot_priors: {trackC_eval.get("use_hotspot_priors", False)}')
print(f'  rgtp_enabled: {trackC_eval.get("rgtp_enabled", False)}')
print(f'  rgtp_rate: {trackC_eval.get("rgtp_rate", 0.0)}')

print('\n=== Metrics Comparison ===')
trackB_metrics = trackB_data.get('metrics', {})
trackC_metrics = trackC_baseline.get('metrics', {})

print(f'Track B: All_top5_mAP={trackB_metrics.get("All_top5_mAP", 0)*100:.2f}%, N_mAP={trackB_metrics.get("N_mAP", 0)*100:.2f}%, Acc={trackB_metrics.get("accuracy", 0)*100:.2f}%')
print(f'Track C: All_top5_mAP={trackC_metrics.get("All_top5_mAP", 0)*100:.2f}%, N_mAP={trackC_metrics.get("N_mAP", 0)*100:.2f}%, Acc={trackC_metrics.get("accuracy", 0)*100:.2f}%')

print('\n=== Difference Analysis ===')
diff_all = (trackB_metrics.get('All_top5_mAP', 0) - trackC_metrics.get('All_top5_mAP', 0)) * 100
diff_n = (trackB_metrics.get('N_mAP', 0) - trackC_metrics.get('N_mAP', 0)) * 100
diff_acc = (trackB_metrics.get('accuracy', 0) - trackC_metrics.get('accuracy', 0)) * 100

print('Track B is better by:')
print(f'  All_top5_mAP: +{diff_all:.2f}%')
print(f'  N_mAP: {diff_n:+.2f}%')
print(f'  Accuracy: {diff_acc:+.2f}%')

# Check for evaluation pipeline differences
print('\n=== Potential Issues ===')
if not trackC_eval.get('use_clip_rerank', False):
    print('❌ Track C does NOT have CLIP enabled (Track B does)')
if trackC_eval.get('rgtp_enabled', False):
    print('⚠️  Track C has RGTP enabled even at rate=0.0')
    
# Check num candidates
trackB_num = trackB_metrics.get('num_candidates', 0)
trackC_num = trackC_metrics.get('num_candidates', 0)
if trackB_num != trackC_num:
    print(f'⚠️  Different candidate counts: TrackB={trackB_num}, TrackC={trackC_num}')
else:
    print(f'✓ Same candidate count: {trackB_num}')
