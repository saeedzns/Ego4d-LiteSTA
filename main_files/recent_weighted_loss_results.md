# Recent Weighted-Loss Results (Track B)

This note summarizes the latest weighted-loss runs and compares them to the Top-5 literature table in `main_files/thesis.md` (Table 8.5/8.6).

## Runs included
- `local_extraction/runs/Track_B/metrics/metrics_val_20251225_213431_summary.json`
- `local_extraction/runs/Track_B/metrics/metrics_val_20251225_181100_summary.json`
- `local_extraction/runs/Track_B/metrics/metrics_val_20251226_013919_summary.json`

## Top-5 mAP results (percent)

| Run | Checkpoint | class_weight_alpha | loss_w_noun / loss_w_verb | N | N+V | N+delta | All |
|---|---|---:|---:|---:|---:|---:|---:|
| 2025-12-25 21:34 | trackB_best_mAP_0.3820_20251225_204041.pt | 0.5 | 2.0 / 2.0 | 12.82 | 3.72 | 10.94 | 3.39 |
| 2025-12-25 18:11 | trackB_best_mAP_0.3530_20251225_180721.pt | 0.7 | 2.0 / 2.0 | 5.78 | 0.13 | 4.86 | 0.13 |
| 2025-12-26 01:39 | trackB_best_mAP_0.3708_20251225_224220.pt | 0.5 | 1.0 / 1.0 | 15.01 | 4.73 | 12.64 | 4.73 |

Notes:
- These are Top-5 mAP values (`N_top5_mAP`, `Nv_top5_mAP`, `N_delta_top5_mAP`, `All_top5_mAP`).
- `class_weight_alpha` and loss weights are pulled from the embedded `train_config` in each metrics file.

## Comparison to literature table in thesis.md

The literature table (Table 8.5) reports Top-5 mAP (%) for N / N+V / N+delta / All. The best reported baseline there is:
- STAformer + MH + AFF: N 29.39, N+V 15.38, N+delta 9.94, All 5.67

The weighted-loss runs show:
- Stronger N and N+delta than the thesis baseline in Table 8.6, especially the 2025-12-26 run.
- N+V and All remain below the strongest literature baselines.
- The alpha=0.7 run underperforms across all semantic metrics, suggesting too-aggressive weighting.

## Full comparison table (papers + best weighted run)

Top-5 mAP (%) values as reported in `main_files/thesis.md` (Table 8.5) plus the best weighted-loss run above.

| Method | N | N+V | N+delta | All |
|---|---:|---:|---:|---:|
| FRCNN+SF | 21.00 | 7.45 | 7.07 | 2.98 |
| StillFast | 20.26 | 10.37 | 7.26 | 3.96 |
| GANO v2 | 20.52 | 10.42 | 7.28 | 3.99 |
| STAformer | 24.85 | 13.45 | 7.41 | 4.90 |
| STAformer + MH + AFF | 29.39 | 15.38 | 9.94 | 5.67 |
| Ego4D-LiteSTA (weighted, best recent) | 15.01 | 4.73 | 12.64 | 4.73 |

Best recent weighted run used:
- `metrics_val_20251226_013919_summary.json`
- checkpoint: `trackB_best_mAP_0.3708_20251225_224220.pt`
- class_weight_alpha: 0.5, loss_w_noun / loss_w_verb: 1.0 / 1.0

## Interpretation (short)
- Weighted loss reduces class collapse and lifts N / N+delta, but does not close the gap on N+V and All.
- The best current weighted run uses alpha=0.5 and lower loss weights (1.0 / 1.0), suggesting aggressive weighting is not yet stable.

## Next steps (optional)
- Run longer training (10-20 epochs) for the alpha=0.5 setting.
- Try a milder weighting schedule: alpha=0.5, loss_w_noun=2.0, loss_w_verb=1.5.
- Re-evaluate with consistent config and checkpoint pairing to avoid mismatched eval settings.
