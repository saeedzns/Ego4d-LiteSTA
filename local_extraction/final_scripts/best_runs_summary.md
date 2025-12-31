# Best Track B/Track C Runs

Selection metrics (weighted-class runs only):
- Track B: composite score (mAP + N_top5 + Nv_top5 + N_delta_top5 + All_top5)
- Track C: composite score (mAP + N_top5 + Nv_top5 + N_delta_top5 + All_top5)
- Weights: `mAP=1,N_top5_mAP=1,Nv_top5_mAP=1,N_delta_top5_mAP=1,All_top5_mAP=1` (missing metrics count as 0)

## Best Track B run
| run | checkpoint | score | mAP | N_top5_mAP | Nv_top5_mAP | N_delta_top5_mAP | All_top5_mAP | ttc_mae_seconds | video_backbone | tokens_root | loss_w_noun | loss_w_verb | class_weight_alpha | use_class_weights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metrics_val_20251226_231700_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1501 | 0.3734 | 0.1528 | 0.0481 | 0.1283 | 0.0481 | 0.1996 | resnet18 | v2/resnet18_tokens | 1.0000 | 1.0000 | 0.5000 | 1.0000 |

## Best Track C run (overall)
| run | checkpoint | score | mAP | N_top5_mAP | Nv_top5_mAP | N_delta_top5_mAP | All_top5_mAP | ttc_mae_seconds | rgtp_rate_request | rgtp_mean_fraction_pruned | pruning_enabled | video_backbone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trackC_val_rate00_20251226_141723_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1402 | 0.3761 | 0.1380 | 0.0335 | 0.1189 | 0.0344 | 0.1989 | 0.0000 | 0.0000 | 0.0000 |  |

## Best Track C runs by pruning rate
| run | checkpoint | score | All_top5_mAP | rgtp_rate_request | rgtp_mean_fraction_pruned | mAP | ttc_mae_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trackC_val_rate00_20251226_141723_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1402 | 0.0344 | 0.0000 | 0.0000 | 0.3761 | 0.1989 |
| trackC_val_rate10_20251226_163017_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1398 | 0.0345 | 0.1000 | 0.0021 | 0.3735 | 0.1992 |
| trackC_val_rate30_20251226_172547_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1346 | 0.0348 | 0.3000 | 0.1019 | 0.3545 | 0.1998 |

## All Track B runs (summary)
| run | checkpoint | score | mAP | N_top5_mAP | Nv_top5_mAP | N_delta_top5_mAP | All_top5_mAP | video_backbone | tokens_root |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metrics_val_20251225_181100_summary.json | trackB_best_mAP_0.3530_20251225_180721.pt | 0.0931 | 0.3562 | 0.0578 | 0.0013 | 0.0486 | 0.0013 | resnet18 | v2/resnet18_tokens |
| metrics_val_20251225_213431_summary.json | trackB_best_mAP_0.3820_20251225_204041.pt | 0.1387 | 0.3851 | 0.1282 | 0.0372 | 0.1094 | 0.0339 | resnet18 | v2/resnet18_tokens |
| metrics_val_20251226_013919_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1489 | 0.3734 | 0.1501 | 0.0473 | 0.1264 | 0.0473 | resnet18 | v2/resnet18_tokens |
| metrics_val_20251226_231700_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1501 | 0.3734 | 0.1528 | 0.0481 | 0.1283 | 0.0481 | resnet18 | v2/resnet18_tokens |
| metrics_val_20251227_001540_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1374 | 0.3734 | 0.1415 | 0.0288 | 0.1148 | 0.0288 | resnet18 | v2/resnet18_tokens |

## All Track C runs (summary)
| run | checkpoint | score | All_top5_mAP | rgtp_rate_request | rgtp_mean_fraction_pruned | mAP | ttc_mae_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trackC_val_rate00_20251226_141723_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1402 | 0.0344 | 0.0000 | 0.0000 | 0.3761 | 0.1989 |
| trackC_val_rate00_20251226_151551_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1401 | 0.0344 | 0.0000 | 0.0000 | 0.3761 | 0.1989 |
| trackC_val_rate00_20251226_153959_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1401 | 0.0344 | 0.0000 | 0.0000 | 0.3761 | 0.1989 |
| trackC_val_rate00_20251226_161030_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1402 | 0.0344 | 0.0000 | 0.0000 | 0.3761 | 0.1989 |
| trackC_val_rate10_20251226_163017_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1398 | 0.0345 | 0.1000 | 0.0021 | 0.3735 | 0.1992 |
| trackC_val_rate10_20251226_165938_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1397 | 0.0344 | 0.1000 | 0.0021 | 0.3735 | 0.1992 |
| trackC_val_rate10_20251226_193527_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1398 | 0.0345 | 0.1000 | 0.0021 | 0.3735 | 0.1992 |
| trackC_val_rate30_20251226_172547_summary.json | trackB_best_mAP_0.3708_20251225_224220.pt | 0.1346 | 0.0348 | 0.3000 | 0.1019 | 0.3545 | 0.1998 |

