# Hotspot Prior JSON — How to Build and Use

`trackB_eval.py` can blend in optional noun–verb priors for re-ranking. The expected JSON format is:

```json
{
  "pairs": {
    "noun_id,verb_id": score,
    "12,3": 0.9,
    "48,7": 0.2
  },
  "default": 0.0
}
```

- Keys are strings `"noun_id,verb_id"` using global STA IDs (same IDs in your StageB manifests).
- Values are float scores (e.g., estimated P(next-active | noun, verb)).
- `default` is used when a pair is not listed.

## How we generate it

Use `local_extraction/trackB/build_hotspot_priors.py` to compute empirical priors from StageB head manifests:
```bash
python local_extraction/trackB/build_hotspot_priors.py \
  --manifests local_extraction/runs/Track_A/trackA_stageB_20251117_184342/head_train.jsonl \
              local_extraction/runs/Track_A/trackA_stageB_20251117_184342/head_val.jsonl \
  --out local_extraction/v2/hotspot_priors_sta_train.json
```

The script:
1. Reads the manifests (JSON/JSONL) and iterates candidates.
2. Uses `noun_id`, `verb_id`, and `is_positive` to count per-pair positives and totals.
3. Computes `score = positives / totals` for each `(noun, verb)` pair.
4. Writes the JSON with those scores and `default: 0.0`.

## Using it in eval

In `trackB_eval.py`, set:
- `use_hotspot_priors = True`
- `hotspot_prior_path = Path("local_extraction/v2/hotspot_priors_sta_train.json")`
- `hotspot_alpha` (blend weight, e.g., 0.2–0.3) and `hotspot_default` as desired.

During evaluation, candidate scores are blended with the prior when both noun and verb are predicted. If the pair isn’t in `pairs`, `default` is used. If you don’t want any prior, set `use_hotspot_priors = False`.
