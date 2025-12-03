# Noun/Verb Mapping Sources Audit

Places still using *non* STA height-540 files to map noun/verb IDs to strings:

- `local_extraction/reports/check_manifest_taxonomy_coverage.py`
  - Loads taxonomy paths from CLI (`--noun_taxonomy`, `--verb_taxonomy`).
  - Intended sources: narration_noun_taxonomy.csv / narration_verb_taxonomy.csv or fho_main_taxonomy.json / fho_lta_taxonomy.json.
  - Purpose: coverage stats; not tied to STA height-540 vocab.

- `local_extraction/trackB/trackB_eval.py`
  - Optional CLIP re-ranking uses `EvalConfig.noun_label_path` to load labels from a taxonomy (JSON with `nouns` or CSV with `label`).
  - Typical sources: narration_* taxonomies or fho_main_taxonomy.json.
  - Used only for CLIP prompts/labels; not derived from STA height-540 files.

Notes:
- `local_extraction/trackB/trackB_show_samples.py` now maps noun/verb names directly from `fho_sta_{train,val}_height-540.json` and no longer relies on narration/main taxonomies.
- Other Track B training/eval code maps IDs from manifests/checkpoints without external taxonomies unless `noun_label_path` is explicitly provided as above.
