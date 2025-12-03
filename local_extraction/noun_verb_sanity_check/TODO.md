# Noun/Verb Sanity Check — TODO

- **Quick run (specific frame):** `python local_extraction/noun_verb_sanity_check/check_labels.py --uid <video_uid> --frame <frame_idx> --org_root local_extraction/v2/org_annotations --frames_root local_extraction/v2/extracted_frames --noun_tax local_extraction/v2/org_annotations/fho_main_taxonomy.json --verb_tax local_extraction/v2/org_annotations/fho_lta_taxonomy.json`
- **Random samples:** `python local_extraction/noun_verb_sanity_check/check_labels.py --split val --k 3` (use `--split train` to sample train).
- **Taxonomy variants:** if you prefer narration CSVs, point `--noun_tax/--verb_tax` to your narration_* CSVs; the defaults above use the JSONs already present under `local_extraction/v2/org_annotations` with Ego4D ordering.
- **Frame roots:** if frames live elsewhere, change `--frames_root` to the correct extracted frame directory (expects `<uid>/<frame>.jpg` layout).
- **Org annotations root:** `--org_root` should point to the folder containing `fho_sta_train_height-540.json`, `fho_sta_val_height-540.json`, and `fho_main.json`.
- **Future improvements (nice-to-have):** add critical-frame jumpers (pre_45/pre_30/pre_15/pnr/contact), filtering by noun/verb IDs, and a scrolling UI to review consecutive frames without closing windows.

# Clip-centric checker (clip_uid space)
- **Specific clip frame:** `python local_extraction/noun_verb_sanity_check/check_labels_clip.py --clip_uid <clip_uid> --clip_frame <frame_idx> --org_root local_extraction/v2/org_annotations --frames_root local_extraction/v2/extracted_frames`
- **Random clip samples:** `python local_extraction/noun_verb_sanity_check/check_labels_clip.py --split val --k 3`
- **Batch flagging (CSV/JSONL):** `python local_extraction/noun_verb_sanity_check/check_labels_clip.py --mode batch --split val --out_csv reports/clip_flags.csv --out_jsonl reports/clip_flags.jsonl`
- **Hardcoded/CLI clip pairs:** edit `DEFAULT_BATCH_PAIRS` in the script or pass `--batch_pairs clip_uid1:frame1 clip_uid2:frame2`.


`python local_extraction/noun_verb_sanity_check/check_labels.py --split val --k 3 --org_root local_extraction/v2/org_annotations --frames_root local_extraction/v2/extracted_frames --noun_tax local_extraction/v2/org_annotations/fho_main_taxonomy.json --verb_tax local_extraction/v2/org_annotations/fho_lta_taxonomy.json`
