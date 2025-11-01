# Ego4D Samples — File-by-File Guide

This document explains every file under `ego4d_samples/` and its subfolders. It summarizes what each file contains, the key fields inside, how to read it, and how each piece is useful for modeling and evaluation. These samples mirror the structure of the full Ego4D dataset (v1 and v2) but are reduced in size for quick inspection and prototyping.

## How These “CSV” Samples Are Structured
- Each sample file is a light wrapper around the real JSON files. Most files have a single logical column (e.g., `videos`, `clips`, or `annotations`) and each row is a single JSON-like object serialized as a string (Python dict literal style).
- Practically, to parse a row you can treat the string as JSON after minor normalization (e.g., replace single quotes with double quotes; escape inner quotes) or use a tolerant parser. For quick experiments in notebooks, many users simply evaluate with `ast.literal_eval` in Python.
- Coordinate systems and timing:
  - Times are commonly provided both in seconds (`*_sec`) and frames (`*_frame`), and often both in video time and clip time (e.g., `video_start_sec` vs `clip_start_sec`).
  - Boxes typically use pixel coordinates in the current frame. Boxes may be `[x1, y1, x2, y2]` or a dict `{x, y, width, height}` depending on task.
- Splits: Files are usually grouped by split: `train`, `val`, and `test_unannotated` (no labels).

## Top-Level

### `ego4d_samples/ego4d_json.csv`
- Purpose: Sample rows from the global `ego4d.json` index of videos.
- Structure: Header `videos`. Each row describes one video with:
  - `video_uid`: Unique identifier for the video.
  - `duration_sec`, `video_metadata`: FPS, frame counts, codecs, resolution, and timing bases.
  - `scenarios`: Free-text list of scene/role descriptions (e.g., “Car mechanic”).
  - `split_*`: Which benchmark splits this video belongs to (e.g., `split_fho`, `split_em`).
  - `s3_path`: Canonical S3 base path for the video.
  - `video_components`: Sub-chunks with their own metadata and exact frame/timestamp spans.
- Usefulness: Serves as the catalog for all videos; helpful for mapping `video_uid` to basic media properties (FPS/resolution) when converting times↔frames or preparing loaders.

## Versioned Layout

The samples mirror the official dataset’s versioned layout:

- `ego4d_samples/v1/` and `ego4d_samples/v2/`
  - `annotations/`: Task-specific annotation files (JSON content as CSV rows).
  - `sta_models/`: Precomputed detections and manifest for STA.
  - `omnivore_video_swinl_fp16/`: Feature manifests for video backbones.

The `v2` files are newer and sometimes include richer fields or additional tasks compared to `v1`.

## Annotations (Common Tasks)

Below, “v1” and “v2” sets present the same family of tasks. Field names can differ slightly across versions but follow the same intent.

### Narrations
- Files:
  - v1: `v1/annotations/narration_json.csv`
  - v2: `v2/annotations/narration_json.csv`
  - v2 (all, redacted): `v2/annotations/all_narrations_redacted_json.csv`
- Structure:
  - Narrations file: Pairs of `video_uid, clip_uid` then a dict with `narrations: [{timestamp_sec, timestamp_frame, narration_text, annotation_uid, ...}]`.
  - All narrations (redacted): A flat list of narration items with `text`, `_clip_time_start`, `_clip_time_end`, `annotator`, etc. (content redacted/normalized).
- Usefulness: Language supervision aligned with time; supports NLQ/VQ baselines, post-hoc labeling, or weakly-supervised training.

### Taxonomies (Noun/Verb)
- Files:
  - v1: `v1/annotations/narration_noun_taxonomy_csv.csv`, `v1/annotations/narration_verb_taxonomy_csv.csv`
  - v2: `v2/annotations/narration_noun_taxonomy_csv.csv`, `v2/annotations/narration_verb_taxonomy_csv.csv`
- Structure: CSV with `label, group` where `group` lists canonicalized synonyms. Example verb label: `adjust_(regulate,_increase/reduce,_change)`.
- Usefulness: Normalizes free-form narrations to canonical labels; useful for mapping to class indices.

### FHO Main (Index of Intervals & Actions)
- File: `v2/annotations/fho_main_json.csv` and taxonomy `v2/annotations/fho_main_taxonomy_json.csv`
- Structure: Header `videos`. Each row has:
  - `annotated_intervals`: Clip spans with `clip_uid`, `start/end_sec`, and `narrated_actions` (each with validity flags, `narration_text`, and optionally `critical_frames` like `pre_45`, `contact_frame`, `pnr_frame`).
  - `video_metadata`: FPS, frames, width, height.
- Usefulness: Master index for FHO tasks; ties narrated actions and critical frames to video/clip timing.

### FHO STA (Short‑Term Anticipation)
- Files:
  - v1: `v1/annotations/fho_sta_train_json.csv`, `..._val_...`, `..._test_unannotated_...`
  - v2: `v2/annotations/fho_sta_train_json.csv`, `..._val_...`, `..._test_unannotated_...`
- Structure: Header `annotations`. Each row is an STA example with:
  - Identifiers: `uid`, `main_uid`, `video_uid`, `clip_uid`, `clip_id`.
  - Target time window: `action_*` in sec/frame; also clip-scoped equivalents.
  - Context interval: `interval_start/end_sec|frame` and parent clip bounds.
  - Objects: List of `{box: [x1,y1,x2,y2], verb_category_id, noun_category_id, time_to_contact}` at key pre‑frames.
- Usefulness: Core supervision for predicting imminent hand‑object interactions; boxes and `time_to_contact` support anticipation heads and metrics.

### FHO LTA (Long‑Term Anticipation)
- Files:
  - v1: `v1/annotations/fho_lta_train_json.csv`, `..._val_...`, `..._test_unannotated_...`, taxonomy `fho_lta_taxonomy_json.csv`
  - v2: same under `v2/annotations/`
- Structure: Header `clips`. Each row includes:
  - Clip bounds (`clip_parent_*`, `interval_*`), action label triplets (`verb`, `noun`, `verb_label`, `noun_label`).
  - Action timing inside the clip (`action_clip_start/end_sec|frame`) and index `action_idx`.
- Usefulness: Supervises longer‑horizon forecasting of action categories and their timing.

### FHO OSCC‑PNR (Object State Change Classification — Point of No Return)
- Files:
  - v1: `v1/annotations/fho_oscc-pnr_train_json.csv`, `..._val_...`, `..._test_unannotated_...`
  - v2: same under `v2/annotations/`
- Structure: Header `clips`. Rows define short clips around a state change with:
  - `clip_*` and `parent_*` start/end (sec, frame), and the `clip_pnr_frame`/`parent_pnr_frame`.
  - `state_change`: Boolean.
- Usefulness: Supervision for detecting the decisive transition instant (PNR) during object state change.

### FHO SCOD (State Change Object Detection)
- Files: `v1/annotations/fho_scod_train_json.csv`, `..._val_...`, `..._test_unannotated_...`
- Structure: Header `clips`. Each row groups three frames (`pre_frame`, `pnr_frame`, `post_frame`), each with:
  - Frame indices and image size.
  - Bounding boxes for hands (`left_hand`, `right_hand`), `object_of_change`, and tools. Box dicts use `{x, y, width, height}`.
- Usefulness: Trains detectors to localize the object undergoing state change around PNR.

### FHO Hands
- Files:
  - v1: `v1/annotations/fho_hands_train_json.csv`, `..._val_...`, `..._test_unannotated_...`
  - v2: only `..._test_unannotated_...` in samples; train/val often provided in v1.
- Structure: Header `clips`. For each action instance, provides key frames (`pre_45`, `pre_30`, `pre_15`, `pre_frame`, `pnr_frame`, `post_frame`) with `boxes` listing hand centroids or small boxes, e.g., `{'right_hand': [x, y]}` or bounding boxes.
- Usefulness: Supervision for hand localization/association around interaction events.

### NLQ (Natural Language Queries)
- Files: `.../nlq_train_json.csv`, `..._val_...`, `..._test_unannotated_...`
- Structure: Header `videos`. Each video has `clips` with `annotations` containing `language_queries`: 
  - Fields: `template`, `query`, `slot_x`, `verb_x`, optionally `slot_y`, `verb_y`, plus aligned `clip_*` and `video_*` times and frames.
- Usefulness: Text‑video retrieval/grounding: locate temporal spans satisfying a natural question.

### VQ (Visual Queries)
- Files: `.../vq_train_json.csv`, `..._val_...`, `..._test_unannotated_...`
- Structure: Header `videos`. Each clip has `annotations` with `query_sets` keyed by ID. For each set:
  - `query_frame`, `query_video_frame` (where the query crop is taken).
  - `visual_crop`: a 2D crop box at the query frame.
  - `response_track`: per‑frame track `{frame_number, x, y, width, height, video_frame_number, original_width, original_height}`.
  - Optional `object_title`, flags (`is_valid`, `errors`, `warnings`).
- Usefulness: Tracks the referred object across time; useful for training/validating ref‑tracking models.

### AV (Audio–Visual / Camera Wearer Speech Segments)
- Files: `.../av_train_json.csv`, `..._val_...`, `..._test_unannotated_...`
- Structure: Header `videos`. Clips include a `camera_wearer` block:
  - `voice_segments`: list with `start_time`, `end_time`, `start_frame`, `end_frame`, and corresponding `video_*` times/frames, tagged with the `person` ID.
  - Some files also hold non‑wearer tracks; in samples, the camera wearer is emphasized.
- Usefulness: Speech diarization/alignment; cross‑modal baselines.

### Moments (Temporal Action Labels)
- Files: `.../moments_train_json.csv`, `..._val_...`, `..._test_unannotated_...`
- Structure: Header `videos`. Each clip has multiple annotators, each listing labeled intervals:
  - For each label: `{start_time, end_time, label, video_start_time, video_end_time, video_start_frame, video_end_frame, primary}`.
- Usefulness: Dense temporal action labeling for moment retrieval or segmentation models.

### Manifests (Where to Find Full Files)
- Files:
  - v1: `v1/annotations/manifest_csv.csv`
  - v2: `v2/annotations/manifest_csv.csv`
- Structure: CSV with `file_uid, annotation, type, canonical_s3_location`.
- Usefulness: Pointers to the canonical JSONs on S3 for full download.

## STA Models (Detections & Manifests)

### `sta_models/object_detections_json.csv`
- Structure: First column is a key like `clip_uid_frame` (e.g., `9c59e9..._0000146`), followed by rows with detection records:
  - `{box: [x1,y1,x2,y2], score, noun_category_id}`.
- Usefulness: Precomputed object detections to bootstrap STA training or as proposals.

### `sta_models/manifest_csv.csv`
- Structure: CSV mapping artifact types to S3 (e.g., `object_detections.json`, `object_detector.pth`).
- Usefulness: Reproducible linkage to pretrained detectors and their outputs.

## Omnivore Features Manifests

### `omnivore_video_swinl_fp16/manifest_csv.csv` (v1 and v2)
- Structure: CSV with `video_uid, type, s3_path, benchmarks`.
- Usefulness: Lists precomputed action features per video (e.g., `.pt` tensors) and applicable benchmarks. Helpful for feature‑based baselines without end‑to‑end training.

## Reading Tips (Practical)
- JSON strings in CSV: Many rows are single‑quoted Python dicts. In Python:
  - Use `ast.literal_eval(row_str)` to safely parse to dict; or normalize quotes and use `json.loads`.
  - For large files, stream line‑by‑line rather than loading all rows.
- Time↔frame conversion: Prefer the per‑row FPS where provided; avoid assuming global FPS.
- Coordinates: Check whether a task uses `[x1,y1,x2,y2]` vs `{x,y,width,height}` before decoding.
- Splits: Respect `*_test_unannotated` files — they intentionally lack ground truth for fair evaluation.

## File Index (by Path)

- Top level
  - `ego4d_samples/ego4d_json.csv` — Global video catalog (sampled).

- v1/annotations (Ego4D v1 samples)
  - `av_{train,val,test_unannotated}_json.csv` — Camera‑wearer voice segments and AV clips.
  - `fho_hands_{train,val,test_unannotated}_json.csv` — Key hand frames and hand/object locations.
  - `fho_lta_{train,val,test_unannotated}_json.csv` — Long‑term anticipation labels and timings.
  - `fho_lta_taxonomy_json.csv` — LTA verb taxonomy (labels).
  - `fho_oscc-pnr_{train,val,test_unannotated}_json.csv` — State‑change PNR clips.
  - `fho_scod_{train,val,test_unannotated}_json.csv` — SCOD frames and bounding boxes.
  - `fho_sta_{train,val,test_unannotated}_json.csv` — Short‑term anticipation objects and TTC.
  - `manifest_csv.csv` — S3 manifest for v1 annotations.
  - `moments_{train,val,test_unannotated}_json.csv` — Temporal action labels.
  - `narration_json.csv` — Video/clip narrations.
  - `narration_noun_taxonomy_csv.csv`, `narration_verb_taxonomy_csv.csv` — Taxonomies.
  - `nlq_{train,val,test_unannotated}_json.csv` — Natural language queries.
  - `vq_{train,val,test_unannotated}_json.csv` — Visual queries/tracks.

- v2/annotations (Ego4D v2 samples)
  - All of the above plus:
    - `fho_main_json.csv` — FHO video index with narrated actions and critical frames.
    - `fho_main_taxonomy_json.csv` — FHO taxonomy.
    - `all_narrations_redacted_json.csv` — Aggregated (redacted) narrations.

- v1/v2/sta_models
  - `object_detections_json.csv` — STA object detections (sample rows).
  - `manifest_csv.csv` — S3 manifest for STA artifacts.

- v1/v2/omnivore_video_swinl_fp16
  - `manifest_csv.csv` — S3 locations for precomputed features per `video_uid`.

## Example: Minimal Parsers

Below is a compact parsing sketch for two common patterns.

```python
# 1) “videos” CSV (e.g., NLQ, VQ, AV, Moments)
import csv, ast
rows = []
with open('ego4d_samples/v2/annotations/nlq_train_json.csv', newline='', encoding='utf-8') as f:
    r = csv.reader(f)
    header = next(r)[0]  # 'videos'
    for (col,) in r:
        rows.append(ast.literal_eval(col))  # each row is a dict

print(rows[0]['clips'][0]['annotations'][0]['language_queries'][0]['query'])

# 2) “annotations” CSV (e.g., STA)
with open('ego4d_samples/v2/annotations/fho_sta_train_json.csv', newline='', encoding='utf-8') as f:
    r = csv.reader(f)
    header = next(r)[0]  # 'annotations'
    first = ast.literal_eval(next(r)[0])
    print(first['objects'][0]['box'], first['objects'][0].get('time_to_contact'))
```

## Suggested Uses by Task
- STA: Train short‑horizon anticipation with object boxes and `time_to_contact`; combine with detections from `sta_models/`.
- LTA: Forecast action categories and timing from `fho_lta_*` rows; exploit `verb_label`/`noun_label` and intervals.
- OSCC‑PNR: Learn to localize PNR; treat each clip as a positive/negative with `clip_pnr_frame` for supervision.
- SCOD: Object detection around state changes; multi‑frame supervision per clip.
- Hands: Hand detection/association; use pre/PNR/post key frames as anchors.
- NLQ/VQ: Language grounding (temporal for NLQ; spatial‑temporal tracks for VQ).
- AV: Cross‑modal alignment; camera‑wearer speech segmentation.
- Narrations: Pretraining or weak supervision from natural language labels aligned in time.

---

If you want me to add per‑task schema tables or ready‑to‑run loaders for these sample CSVs, say the word and I’ll drop them into a `utils/` module or a notebook.

## Auto‑Parsed Fields (per file)
- I auto‑inspected every CSV in `ego4d_samples/` and summarized headers, structure type, and discovered top‑level keys (plus examples of nested keys) for each file.
- See full, generated summary: `ego4d_samples/_auto_fields.md`
- This is useful to quickly see what fields are present without opening large files.

## Field Glossary (key meanings)
- `video_uid`: Unique identifier of a full video in Ego4D.
- `clip_uid`: Unique identifier for a sub‑segment (clip) within a video.
- `clip_id`: Numeric/string ID for a clip used by some tasks; not globally unique.
- `source_clip_uid`: Original clip UID from the canonical dataset when a clip is re‑used.
- `split`: Dataset split; typically `train`, `val`, or `test_unannotated`.
- `s3_path` / `canonical_s3_location`: Canonical path to the asset on S3 (full dataset, not samples).

- `video_start_sec` / `video_end_sec`: Start/end times in seconds within the full video.
- `video_start_frame` / `video_end_frame`: Start/end frames within the full video.
- `clip_start_sec` / `clip_end_sec`: Start/end times in seconds within the derived clip.
- `clip_start_frame` / `clip_end_frame`: Start/end frames within the derived clip.
- `clip_parent_start_sec` / `clip_parent_end_sec`: Clip bounds within the parent video (alternate naming used by some files).
- `clip_parent_start_frame` / `clip_parent_end_frame`: Same as above in frames.
- `interval_start_sec` / `interval_end_sec`: Context window around an action (often for LTA/STA), in seconds.
- `interval_start_frame` / `interval_end_frame`: Same interval in frames.

- `action_start_sec` / `action_end_sec`: Target action window in seconds.
- `action_start_frame` / `action_end_frame`: Same target action window in frames.
- `action_clip_start_sec` / `action_clip_end_sec`: Action window expressed in the clip’s time base.
- `action_clip_start_frame` / `action_clip_end_frame`: Same in frames.
- `action_idx`: Index of the action within a clip (LTA).

- `pnr_frame`: Point‑of‑No‑Return frame index (OSCC‑PNR) inside the clip; decisive transition.
- `parent_pnr_frame`: PNR in parent video frame numbering (if provided).

- `frames`: Container used by Hands/SCOD; holds keyframes and annotations.
- `pre_45` / `pre_30` / `pre_15`: Key frames 45/30/15 frames before the contact/PNR.
- `pre_frame` / `post_frame` / `contact_frame`: Pre/after/contact frames around the event.

- `boxes` / `bbox` / `box`:
  - `box` often `[x1, y1, x2, y2]` (STA detections/objects).
  - `bbox` often `{x, y, width, height}` (SCOD frames; some hands entries).
  - `boxes` is a list of named objects (e.g., `left_hand`, `right_hand`, `object_of_change`).

- `objects` (STA): List of objects relevant to anticipation with fields:
  - `box`: Bounding box as `[x1, y1, x2, y2]`.
  - `verb_category_id`: Verb class index.
  - `noun_category_id`: Noun class index.
  - `time_to_contact`: Time (sec) until contact from the given pre‑frame.

- `verb` / `noun`: Canonical verb/noun labels (LTA); see taxonomy files for full label set.
- `verb_label` / `noun_label`: Integer class indices matching the taxonomy order.

- `narrations`: List of narration entries aligned to time with fields like:
  - `timestamp_sec`, `timestamp_frame`: When the narration applies.
  - `narration_text`: Text (may include `#C` camera wearer / `#O` other tags).
  - `annotation_uid`: Narration annotation identifier.

- `annotations` (varies by task):
  - NLQ: `language_queries` with `template`, `query`, `slot_x`, optional `slot_y`, `verb_x`, optional `verb_y`, and aligned `clip_*` / `video_*` time/frame bounds.
  - VQ: `query_sets` keyed by ID; includes `query_frame`, `query_video_frame`, `visual_crop` (crop box at query), `response_track` (per‑frame `[x,y,width,height]` + frame numbers), optional `object_title`, and validity flags.
  - Moments: `labels` per annotator with `{start_time, end_time, label, video_start_time, video_end_time, video_start_frame, video_end_frame, primary}`.

- `camera_wearer`: Block for AV with voice/talking/looking segments for the camera wearer.
- `voice_segments` (AV): Segments of wearer speech with `start_time`, `end_time`, plus frame/time alignment.
- `persons`, `social_segments_talking`, `social_segments_looking`: Optional per‑person social interaction tracks.

- `video_metadata`: Global media info for a video; expect `fps`, `num_frames`, `display_resolution_*`, `sample_resolution_*`, codecs, mp4 duration, internal timing bases (`video_base_*`, `audio_base_*`, PTS fields).
- `video_components`: Sub‑chunks of a video with their own `canonical_video_*` ranges and optional sub‑metadata.

- `benchmarks` (features manifest): Which benchmarks a feature file applies to (e.g., `[FHO]`).

- STA detections (in `sta_models/object_detections_json.csv`): rows contain `{box, score, noun_category_id}` under keys grouped by `<clip_uid>_<frame>`.

For a per‑file list of discovered keys, open `ego4d_samples/_auto_fields.md` (generated from the current samples in this repo). If you’d like, I can keep this file auto‑updated via a tiny script or add compact schema tables per task section above.
## CSV Columns — What Each File Contains

Most “JSON-as-CSV” samples here use a single logical column whose header indicates the payload type (e.g., `videos`, `clips`, `annotations`). The cell value is a JSON-like string representing a Python dict. Below lists the concrete columns for each file family and notable exceptions.

General JSON-as-CSV patterns
- `.../narration_json.csv`: usually organized as pairs per record — a key line (often `video_uid,clip_uid`) followed by a line containing a JSON string with a `narrations` array. Treat as two-column plus payload when parsing; not a strict CSV table.
- `.../fho_main_json.csv`: header `videos`; rows contain a dict with `annotated_intervals`, `video_metadata`, `video_uid`.
- `.../fho_sta_*_json.csv`: header `annotations`; each row is a dict for one STA sample with identifiers, timing, and `objects`.
- `.../fho_lta_*_json.csv`: header `clips`; each row is a dict with clip bounds and a single action label instance (`verb`, `noun`, labels, timing).
- `.../fho_oscc-pnr_*_json.csv`: header `clips`; rows specify clip timing and PNR fields.
- `.../fho_scod_*_json.csv`: header `clips`; rows bundle `pre_frame`, `pnr_frame`, `post_frame` each with image size and per-object bounding boxes.
- `.../fho_hands_*_json.csv`: header `clips`; rows provide key frames (e.g., `pre_45`, `pre_30`, `pre_15`, `pre_frame`, `pnr_frame`, `post_frame`) with hand coordinates/boxes.
- `.../nlq_*_json.csv`: header `videos`; rows have `clips` with `annotations.language_queries` entries and their temporal alignment.
- `.../vq_*_json.csv`: header `videos`; rows have `clips` with `annotations.query_sets` containing `visual_crop`, `response_track`, etc.
- `.../av_*_json.csv`: header `videos`; rows include `camera_wearer.voice_segments` with times and frames.
- `.../moments_*_json.csv`: header `videos`; rows contain per-annotator `labels` with start/end times and frames.

Taxonomy CSVs (true CSV tables)
- `.../narration_noun_taxonomy_csv.csv`
  - Columns: `label` (string), `group` (stringified list of synonyms/categories)
- `.../narration_verb_taxonomy_csv.csv`
  - Columns: `label` (string), `group` (stringified list)
- `.../fho_lta_taxonomy_json.csv`
  - Column: `verbs` (one verb label per row)
- `.../fho_main_taxonomy_json.csv`
  - Column: `nouns` (one noun label per row)

Manifests (true CSV tables)
- `v1/annotations/manifest_csv.csv`, `v2/annotations/manifest_csv.csv`
  - Columns: `file_uid` (UUID of file), `annotation` (logical name), `type` (always `file` here), `canonical_s3_location` (S3 path to the full JSON)
- `v1/omnivore_video_swinl_fp16/manifest_csv.csv`, `v2/omnivore_video_swinl_fp16/manifest_csv.csv`
  - Columns: `video_uid` (UUID), `type` (`file`), `s3_path` (S3 path to `.pt` features), `benchmarks` (e.g., `[FHO]`)
- `v2/sta_models/manifest_csv.csv`, `v1/sta_models/manifest_csv.csv`
  - Columns: `video_uid` (UUID or artifact key), `type` (`file`), `canonical_s3_location` (S3 path to detections/model)

Special formats (block-structured CSV-ish)
- `v2/annotations/narration_json.csv`
  - Observed rows appear in pairs: a line with `video_uid,clip_uid`, then a line with a single JSON payload containing `{narrations: [...]}`. Treat as key/value blocks rather than a flat table.
- `v1/annotations/narration_json.csv`
  - Same pattern as v2 in these samples.
- `v1|v2/sta_models/object_detections_json.csv`
  - Organized as blocks keyed by a frame identifier like `<clip_uid>_<frame_number>`. The key line is followed by one or more JSON rows, each detection containing `box` `[x1,y1,x2,y2]`, `score`, `noun_category_id`. Parse by grouping rows until the next non-quoted key line.
- `v2/annotations/all_narrations_redacted_json.csv`
  - Column header: `_map_errs` in sample; each row is a JSON string with at least `text`, `is_summary`, `_clip_time_start`, `_clip_time_end`, `annotator`, `_annotation_uid`.

Top-level catalog (JSON-as-CSV)
- `ego4d_samples/ego4d_json.csv`
  - Column: `videos` (each row is one video dict with `video_uid`, `duration_sec`, `scenarios`, `video_metadata`, `s3_path`, `video_components`, etc.)

Notes on parsing
- For single-column JSON rows, read the header name (e.g., `videos`, `clips`, `annotations`) and parse the cell value with a tolerant parser (e.g., Python `ast.literal_eval`).
- For “pair” or “block” files (e.g., `narration_json.csv`, `object_detections_json.csv`), iterate line-by-line and assemble records by grouping key lines with the following JSON payload lines.
