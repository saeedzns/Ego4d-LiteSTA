# Auto-Parsed Field Summary

## ego4d_samples/ego4d_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: concurrent_sets, device, duration_sec, fb_participant_id, gaps, gaze_metadata, group_video_uids, has_gaze, has_imu, has_redacted_regions, imu_metadata, is_grouped, is_stereo, manifold_path, origin_video_id, physical_setting_name, redacted_intervals, s3_path, scenarios, split_av, split_em, split_fho, split_goalstep, video_components, video_metadata, video_source, video_uid
- nested_example_keys:
  - video_metadata: audio_base_denominator, audio_base_numerator, audio_codec, audio_duration_pts, audio_duration_sec, audio_start_pts, audio_start_sec, display_resolution_height, display_resolution_width, fps, mp4_duration_sec, num_frames, sample_resolution_height, sample_resolution_width, video_base_denominator, video_base_numerator, video_codec, video_duration_pts, video_duration_sec, video_start_pts, video_start_sec
  - video_components: canonical_video_end_frame, canonical_video_end_sec, canonical_video_start_frame, canonical_video_start_sec, component_idx, redacted, video_component_uid, video_metadata, video_uid

## ego4d_samples/v1/annotations/av_test_unannotated_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: camera_wearer, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, missing_voice_segments, persons, social_segments_looking, social_segments_talking, source_clip_uid, transcriptions, valid, video_end_frame, video_end_sec, video_start_frame, video_start_sec, video_uid

## ego4d_samples/v1/annotations/av_train_json.csv
- error: field larger than field limit (131072)

## ego4d_samples/v1/annotations/av_val_json.csv
- error: field larger than field limit (131072)

## ego4d_samples/v1/annotations/fho_hands_test_unannotated_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_id, clip_uid, frames, video_uid
- nested_example_keys:
  - frames: pre_45

## ego4d_samples/v1/annotations/fho_hands_train_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_id, clip_uid, frames, video_uid
- nested_example_keys:
  - frames: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_end_frame, action_end_sec, action_start_frame, action_start_sec, contact_frame, pnr_frame, post_frame, pre_15, pre_30, pre_45, pre_frame

## ego4d_samples/v1/annotations/fho_hands_val_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_id, clip_uid, frames, video_uid
- nested_example_keys:
  - frames: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_end_frame, action_end_sec, action_start_frame, action_start_sec, contact_frame, pnr_frame, post_frame, pre_15, pre_30, pre_45, pre_frame

## ego4d_samples/v1/annotations/fho_lta_taxonomy_json.csv
- header: verbs
- type: json-rows
- column: verbs
- top_level_keys: 

## ego4d_samples/v1/annotations/fho_lta_test_unannotated_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_idx, clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, video_uid

## ego4d_samples/v1/annotations/fho_lta_train_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_idx, clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, noun, noun_label, verb, verb_label, video_uid

## ego4d_samples/v1/annotations/fho_lta_val_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_idx, clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, noun, noun_label, verb, verb_label, video_uid

## ego4d_samples/v1/annotations/fho_oscc-pnr_test_unannotated_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_id, parent_end_frame, parent_end_sec, parent_start_frame, parent_start_sec, unique_id, video_uid

## ego4d_samples/v1/annotations/fho_oscc-pnr_train_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_end_frame, clip_end_sec, clip_id, clip_pnr_frame, clip_start_frame, clip_start_sec, clip_uid, parent_end_frame, parent_end_sec, parent_pnr_frame, parent_start_frame, parent_start_sec, state_change, unique_id, video_uid

## ego4d_samples/v1/annotations/fho_oscc-pnr_val_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_end_frame, clip_end_sec, clip_id, clip_pnr_frame, clip_start_frame, clip_start_sec, clip_uid, parent_end_frame, parent_end_sec, parent_pnr_frame, parent_start_frame, parent_start_sec, state_change, unique_id, video_uid

## ego4d_samples/v1/annotations/fho_scod_test_unannotated_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, pnr_frame, post_frame, pre_frame, video_uid

## ego4d_samples/v1/annotations/fho_scod_train_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, pnr_frame, post_frame, pre_frame, video_uid

## ego4d_samples/v1/annotations/fho_scod_val_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, pnr_frame, post_frame, pre_frame, video_uid

## ego4d_samples/v1/annotations/fho_sta_test_unannotated_json.csv
- header: annotations
- type: json-rows
- column: annotations
- top_level_keys: clip_frame, clip_id, clip_uid, frame, uid, video_id

## ego4d_samples/v1/annotations/fho_sta_train_json.csv
- header: annotations
- type: json-rows
- column: annotations
- top_level_keys: clip_frame, clip_id, clip_uid, frame, objects, uid, video_id
- nested_example_keys:
  - objects: box, noun_category_id, time_to_contact, verb_category_id

## ego4d_samples/v1/annotations/fho_sta_val_json.csv
- header: annotations
- type: json-rows
- column: annotations
- top_level_keys: clip_frame, clip_id, clip_uid, frame, objects, uid, video_id
- nested_example_keys:
  - objects: box, noun_category_id, time_to_contact, verb_category_id

## ego4d_samples/v1/annotations/manifest_csv.csv
- header: file_uid, annotation, type, canonical_s3_location
- type: table
- columns: file_uid, annotation, type, canonical_s3_location

## ego4d_samples/v1/annotations/moments_test_unannotated_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v1/annotations/moments_train_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotations, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v1/annotations/moments_val_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotations, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v1/annotations/narration_json.csv
- header: 77cc4654-4eec-44c6-af05-dbdf71f9a401, 3e08beb0-9108-4e77-b2ae-80f91ceac474
- type: special

## ego4d_samples/v1/annotations/narration_noun_taxonomy_csv.csv
- header: label, group
- type: table
- columns: label, group

## ego4d_samples/v1/annotations/narration_verb_taxonomy_csv.csv
- header: label, group
- type: table
- columns: label, group

## ego4d_samples/v1/annotations/nlq_test_unannotated_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotations, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v1/annotations/nlq_train_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotations, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v1/annotations/nlq_val_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotations, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v1/annotations/vq_test_unannotated_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotation_complete, annotations, clip_end_frame, clip_end_sec, clip_fps, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v1/annotations/vq_train_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotation_complete, annotations, clip_end_frame, clip_end_sec, clip_fps, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v1/annotations/vq_val_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotation_complete, annotations, clip_end_frame, clip_end_sec, clip_fps, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v1/omnivore_video_swinl_fp16/manifest_csv.csv
- header: video_uid, type, s3_path, benchmarks
- type: table
- columns: video_uid, type, s3_path, benchmarks

## ego4d_samples/v1/sta_models/manifest_csv.csv
- header: video_uid, type, canonical_s3_location
- type: special

## ego4d_samples/v1/sta_models/object_detections_json.csv
- header: 9c59e912-2340-4400-b2df-7db3d4066723_0000146
- type: special
- block: object_detections
- detection_keys: box, noun_category_id, score

## ego4d_samples/v2/annotations/all_narrations_redacted_json.csv
- header: _map_errs
- type: json-rows
- column: _map_errs
- top_level_keys: _annotation_uid, _clip_time_end, _clip_time_start, _project_id, annotator, is_summary, text

## ego4d_samples/v2/annotations/av_test_unannotated_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: camera_wearer, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, missing_voice_segments, persons, social_segments_looking, social_segments_talking, source_clip_uid, transcriptions, valid, video_end_frame, video_end_sec, video_start_frame, video_start_sec, video_uid

## ego4d_samples/v2/annotations/av_train_json.csv
- error: field larger than field limit (131072)

## ego4d_samples/v2/annotations/av_val_json.csv
- error: field larger than field limit (131072)

## ego4d_samples/v2/annotations/fho_hands_test_unannotated_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_id, clip_uid, frames, video_uid
- nested_example_keys:
  - frames: pre_45

## ego4d_samples/v2/annotations/fho_lta_taxonomy_json.csv
- header: verbs
- type: json-rows
- column: verbs
- top_level_keys: 

## ego4d_samples/v2/annotations/fho_lta_test_unannotated_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_idx, clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, video_uid

## ego4d_samples/v2/annotations/fho_lta_train_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_idx, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, noun, noun_label, verb, verb_label, video_uid

## ego4d_samples/v2/annotations/fho_lta_val_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_idx, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, noun, noun_label, verb, verb_label, video_uid

## ego4d_samples/v2/annotations/fho_main_json.csv
- error: field larger than field limit (131072)

## ego4d_samples/v2/annotations/fho_main_taxonomy_json.csv
- header: nouns
- type: json-rows
- column: nouns
- top_level_keys: 

## ego4d_samples/v2/annotations/fho_oscc-pnr_test_unannotated_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_id, parent_end_frame, parent_end_sec, parent_start_frame, parent_start_sec, unique_id, video_uid

## ego4d_samples/v2/annotations/fho_oscc-pnr_train_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_end_frame, clip_end_sec, clip_id, clip_pnr_frame, clip_start_frame, clip_start_sec, clip_uid, parent_end_frame, parent_end_sec, parent_pnr_frame, parent_start_frame, parent_start_sec, state_change, unique_id, video_uid

## ego4d_samples/v2/annotations/fho_oscc-pnr_val_json.csv
- header: clips
- type: json-rows
- column: clips
- top_level_keys: clip_end_frame, clip_end_sec, clip_id, clip_pnr_frame, clip_start_frame, clip_start_sec, clip_uid, parent_end_frame, parent_end_sec, parent_pnr_frame, parent_start_frame, parent_start_sec, state_change, unique_id, video_uid

## ego4d_samples/v2/annotations/fho_sta_test_unannotated_json.csv
- header: annotations
- type: json-rows
- column: annotations
- top_level_keys: clip_frame, clip_id, clip_uid, frame, uid, video_uid

## ego4d_samples/v2/annotations/fho_sta_train_json.csv
- header: annotations
- type: json-rows
- column: annotations
- top_level_keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_end_frame, action_end_sec, action_start_frame, action_start_sec, clip_frame, clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, frame, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, main_uid, objects, uid, video_uid
- nested_example_keys:
  - objects: box, noun_category_id, time_to_contact, verb_category_id

## ego4d_samples/v2/annotations/fho_sta_val_json.csv
- header: annotations
- type: json-rows
- column: annotations
- top_level_keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_end_frame, action_end_sec, action_start_frame, action_start_sec, clip_frame, clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, frame, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, main_uid, objects, uid, video_uid
- nested_example_keys:
  - objects: box, noun_category_id, time_to_contact, verb_category_id

## ego4d_samples/v2/annotations/manifest_csv.csv
- header: file_uid, annotation, type, canonical_s3_location
- type: table
- columns: file_uid, annotation, type, canonical_s3_location

## ego4d_samples/v2/annotations/moments_test_unannotated_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v2/annotations/moments_train_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotations, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v2/annotations/moments_val_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotations, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v2/annotations/narration_json.csv
- header: 77cc4654-4eec-44c6-af05-dbdf71f9a401, 3e08beb0-9108-4e77-b2ae-80f91ceac474
- type: special

## ego4d_samples/v2/annotations/narration_noun_taxonomy_csv.csv
- header: label, group
- type: table
- columns: label, group

## ego4d_samples/v2/annotations/narration_verb_taxonomy_csv.csv
- header: label, group
- type: table
- columns: label, group

## ego4d_samples/v2/annotations/nlq_test_unannotated_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotations, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v2/annotations/nlq_train_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotations, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v2/annotations/nlq_val_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotations, clip_end_frame, clip_end_sec, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v2/annotations/vq_test_unannotated_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotation_complete, annotations, clip_end_frame, clip_end_sec, clip_fps, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v2/annotations/vq_train_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotation_complete, annotations, clip_end_frame, clip_end_sec, clip_fps, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v2/annotations/vq_val_json.csv
- header: videos
- type: json-rows
- column: videos
- top_level_keys: clips, split, video_uid
- nested_example_keys:
  - clips: annotation_complete, annotations, clip_end_frame, clip_end_sec, clip_fps, clip_start_frame, clip_start_sec, clip_uid, source_clip_uid, video_end_frame, video_end_sec, video_start_frame, video_start_sec

## ego4d_samples/v2/omnivore_video_swinl_fp16/manifest_csv.csv
- header: video_uid, type, s3_path, benchmarks
- type: table
- columns: video_uid, type, s3_path, benchmarks

## ego4d_samples/v2/sta_models/manifest_csv.csv
- header: video_uid, type, canonical_s3_location
- type: special

## ego4d_samples/v2/sta_models/object_detections_json.csv
- header: 9c59e912-2340-4400-b2df-7db3d4066723_0000146
- type: special
- block: object_detections
- detection_keys: box, noun_category_id, score