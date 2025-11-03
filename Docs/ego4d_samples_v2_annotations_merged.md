# ego4d_samples/v2/annotations — Merged Samples
One example per sample CSV, with the mapped original annotation file (from file_listings/ego4d_file_listing.csv) and a short schema.

## all_narrations_redacted_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/all_narrations_redacted.json
Header:
```
_map_errs
```
Sample row:
```
{'text': '#C C holds knitting sticks by hand.', 'is_summary': False, '_clip_time_start': 0, '_clip_time_end': 0, 'annotator': '4130600123644721', '_project_id': '3111279962304712', '_annotation_uid': '385dff3f-de20-4a80-9de9-5cc25bde2abe'}
```
Short schema:
- top-level keys: _annotation_uid, _clip_time_end, _clip_time_start, _project_id, annotator, is_summary, text

## av_test_unannotated_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/av_test_unannotated.json
Header:
```
videos
```
Sample row:
```
{'video_uid': 'ec4b530e-f01c-420a-915d-4a11bc26c3ae', 'split': 'test', 'clips': [{'clip_uid': '153f5883-8ed4-4a57-bae0-78ad5d020571', 'source_clip_uid': 'cecddacc-8fd1-45cd-843c-3ba39ffd82a3', 'video_uid': 'ec4b530e-f01c-420a-915d-4a11bc26c3ae', 'video_start_sec': 1259.9876952666666, 'video_end_sec': 1559.9876952666666, 'video_start_frame': 37800, 'video_end_frame': 46800, 'clip_start_sec': 0, 'clip_end_sec': 300.0, 'clip_start_frame': 0, 'clip_end_frame': 9000, 'valid': True, 'camera_wearer': None, 'persons': [], 'missing_voice_segments': [], 'transcriptions': None, 'social_segments_talking': None, 'social_segments_looking': None}, {'clip_uid': 'e03a39eb-52f2-4c86-af99-c6a00baea7e1', 'source_clip_uid': '47f5181c-c613-46b1-99ea-c97315c4485e', 'video_uid': 'ec4b530e-f01c-420a-915d-4a11bc26c3ae', 'video_start_sec': 1559.9876952666666, 'video_end_sec': 1859.9876952666666, 'video_start_frame': 46800, 'video_end_frame': 55800, 'clip_start_sec': 0, 'clip_end_sec': 300.0, 'clip_start_frame': 0, 'clip_end_frame': 9000, 'valid': True, 'camera_wearer': None, 'persons': [], 'missing_voice_segments': [], 'transcriptions': None, 'social_segments_talking': None, 'social_segments_looking': None}, {'clip_uid': 'c6d8d765-fa5d-4ec5-9feb-f6f294271c25', 'source_clip_uid': '14c263b6-4452-4971-af4e-d3e8b8e13ab9', 'video_uid': 'ec4b530e-f01c-420a-915d-4a11bc26c3ae', 'video_start_sec': 2459.987695266667, 'video_end_sec': 2759.987695266667, 'video_start_frame': 73800, 'video_end_frame': 82800, 'clip_start_sec': 0, 'clip_end_sec': 300.0, 'clip_start_frame': 0, 'clip_end_frame': 9000, 'valid': True, 'camera_wearer': None, 'persons': [], 'missing_voice_segments': [], 'transcriptions': None, 'social_segments_talking': None, 'social_segments_looking': None}, {'clip_uid': '907def38-008c-4516-802e-b549ac5764f3', 'source_clip_uid': '4687e789-31c4-4baa-a053-12aeea90b079', 'video_uid': 'ec4b530e-f01c-420a-915d-4a11bc26c3ae', 'video_start_sec': 6659.987695266666, 'video_end_sec': 6959.987695266666, 'v ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, source_clip_uid, video_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame

## av_train_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/av_train.json
Header:
```
videos
```
Sample row:
```
{'video_uid': '733ac083-1962-46c3-9a46-1efee74fee90', 'split': 'train', 'clips': [{'clip_uid': 'fc2b2014-9dc4-4a5d-8a1d-25a6911bff7c', 'source_clip_uid': '417d4e6e-6766-4df1-8156-625c34845145', 'video_uid': '733ac083-1962-46c3-9a46-1efee74fee90', 'video_start_sec': 80.99869786666667, 'video_end_sec': 380.96536453333334, 'video_start_frame': 2430, 'video_end_frame': 11429, 'clip_start_sec': 0, 'clip_end_sec': 299.9666666666667, 'clip_start_frame': 0, 'clip_end_frame': 8999, 'valid': True, 'camera_wearer': {'person_id': '0', 'camera_wearer': True, 'tracking_paths': [], 'voice_segments': [{'start_time': 26.56722, 'end_time': 27.14449, 'start_frame': 797, 'end_frame': 814, 'video_start_time': 107.56591786666667, 'video_end_time': 108.14318786666668, 'video_start_frame': 3227, 'video_end_frame': 3244, 'person': '0'}, {'start_time': 28.36942, 'end_time': 28.74956, 'start_frame': 851, 'end_frame': 862, 'video_start_time': 109.36811786666668, 'video_end_time': 109.74825786666668, 'video_start_frame': 3281, 'video_end_frame': 3292, 'person': '0'}, {'start_time': 32.67486, 'end_time': 32.95219, 'start_frame': 980, 'end_frame': 988, 'video_start_time': 113.67355786666667, 'video_end_time': 113.95088786666668, 'video_start_frame': 3410, 'video_end_frame': 3418, 'person': '0'}, {'start_time': 40.74274, 'end_time': 44.48279, 'start_frame': 1222, 'end_frame': 1334, 'video_start_time': 121.74143786666667, 'video_end_time': 125.48148786666667, 'video_start_frame': 3652, 'video_end_frame': 3764, 'person': '0'}, {'start_time': 52.27891, 'end_time': 53.01135, 'start_frame': 1568, 'end_frame': 1590, 'video_start_time': 133.27760786666667, 'video_end_time': 134.01004786666667, 'video_start_frame': 3998, 'video_end_frame': 4020, 'person': '0'}, {'start_time': 56.72592, 'end_time': 57.06421, 'start_frame': 1701, 'end_frame': 1711, 'video_start_time': 137.72461786666668, 'video_end_time': 138.06290786666668, 'video_start_frame': 4131, 'video_end_frame': 4141, 'person': '0'}, {'start_time':  ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, source_clip_uid, video_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame

## av_val_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/av_val.json
Header:
```
videos
```
Sample row:
```
{'video_uid': '85775377-b334-4bd7-8cfc-16885099cc9a', 'split': 'val', 'clips': [{'clip_uid': 'c2413391-7c1b-4fd6-8b1d-98ee7888b9f8', 'source_clip_uid': 'ee778366-fbe4-46e8-a53b-28eb8ce3a31c', 'video_uid': '85775377-b334-4bd7-8cfc-16885099cc9a', 'video_start_sec': 414.96536453333334, 'video_end_sec': 714.9320312, 'video_start_frame': 12449, 'video_end_frame': 21448, 'clip_start_sec': 0, 'clip_end_sec': 299.96666666666664, 'clip_start_frame': 0, 'clip_end_frame': 8999, 'valid': True, 'camera_wearer': {'person_id': '0', 'camera_wearer': True, 'tracking_paths': [], 'voice_segments': [{'start_time': 7.74067, 'end_time': 8.233, 'start_frame': 232, 'end_frame': 246, 'video_start_time': 422.70603453333337, 'video_end_time': 423.19836453333335, 'video_start_frame': 12681, 'video_end_frame': 12695, 'person': '0'}, {'start_time': 27.874, 'end_time': 28.10733, 'start_frame': 836, 'end_frame': 843, 'video_start_time': 442.83936453333337, 'video_end_time': 443.07269453333333, 'video_start_frame': 13285, 'video_end_frame': 13292, 'person': '0'}, {'start_time': 111.174, 'end_time': 111.61922, 'start_frame': 3335, 'end_frame': 3348, 'video_start_time': 526.1393645333333, 'video_end_time': 526.5845845333333, 'video_start_frame': 15784, 'video_end_frame': 15797, 'person': '0'}, {'start_time': 125.10733, 'end_time': 125.36846, 'start_frame': 3753, 'end_frame': 3761, 'video_start_time': 540.0726945333333, 'video_end_time': 540.3338245333334, 'video_start_frame': 16202, 'video_end_frame': 16210, 'person': '0'}, {'start_time': 169.374, 'end_time': 170.23879, 'start_frame': 5081, 'end_frame': 5107, 'video_start_time': 584.3393645333333, 'video_end_time': 585.2041545333334, 'video_start_frame': 17530, 'video_end_frame': 17556, 'person': '0'}, {'start_time': 178.99343, 'end_time': 184.12878, 'start_frame': 5369, 'end_frame': 5523, 'video_start_time': 593.9587945333333, 'video_end_time': 599.0941445333333, 'video_start_frame': 17818, 'video_end_frame': 17972, 'person': '0'}, {'start_time': 19 ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, source_clip_uid, video_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame

## fho_hands_test_unannotated_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_hands_test_unannotated.json
Header:
```
clips
```
Sample row:
```
{'clip_id': 3, 'clip_uid': 'de4a85e5-1809-4547-ae11-9161aa9fdbe9', 'video_uid': '31ea56e6-7a99-4b91-b74b-3b1e6de05e99', 'frames': [{'pre_45': {'frame': 36278, 'clip_frame': 519}, 'pre_30': {'frame': 36293, 'clip_frame': 534}, 'pre_15': {'frame': 36308, 'clip_frame': 549}, 'pre_frame': {'frame': 36323, 'clip_frame': 564}}, {'pre_45': {'frame': 36481, 'clip_frame': 722}, 'pre_30': {'frame': 36496, 'clip_frame': 737}, 'pre_15': {'frame': 36511, 'clip_frame': 752}, 'pre_frame': {'frame': 36526, 'clip_frame': 767}}, {'pre_45': {'frame': 37019, 'clip_frame': 1260}, 'pre_30': {'frame': 37034, 'clip_frame': 1275}, 'pre_15': {'frame': 37049, 'clip_frame': 1290}, 'pre_frame': {'frame': 37064, 'clip_frame': 1305}}, {'pre_45': {'frame': 37174, 'clip_frame': 1415}, 'pre_30': {'frame': 37189, 'clip_frame': 1430}, 'pre_15': {'frame': 37204, 'clip_frame': 1445}, 'pre_frame': {'frame': 37219, 'clip_frame': 1460}}, {'pre_45': {'frame': 37577, 'clip_frame': 1818}, 'pre_30': {'frame': 37592, 'clip_frame': 1833}, 'pre_15': {'frame': 37607, 'clip_frame': 1848}, 'pre_frame': {'frame': 37622, 'clip_frame': 1863}}, {'pre_45': {'frame': 38257, 'clip_frame': 2498}, 'pre_30': {'frame': 38272, 'clip_frame': 2513}, 'pre_15': {'frame': 38287, 'clip_frame': 2528}, 'pre_frame': {'frame': 38302, 'clip_frame': 2543}}, {'pre_45': {'frame': 38606, 'clip_frame': 2847}, 'pre_30': {'frame': 38621, 'clip_frame': 2862}, 'pre_15': {'frame': 38636, 'clip_frame': 2877}, 'pre_frame': {'frame': 38651, 'clip_frame': 2892}}, {'pre_45': {'frame': 39367, 'clip_frame': 3608}, 'pre_30': {'frame': 39382, 'clip_frame': 3623}, 'pre_15': {'frame': 39397, 'clip_frame': 3638}, 'pre_frame': {'frame': 39412, 'clip_frame': 3653}}, {'pre_45': {'frame': 39880, 'clip_frame': 4121}, 'pre_30': {'frame': 39895, 'clip_frame': 4136}, 'pre_15': {'frame': 39910, 'clip_frame': 4151}, 'pre_frame': {'frame': 39925, 'clip_frame': 4166}}, {'pre_45': {'frame': 40616, 'clip_frame': 4857}, 'pre_30': {'frame': 40631, 'clip_frame': 4872}, 'pre_15 ... [truncated]
```
Short schema:
- top-level keys: clip_id, clip_uid, frames, video_uid
- nested frames: pre_45, pre_30, pre_15, pre_frame

## fho_lta_taxonomy_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_lta_taxonomy.json
Header:
```
verbs
```
Sample row:
```
adjust_(regulate,_increase/reduce,_change)
```
Short schema:
- top-level keys: 

## fho_lta_test_unannotated_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_lta_test_unannotated.json
Header:
```
clips
```
Sample row:
```
{'video_uid': '9c59e912-2340-4400-b2df-7db3d4066723', 'clip_uid': '77af34ff-5dad-4912-98ed-2d4492be0666', 'clip_parent_start_sec': 292.0, 'clip_parent_end_sec': 608.0, 'clip_parent_start_frame': 8759, 'clip_parent_end_frame': 18239, 'interval_start_frame': 8999, 'interval_end_frame': 17999, 'interval_start_sec': 300.0, 'interval_end_sec': 600.0, 'action_clip_start_sec': 12.721028600000011, 'action_clip_end_sec': 20.72102860000001, 'action_clip_start_frame': 382, 'action_clip_end_frame': 622, 'clip_id': 1806, 'action_idx': 0}
```
Short schema:
- top-level keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_idx, clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, video_uid

## fho_lta_train_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_lta_train.json
Header:
```
clips
```
Sample row:
```
{'video_uid': '4c642620-db0e-4096-9ece-2b2c6fdb47b0', 'clip_uid': '5f4558be-ecb4-4597-871f-cf1221301f91', 'clip_parent_start_sec': 292.0, 'clip_parent_end_sec': 608.0, 'clip_parent_start_frame': 8760, 'clip_parent_end_frame': 18240, 'interval_start_frame': 9000, 'interval_end_frame': 18000, 'interval_start_sec': 300.0, 'interval_end_sec': 600.0, 'verb': 'touch', 'noun': 'dough', 'action_clip_start_sec': 4.7666666666666515, 'action_clip_end_sec': 12.766666666666652, 'action_clip_start_frame': 143, 'action_clip_end_frame': 383, 'action_idx': 0, 'verb_label': 98, 'noun_label': 131}
```
Short schema:
- top-level keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_idx, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, noun, noun_label, verb, verb_label, video_uid

## fho_lta_val_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_lta_val.json
Header:
```
clips
```
Sample row:
```
{'video_uid': '99a632bf-2f91-4e9a-9974-837dcf2a0fea', 'clip_uid': 'c66c30dd-f7c1-492a-9649-84e9147a3fd2', 'clip_parent_start_sec': 592.0, 'clip_parent_end_sec': 908.0, 'clip_parent_start_frame': 17759, 'clip_parent_end_frame': 27239, 'interval_start_frame': 17999, 'interval_end_frame': 26999, 'interval_start_sec': 600.0, 'interval_end_sec': 900.0, 'verb': 'arrange_(straighten,_sort,_distribute,_align)', 'noun': 'bag_(bag,_grocery,_nylon,_polythene,_pouch,_sachet,_sack,_suitcase)', 'action_clip_start_sec': 47.02102860000002, 'action_clip_end_sec': 55.02102860000002, 'action_clip_start_frame': 1411, 'action_clip_end_frame': 1651, 'action_idx': 0, 'verb_label': 2, 'noun_label': 10}
```
Short schema:
- top-level keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_idx, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec, noun, noun_label, verb, verb_label, video_uid

## fho_main_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_main.json
Header:
```
videos
```
Sample row:
```
{'annotated_intervals': [{'clip_id': '451', 'clip_uid': 'a102c79b-405b-4a13-b21e-ab7dc6135b22', 'start_sec': 0.0, 'end_sec': 207.63, 'clip_parent_start_sec': 0.0, 'clip_parent_end_sec': 207.633, 'narrated_actions': [], 'start_frame': 0, 'end_frame': 6229, 'clip_parent_start_frame': 0, 'clip_parent_end_frame': 6229, 'redacted': True}], 'video_metadata': {'video_start_pts': 0, 'video_base_numerator': 1, 'video_base_denominator': 15360, 'duration_sec': 207.633, 'num_frames': 6229, 'fps': 30.0, 'width': 2560, 'height': 1920, 'rotation': None}, 'video_uid': 'ee6f0404-9c97-4240-bd7b-2dcb340031d5'}
```
Short schema:
- top-level keys: annotated_intervals, video_metadata, video_uid
- nested video_metadata: video_start_pts, video_base_numerator, video_base_denominator, duration_sec, num_frames, fps, width, height, rotation

## fho_main_taxonomy_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_main_taxonomy.json
Header:
```
nouns
```
Sample row:
```
squirrel
```
Short schema:
- top-level keys: 

## fho_oscc-pnr_test_unannotated_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_oscc-pnr_test_unannotated.json
Header:
```
clips
```
Sample row:
```
{'clip_id': '1397', 'unique_id': '1397-230-238-1397', 'video_uid': 'a6bd7096-ada1-4fe1-a81b-3b7609a39f31', 'parent_start_sec': 230.63333333333333, 'parent_end_sec': 238.0, 'parent_start_frame': 6919, 'parent_end_frame': 7140}
```
Short schema:
- top-level keys: clip_id, parent_end_frame, parent_end_sec, parent_start_frame, parent_start_sec, unique_id, video_uid

## fho_oscc-pnr_train_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_oscc-pnr_train.json
Header:
```
clips
```
Sample row:
```
{'clip_uid': None, 'clip_id': '1042', 'unique_id': 'c31c6a6f-e65d-449f-a624-f9dceef9057f-1_1543619333333333-9_154361933333332', 'video_uid': '4e3fc1e9-424f-4921-9068-d468c135f347', 'clip_start_sec': 1.1543619333333333, 'clip_end_sec': 9.154361933333332, 'clip_start_frame': 34, 'clip_end_frame': 274, 'clip_pnr_frame': 157, 'parent_start_sec': 1.1543619333333333, 'parent_end_sec': 9.154361933333332, 'parent_start_frame': 34, 'parent_end_frame': 274, 'state_change': True, 'parent_pnr_frame': 157}
```
Short schema:
- top-level keys: clip_end_frame, clip_end_sec, clip_id, clip_pnr_frame, clip_start_frame, clip_start_sec, clip_uid, parent_end_frame, parent_end_sec, parent_pnr_frame, parent_start_frame, parent_start_sec, state_change, unique_id, video_uid

## fho_oscc-pnr_val_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_oscc-pnr_val.json
Header:
```
clips
```
Sample row:
```
{'clip_uid': '857978a5-cab3-4449-928b-6da304c5f7dc', 'clip_id': '622', 'unique_id': '5f0b3bcc-54fb-4cd6-a480-294073e7f492-1496_4-1504_4', 'video_uid': '7c0c0057-3bbc-40ef-a91e-4d6e1a637642', 'clip_start_sec': 5.2333333333333485, 'clip_end_sec': 12.400000000000091, 'clip_start_frame': 157, 'clip_end_frame': 372, 'clip_pnr_frame': 268, 'parent_start_sec': 1497.2333333333333, 'parent_end_sec': 1504.4, 'parent_start_frame': 44917, 'parent_end_frame': 45132, 'state_change': True, 'parent_pnr_frame': 45028}
```
Short schema:
- top-level keys: clip_end_frame, clip_end_sec, clip_id, clip_pnr_frame, clip_start_frame, clip_start_sec, clip_uid, parent_end_frame, parent_end_sec, parent_pnr_frame, parent_start_frame, parent_start_sec, state_change, unique_id, video_uid

## fho_sta_test_unannotated_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_sta_test_unannotated.json
Header:
```
annotations
```
Sample row:
```
{'uid': '9c59e912-2340-4400-b2df-7db3d4066723_0011727', 'frame': 11727, 'clip_id': 1806, 'clip_uid': '77af34ff-5dad-4912-98ed-2d4492be0666', 'clip_frame': 2968, 'video_uid': '9c59e912-2340-4400-b2df-7db3d4066723'}
```
Short schema:
- top-level keys: clip_frame, clip_id, clip_uid, frame, uid, video_uid

## fho_sta_train_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_sta_train.json
Header:
```
annotations
```
Sample row:
```
{'uid': '26202090-684d-4be8-b3cc-de04da827e91_0000984', 'main_uid': '3b867f4f-958b-4735-aecd-5984998658a7', 'video_uid': '26202090-684d-4be8-b3cc-de04da827e91', 'frame': 984, 'clip_id': 121, 'clip_uid': '4f68183f-610a-44de-b102-e7f300b49dcd', 'clip_frame': 984, 'action_start_sec': 30.654361933333334, 'action_end_sec': 38.654361933333334, 'action_start_frame': 919, 'action_end_frame': 1159, 'action_clip_start_sec': 30.654361933333334, 'action_clip_end_sec': 38.654361933333334, 'action_clip_start_frame': 919, 'action_clip_end_frame': 1159, 'interval_start_frame': 0, 'interval_end_frame': 8999, 'interval_start_sec': 0.0, 'interval_end_sec': 300.0, 'clip_parent_start_sec': 0.0, 'clip_parent_end_sec': 308.0, 'clip_parent_start_frame': 0, 'clip_parent_end_frame': 9239, 'objects': [{'box': [235.32, 43.3, 581.69, 440.18], 'verb_category_id': 34, 'noun_category_id': 1, 'time_to_contact': 1.7666666666666666}]}
```
Short schema:
- top-level keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_end_frame, action_end_sec, action_start_frame, action_start_sec, clip_frame, clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, frame, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec
- nested objects: box, verb_category_id, noun_category_id, time_to_contact

## fho_sta_val_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/fho_sta_val.json
Header:
```
annotations
```
Sample row:
```
{'uid': 'd8c894ab-7b08-4983-9e80-fdb5d6ee0202_0149227', 'main_uid': 'f06f4635-9f5e-4cf5-9482-d6be03c6f786', 'video_uid': 'd8c894ab-7b08-4983-9e80-fdb5d6ee0202', 'frame': 149227, 'clip_id': 930, 'clip_uid': 'a8f95c5f-7d7b-419b-af29-60d97d8fe379', 'clip_frame': 241, 'action_start_sec': 4971.887695266666, 'action_end_sec': 4979.887695266666, 'action_start_frame': 149156, 'action_end_frame': 149396, 'action_clip_start_sec': 5.677695266665978, 'action_clip_end_sec': 13.677695266665978, 'action_clip_start_frame': 170, 'action_clip_end_frame': 410, 'interval_start_frame': 149226, 'interval_end_frame': 151901, 'interval_start_sec': 4974.21, 'interval_end_sec': 5063.38, 'clip_parent_start_sec': 4966.21, 'clip_parent_end_sec': 5063.391028645833, 'clip_parent_start_frame': 148986, 'clip_parent_end_frame': 151901, 'objects': [{'box': [591.8, 2.59, 1436.12, 753.6700000000001], 'verb_category_id': 34, 'noun_category_id': 20, 'time_to_contact': 1.8}]}
```
Short schema:
- top-level keys: action_clip_end_frame, action_clip_end_sec, action_clip_start_frame, action_clip_start_sec, action_end_frame, action_end_sec, action_start_frame, action_start_sec, clip_frame, clip_id, clip_parent_end_frame, clip_parent_end_sec, clip_parent_start_frame, clip_parent_start_sec, clip_uid, frame, interval_end_frame, interval_end_sec, interval_start_frame, interval_start_sec
- nested objects: box, verb_category_id, noun_category_id, time_to_contact

## manifest_csv.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/manifest.csv
Header:
```
file_uid,annotation,type,canonical_s3_location
```
Sample row:
```
8a7d977a-293e-4411-a56f-56dff0a907cd,narrations,file,s3://ego4d-consortium-sharing/public/v2/annotations/narration.json
```
Short schema:
- top-level keys: 

## moments_test_unannotated_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/moments_test_unannotated.json
Header:
```
videos
```
Sample row:
```
{'video_uid': '9faf5095-8741-4b2d-8b2e-e803467c7130', 'split': 'test', 'clips': [{'clip_uid': '375e2d5b-31fa-4d21-8aa9-8bdfba124681', 'video_start_sec': 1800.0210286, 'video_end_sec': 2280.0210286, 'video_start_frame': 54001, 'video_end_frame': 68401, 'clip_start_sec': 0, 'clip_end_sec': 480.0000000000002, 'clip_start_frame': 0, 'clip_end_frame': 14400, 'source_clip_uid': 'd2c6f0e5-96e9-4ab1-9f49-62b19a62f473'}]}
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, source_clip_uid

## moments_train_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/moments_train.json
Header:
```
videos
```
Sample row:
```
{'video_uid': 'dd08bc58-b614-4ba7-b883-a213560621dd', 'split': 'train', 'clips': [{'clip_uid': '9df49083-577b-43f9-9874-6e4b21f104b4', 'video_start_sec': 0.0, 'video_end_sec': 349.0, 'video_start_frame': 0, 'video_end_frame': 10470, 'clip_start_sec': 0, 'clip_end_sec': 349.0, 'clip_start_frame': 0, 'clip_end_frame': 10470, 'source_clip_uid': '9aa7175c-3244-4289-83d3-cd01979395b3', 'annotations': [{'annotator_uid': '1829951177105760', 'labels': [{'start_time': 347.62496, 'end_time': 349, 'label': 'use_phone', 'video_start_time': 347.62496, 'video_end_time': 349.0, 'video_start_frame': 10429, 'video_end_frame': 10470, 'primary': True}, {'start_time': 45.79368, 'end_time': 47.67616, 'label': 'water_soil_/_plants_/_crops', 'video_start_time': 45.79368, 'video_end_time': 47.67616, 'video_start_frame': 1374, 'video_end_frame': 1430, 'primary': True}, {'start_time': 334.38207, 'end_time': 345.99902, 'label': 'clean_/_wipe_a_table_or_kitchen_counter', 'video_start_time': 334.38207, 'video_end_time': 345.99902, 'video_start_frame': 10031, 'video_end_frame': 10380, 'primary': True}, {'start_time': 287.47733, 'end_time': 302.6363, 'label': 'walk_down_stairs_/_walk_up_stairs', 'video_start_time': 287.47733, 'video_end_time': 302.6363, 'video_start_frame': 8624, 'video_end_frame': 9079, 'primary': True}]}, {'annotator_uid': '2897148146987010', 'labels': [{'start_time': 187.25048, 'end_time': 215.36786, 'label': 'arrange_/_organize_other_items', 'video_start_time': 187.25048, 'video_end_time': 215.36786, 'video_start_frame': 5618, 'video_end_frame': 6461, 'primary': True}, {'start_time': 287.3921, 'end_time': 301.32925, 'label': 'walk_down_stairs_/_walk_up_stairs', 'video_start_time': 287.3921, 'video_end_time': 301.32925, 'video_start_frame': 8622, 'video_end_frame': 9040, 'primary': True}, {'start_time': 331.8498, 'end_time': 346.28784, 'label': 'clean_/_wipe_other_surface_or_object', 'video_start_time': 331.8498, 'video_end_time': 346.28784, 'video_start_frame': 9955, 'video_e ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, source_clip_uid

## moments_val_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/moments_val.json
Header:
```
videos
```
Sample row:
```
{'video_uid': 'c922a13b-c85e-41e4-a523-f4305f6c0a81', 'split': 'val', 'clips': [{'clip_uid': '63c1d2c4-286a-4b6d-bdf8-8de7a56310a6', 'video_start_sec': 1182.8876952666665, 'video_end_sec': 1662.0210286, 'video_start_frame': 35487, 'video_end_frame': 49861, 'clip_start_sec': 0, 'clip_end_sec': 479.13333333333344, 'clip_start_frame': 0, 'clip_end_frame': 14374, 'source_clip_uid': 'b7fb5f28-d92c-4078-ac74-e742ac474e9c', 'annotations': [{'annotator_uid': '4482258218481544', 'labels': [{'start_time': 116.08176, 'end_time': 123.6333, 'label': 'take_photo_/_record_video_with_a_camera', 'video_start_time': 1298.9694552666665, 'video_end_time': 1306.5209952666664, 'video_start_frame': 38969, 'video_end_frame': 39196, 'primary': True}, {'start_time': 303.26591, 'end_time': 325.3276, 'label': 'hang_clothes_in_closet_/_on_hangers', 'video_start_time': 1486.1536052666665, 'video_end_time': 1508.2152952666665, 'video_start_frame': 44585, 'video_end_frame': 45247, 'primary': True}, {'start_time': 328.68334, 'end_time': 335.42606, 'label': 'browse_through_clothing_items_on_rack_/_shelf_/_hanger', 'video_start_time': 1511.5710352666665, 'video_end_time': 1518.3137552666665, 'video_start_frame': 45348, 'video_end_frame': 45550, 'primary': True}, {'start_time': 402.58138, 'end_time': 452.54089, 'label': 'withdraw_money_from_atm_/_operate_atm', 'video_start_time': 1585.4690752666666, 'video_end_time': 1635.4285852666665, 'video_start_frame': 47564, 'video_end_frame': 49063, 'primary': True}]}, {'annotator_uid': '2811294845609145', 'labels': [{'start_time': 120.85575, 'end_time': 123.834, 'label': 'take_photo_/_record_video_with_a_camera', 'video_start_time': 1303.7434452666664, 'video_end_time': 1306.7216952666665, 'video_start_frame': 39113, 'video_end_frame': 39202, 'primary': True}, {'start_time': 314.66703, 'end_time': 324.329, 'label': 'hang_clothes_in_closet_/_on_hangers', 'video_start_time': 1497.5547252666665, 'video_end_time': 1507.2166952666664, 'video_start_frame': 44927, 'v ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, source_clip_uid

## narration_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/narration.json
Header:
```
77cc4654-4eec-44c6-af05-dbdf71f9a401,3e08beb0-9108-4e77-b2ae-80f91ceac474
```
Sample row:
```
{'narrations': [{'timestamp_sec': 0.0, 'timestamp_frame': 0, '_unmapped_timestamp_sec': 0.0, 'narration_text': '#C C interacts with a woman X', 'annotation_uid': '920182f7-5385-488b-99f9-caf8f0d9fe6b'}, {'timestamp_sec': 4.53806, 'timestamp_frame': 136, '_unmapped_timestamp_sec': 4.53806, 'narration_text': '#C C walks into the kitchen', 'annotation_uid': '920182f7-5385-488b-99f9-caf8f0d9fe6b'}, {'timestamp_sec': 12.92098, 'timestamp_frame': 388, '_unmapped_timestamp_sec': 12.92098, 'narration_text': '#C C opens a shelf', 'annotation_uid': '920182f7-5385-488b-99f9-caf8f0d9fe6b'}, {'timestamp_sec': 15.10264, 'timestamp_frame': 453, '_unmapped_timestamp_sec': 15.10264, 'narration_text': '#C C brings out a basket from the shelf', 'annotation_uid': '920182f7-5385-488b-99f9-caf8f0d9fe6b'}, {'timestamp_sec': 17.15749, 'timestamp_frame': 515, '_unmapped_timestamp_sec': 17.15749, 'narration_text': '#C C puts back the basket into the shelf', 'annotation_uid': '920182f7-5385-488b-99f9-caf8f0d9fe6b'}, {'timestamp_sec': 18.94618, 'timestamp_frame': 568, '_unmapped_timestamp_sec': 18.94618, 'narration_text': '#C C adjusts a basket in the shelf', 'annotation_uid': '920182f7-5385-488b-99f9-caf8f0d9fe6b'}, {'timestamp_sec': 19.55043, 'timestamp_frame': 587, '_unmapped_timestamp_sec': 19.55043, 'narration_text': '#C C closes the shelf', 'annotation_uid': '920182f7-5385-488b-99f9-caf8f0d9fe6b'}, {'timestamp_sec': 23.06551, 'timestamp_frame': 692, '_unmapped_timestamp_sec': 23.06551, 'narration_text': '#C C looks through the window', 'annotation_uid': '920182f7-5385-488b-99f9-caf8f0d9fe6b'}, {'timestamp_sec': 26.83272, 'timestamp_frame': 805, '_unmapped_timestamp_sec': 26.83272, 'narration_text': '#C C walks to the living room', 'annotation_uid': '920182f7-5385-488b-99f9-caf8f0d9fe6b'}, {'timestamp_sec': 30.54697, 'timestamp_frame': 916, '_unmapped_timestamp_sec': 30.54697, 'narration_text': '#C C interacts with a woman X', 'annotation_uid': '920182f7-5385-488b-99f9-caf8f0d9fe6b'}, {'t ... [truncated]
```
Short schema:
- top-level keys: 

## narration_noun_taxonomy_csv.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/narration_noun_taxonomy.csv
Header:
```
label,group
```
Sample row:
```
ambulance,['ambulance']
```
Short schema:
- top-level keys: 

## narration_verb_taxonomy_csv.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/narration_verb_taxonomy.csv
Header:
```
label,group
```
Sample row:
```
adjust_(regulate,_increase/reduce,_change),['adjust', 'change', 'expand', 'increase/reduce', 'level', 'levels', 'low', 'lower', 'lowers', 'raise', 'raises', 'readjust', 'reduce', 'regulate', 'regulates', 'shrink', 'switch', 'tune']
```
Short schema:
- top-level keys: 

## nlq_test_unannotated_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/nlq_test_unannotated.json
Header:
```
videos
```
Sample row:
```
{'video_uid': 'c9c44dea-c37b-461d-aa14-20e934126df5', 'clips': [{'clip_uid': 'a603669a-57f9-4db4-8a81-0a6720946d45', 'video_start_sec': 1489.0943619333332, 'video_end_sec': 1969.1310359242186, 'video_start_frame': 66429, 'video_end_frame': 66429, 'clip_start_sec': 0, 'clip_end_sec': 480.03667399088545, 'clip_start_frame': 0, 'clip_end_frame': 14401, 'source_clip_uid': '4ee7dc88-3d7f-4607-a110-9419fb0eb93d', 'annotations': [{'language_queries': [{'query': 'What did in put in the sack?'}, {'query': 'Who did i talk to in the shop?'}, {'query': 'Where did i put the money?'}, {'query': 'Who did i talk to in the shop?'}, {'query': 'Where is the camera?'}], 'annotation_uid': 'f7e9c1c6-1381-4df2-9121-f0b1f437282d'}, {'language_queries': [{'query': ''}, {'query': 'Which scrapper did i pick?'}, {'query': 'How many white nylons did i pick from the table?'}, {'query': 'Which ball did i put in the white nylon?'}, {'query': 'Where was the torn ball before i picked it?'}, {'query': 'Where is the white bucket filled with water?'}, {'query': 'Where is the brown thread?'}, {'query': 'Where is the yellow screwdriver?'}, {'query': 'What did i drop on the shoes?'}], 'annotation_uid': 'dac51ee2-bdec-4271-9ac9-85be440f5cf6'}]}, {'clip_uid': 'faee7b8e-49ed-4383-9aa1-e0d1141880a7', 'video_start_sec': 2569.0910286000008, 'video_end_sec': 3049.1270519242194, 'video_start_frame': 109905, 'video_end_frame': 109905, 'clip_start_sec': 0, 'clip_end_sec': 480.0360233242186, 'clip_start_frame': 0, 'clip_end_frame': 14401, 'source_clip_uid': '2d2d90ef-1955-4f27-b38c-6b8aed4cab25', 'annotations': [{'language_queries': [{'query': 'What object did I press?'}, {'query': 'What object did I pull?'}, {'query': 'Where was the phone before I picked it?'}, {'query': 'Where was the phone after I dropped it?'}, {'query': 'How many times did I tie the thread?'}, {'query': 'Where was the tool before I picked it?'}, {'query': 'What did I put in the ball?'}, {'query': 'What did I put in the bag?'}], 'annotation_uid' ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, source_clip_uid

## nlq_train_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/nlq_train.json
Header:
```
videos
```
Sample row:
```
{'video_uid': '216e3f0e-ccb9-4d54-ba56-d275fedbf52f', 'clips': [{'clip_uid': 'f06d1935-550f-4caa-909c-b2db4c28f599', 'video_start_sec': 0.0210286, 'video_end_sec': 480.0210286, 'video_start_frame': 0, 'video_end_frame': 14400, 'clip_start_sec': 0, 'clip_end_sec': 480.0, 'clip_start_frame': 0, 'clip_end_frame': 14400, 'source_clip_uid': '2cdc7965-9007-48d3-b6d1-d589179f1670', 'annotations': [{'language_queries': [{'clip_start_sec': 17.25669, 'clip_end_sec': 27.256, 'video_start_sec': 17.2777186, 'video_end_sec': 27.2770286, 'video_start_frame': 518, 'video_end_frame': 818, 'template': 'Objects: What did I put in X?', 'query': 'what did I pick from the fridge?', 'slot_x': 'fridge', 'verb_x': 'pick', 'raw_tags': ['Objects: What did I put in X?', 'what did I pick from the fridge?', 'fridge', 'pick']}, {'clip_start_sec': 56.73617, 'clip_end_sec': 59.932, 'video_start_sec': 56.7571986, 'video_end_sec': 59.9530286, 'video_start_frame': 1702, 'video_end_frame': 1798, 'template': 'Objects: What did I put in X?', 'query': 'what did I pick from the shelf?', 'slot_x': 'shelf', 'verb_x': 'pick', 'raw_tags': ['Objects: What did I put in X?', 'what did I pick from the shelf?', 'shelf', 'pick']}, {'clip_start_sec': 123.17645, 'clip_end_sec': 124.176, 'video_start_sec': 123.1974786, 'video_end_sec': 124.1970286, 'video_start_frame': 3695, 'video_end_frame': 3725, 'template': 'Place: Where did I put X?', 'query': 'where did I put the egg shell?', 'slot_x': 'egg shell', 'verb_x': 'put', 'raw_tags': ['Place: Where did I put X?', 'where did I put the egg shell?', 'egg shell', 'put']}, {'clip_start_sec': 139.21348, 'clip_end_sec': 141.213, 'video_start_sec': 139.2345086, 'video_end_sec': 141.2340286, 'video_start_frame': 4176, 'video_end_frame': 4236, 'template': 'Objects: What did I put in X?', 'query': 'what did I put in the fridge?', 'slot_x': 'fridge', 'verb_x': 'put', 'raw_tags': ['Objects: What did I put in X?', 'what did I put in the fridge?', 'fridge', 'put']}, {'clip_start_sec': ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, source_clip_uid

## nlq_val_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/nlq_val.json
Header:
```
videos
```
Sample row:
```
{'video_uid': '72295d26-19f7-4c6a-874e-85ba8654861e', 'clips': [{'clip_uid': 'cc2d7790-67f7-4e52-9fa9-33121c9431a2', 'video_start_sec': -3.999999999976245e-07, 'video_end_sec': 480.0366735908854, 'video_start_frame': 0, 'video_end_frame': 14400, 'clip_start_sec': 0, 'clip_end_sec': 480.0156449908854, 'clip_start_frame': 0, 'clip_end_frame': 14400, 'source_clip_uid': '806bae1d-3cf4-45b4-b7c1-4230a2929398', 'annotations': [{'language_queries': [{'clip_start_sec': 28.60389, 'clip_end_sec': 29.013, 'video_start_sec': 28.603889600000002, 'video_end_sec': 29.012999600000004, 'video_start_frame': 857, 'video_end_frame': 870, 'template': 'Place: Where did I put X?', 'query': 'Where did i put the fire gun ?', 'slot_x': 'fire gun ', 'verb_x': 'put', 'raw_tags': ['Place: Where did I put X?', 'Where did i put the fire gun ?', 'fire gun ', 'put']}, {'clip_start_sec': 78.29272, 'clip_end_sec': 80.013, 'video_start_sec': 78.2927196, 'video_end_sec': 80.0129996, 'video_start_frame': 2348, 'video_end_frame': 2400, 'template': 'Objects: Where is object X?', 'query': 'Where was the drill machine ?', 'slot_x': 'drill machine', 'verb_x': '[verb_not_applicable]', 'raw_tags': ['Objects: Where is object X?', 'Where was the drill machine ?', 'drill machine', '[verb_not_applicable]']}, {'clip_start_sec': 103.14923, 'clip_end_sec': 103.555, 'video_start_sec': 103.1492296, 'video_end_sec': 103.5549996, 'video_start_frame': 3094, 'video_end_frame': 3106, 'template': 'Place: Where did I put X?', 'query': 'Where did i put the screw driver?', 'slot_x': 'screw driver', 'verb_x': 'put', 'raw_tags': ['Place: Where did I put X?', 'Where did i put the screw driver?', 'screw driver', 'put']}, {'clip_start_sec': 136.147, 'clip_end_sec': 137.204, 'video_start_sec': 136.16802864583332, 'video_end_sec': 137.22502864583333, 'video_start_frame': 4084, 'video_end_frame': 4116, 'template': 'Objects: How many X’s? (quantity question)', 'query': 'How many cables did I drop on the table?', 'slot_x': 'cables did I  ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, source_clip_uid

## vq_test_unannotated_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/vq_test_unannotated.json
Header:
```
videos
```
Sample row:
```
{'video_uid': 'b454135c-b78b-47d1-875c-1f564ef6bd45', 'split': 'test', 'clips': [{'clip_uid': '47962bf9-b324-4954-a415-a316df4d35c8', 'video_start_sec': 0.0210286, 'video_end_sec': 299.98769526666666, 'video_start_frame': 0, 'video_end_frame': 8999, 'clip_start_sec': 0, 'clip_end_sec': 299.96666666666664, 'clip_start_frame': 0, 'clip_end_frame': 8999, 'clip_fps': 5.0, 'annotation_complete': True, 'source_clip_uid': '5b9a9caa-ab14-4d58-aed8-438cab090112', 'annotations': [{'query_sets': {'2': {'is_valid': True, 'errors': [], 'warnings': ['Extra query set annotation'], 'query_frame': 333, 'query_video_frame': 1997, 'object_title': 'bucket', 'visual_crop': {'frame_number': 603, 'x': 526.49, 'y': 780.52, 'width': 220.18, 'height': 270.62, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 3617}}, '1': {'is_valid': True, 'errors': [], 'warnings': ['Extra query set annotation'], 'query_frame': 405, 'query_video_frame': 2429, 'object_title': 'plastic bottle', 'visual_crop': {'frame_number': 184, 'x': 1018.08, 'y': 587.08, 'width': 219.01, 'height': 336.67, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 1103}}, '3': {'is_valid': True, 'errors': [], 'warnings': ['Extra query set annotation'], 'query_frame': 998, 'query_video_frame': 5987, 'object_title': 'plastic lid', 'visual_crop': {'frame_number': 381, 'x': 653.87, 'y': 374.36, 'width': 295.51, 'height': 223.47, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 2285}}}, 'warnings': [], 'annotation_uid': 'c80eb259-b725-4e44-8ca1-c380545f4e9d'}, {'query_sets': {'1': {'is_valid': True, 'errors': [], 'warnings': ['Extra query set annotation'], 'query_frame': 406, 'query_video_frame': 2435, 'object_title': 'persil', 'visual_crop': {'frame_number': 185, 'x': 1011.86, 'y': 614.51, 'width': 235.19, 'height': 333.4, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 1109}}, '3': {'is_valid ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, clip_fps

## vq_train_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/vq_train.json
Header:
```
videos
```
Sample row:
```
{'video_uid': '56a3b49c-6979-4253-9ff5-2733c3e2f229', 'split': 'train', 'clips': [{'clip_uid': '3fe39989-54ed-4a3a-bea4-88be05ff8df7', 'video_start_sec': 269.98769526666666, 'video_end_sec': 569.9876952666667, 'video_start_frame': 8099, 'video_end_frame': 17099, 'clip_start_sec': 0, 'clip_end_sec': 300.00000000000006, 'clip_start_frame': 0, 'clip_end_frame': 9000, 'clip_fps': 5.0, 'annotation_complete': True, 'source_clip_uid': 'e970b901-fd09-4e00-ad86-12bbc04b02fb', 'annotations': [{'query_sets': {'1': {'is_valid': True, 'errors': [], 'warnings': ['Extra query set annotation'], 'query_frame': 1190, 'query_video_frame': 15239, 'response_track': [{'frame_number': 1099, 'x': 929.68, 'y': 540.42, 'width': 11.2, 'height': 41.4, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 14693}, {'frame_number': 1100, 'x': 917.51, 'y': 542.37, 'width': 12.18, 'height': 43.34, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 14699}, {'frame_number': 1101, 'x': 910.69, 'y': 546.26, 'width': 12.18, 'height': 48.12, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 14705}, {'frame_number': 1102, 'x': 930.17, 'y': 544.32, 'width': 18.51, 'height': 51.9, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 14711}, {'frame_number': 1103, 'x': 966.7, 'y': 547.33, 'width': 17.23, 'height': 56.89, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 14717}, {'frame_number': 1104, 'x': 1004.2, 'y': 557.32, 'width': 21.08, 'height': 62.97, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 14723}, {'frame_number': 1105, 'x': 1032.44, 'y': 549.99, 'width': 24.4, 'height': 74.68, 'rotation': 0, 'original_width': 1440, 'original_height': 1080, 'video_frame_number': 14729}, {'frame_number': 1106, 'x': 1066.68, 'y': 542.66, 'width': 34.13, 'height': 87.75, 'rotation': 0, 'original_width': 14 ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, clip_fps

## vq_val_json.csv
Original: /content/drive/MyDrive/ego4d_data/v2/annotations/vq_val.json
Header:
```
videos
```
Sample row:
```
{'video_uid': 'e14e03f8-13e4-4df2-87b0-e1ad8a175f7c', 'split': 'val', 'clips': [{'clip_uid': 'd5935c29-1b8d-417d-9bbb-5ebd47e9256d', 'video_start_sec': 0.0, 'video_end_sec': 480.0, 'video_start_frame': 0, 'video_end_frame': 14400, 'clip_start_sec': 0, 'clip_end_sec': 480.0, 'clip_start_frame': 0, 'clip_end_frame': 14400, 'clip_fps': 5.0, 'annotation_complete': True, 'source_clip_uid': '8828c00f-3899-46ed-bf50-b4c85c4a5baa', 'annotations': [{'query_sets': {'3': {'is_valid': True, 'errors': [], 'warnings': ['Extra query set annotation'], 'query_frame': 1433, 'query_video_frame': 8598, 'response_track': [{'frame_number': 1292, 'x': 1.1, 'y': 1007.51, 'width': 116.43, 'height': 70.69, 'rotation': 0, 'original_width': 1920, 'original_height': 1080, 'video_frame_number': 7752}, {'frame_number': 1293, 'x': -0.23, 'y': 926.14, 'width': 126.4, 'height': 153.53, 'rotation': 0, 'original_width': 1920, 'original_height': 1080, 'video_frame_number': 7758}, {'frame_number': 1294, 'x': -1.38, 'y': 881.55, 'width': 82.32, 'height': 197.79, 'rotation': 0, 'original_width': 1920, 'original_height': 1080, 'video_frame_number': 7764}, {'frame_number': 1295, 'x': -0.23, 'y': 811.81, 'width': 81.18, 'height': 267.54, 'rotation': 0, 'original_width': 1920, 'original_height': 1080, 'video_frame_number': 7770}, {'frame_number': 1296, 'x': 2.05, 'y': 784.37, 'width': 85.75, 'height': 292.69, 'rotation': 0, 'original_width': 1920, 'original_height': 1080, 'video_frame_number': 7776}, {'frame_number': 1297, 'x': 2.05, 'y': 784.37, 'width': 85.75, 'height': 292.69, 'rotation': 0, 'original_width': 1920, 'original_height': 1080, 'video_frame_number': 7782}, {'frame_number': 1298, 'x': 2.05, 'y': 784.37, 'width': 94.37, 'height': 295.25, 'rotation': 0, 'original_width': 1920, 'original_height': 1080, 'video_frame_number': 7788}, {'frame_number': 1299, 'x': 2.05, 'y': 784.37, 'width': 89.84, 'height': 293.75, 'rotation': 0, 'original_width': 1920, 'original_height': 1080, 'video_frame_number': 779 ... [truncated]
```
Short schema:
- top-level keys: clips, split, video_uid
- nested clips: clip_uid, video_start_sec, video_end_sec, video_start_frame, video_end_frame, clip_start_sec, clip_end_sec, clip_start_frame, clip_end_frame, clip_fps
