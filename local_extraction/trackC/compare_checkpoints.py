#!/usr/bin/env python3
"""Compare November vs December checkpoint structures."""

import torch

# Load both checkpoints
nov = torch.load('d:/Thesis/Ego4d-LiteSTA/local_extraction_old/runs/Track_B/checkpoints/trackB_best_1122_1735.pt', map_location='cpu')
dec = torch.load('d:/Thesis/Ego4d-LiteSTA/local_extraction/runs/Track_B/checkpoints/trackB_best_mAP_0.3580_20251206_083053.pt', map_location='cpu')

print('=== NOVEMBER CHECKPOINT KEYS ===')
print(list(nov.keys()))
print()
print('=== DECEMBER CHECKPOINT KEYS ===') 
print(list(dec.keys()))
print()

# Compare model sizes
print('=== MODEL PARAMETER COUNTS ===')
for key in ['projector', 'fusion', 'head']:
    if key in nov and key in dec:
        nov_params = sum(p.numel() for p in nov[key].values())
        dec_params = sum(p.numel() for p in dec[key].values())
        print(f'{key}: Nov={nov_params:,} Dec={dec_params:,} Same={nov_params==dec_params}')

print()
print('=== NOUN/VERB VOCAB SIZE ===')
print(f"Nov noun_id_list: {len(nov.get('noun_id_list', []))}")
print(f"Dec noun_id_list: {len(dec.get('noun_id_list', []))}")
print(f"Nov verb_id_list: {len(nov.get('verb_id_list', []))}")
print(f"Dec verb_id_list: {len(dec.get('verb_id_list', []))}")

print()
print('=== HEAD STATE DICT KEYS ===')
print('November head keys:', list(nov['head'].keys()))
print('December head keys:', list(dec['head'].keys()))

# Check specific layer shapes
print()
print('=== KEY LAYER SHAPES ===')
for key in nov['head'].keys():
    nov_shape = nov['head'][key].shape
    dec_shape = dec['head'].get(key, torch.empty(0)).shape
    if nov_shape != dec_shape:
        print(f"DIFF {key}: Nov={nov_shape} Dec={dec_shape}")
    else:
        print(f"SAME {key}: {nov_shape}")
