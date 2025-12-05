2. Build VideoMAE Clip Manifest (JSONL)

Task: create a manifest that Colab reads directly to know clip locations and metadata.

Target file:

ego4d_sta_clips.jsonl

Each line: one JSON object, e.g.:

{
  "uid": "94797ff8-a780-4f2b-8e0b-0242f8ffee19",
  "path": "/content/drive/MyDrive/ego4d_data/clip540s/94797ff8-a780-4f2b-8e0b-0242f8ffee19.mp4",
  "num_frames": 131,
  "fps": 30.0,
  "width": 960,
  "height": 540
}

2.1. UID extraction rule

Copilot must define a deterministic rule for uid:

If filenames are <uid>.mp4, then uid = basename_without_extension.

If filenames differ, parse & document the rule in comments.

2.2. Manifest builder script

Implement a Python script (e.g. build_videomae_manifest.py) that:

Reads sta_clips_ok.txt.

For each mp4 path:

Extract uid by the defined rule.

Use ffprobe or OpenCV to get num_frames, fps, width, height. 

Construct the Colab path by replacing the local root with the future mount point:

local address/.../<uid>.mp4

colab: /content/drive/MyDrive/ego4d_data/clip540s/<uid>.mp4

Append a JSON line with the fields above to ego4d_sta_clips.jsonl.

Log summary to local_logs/manifest_summary.txt:

clips processed, min/max/mean frames & resolution.


3. Precompute Frame Tensors

Purpose: avoid decoding mp4 on Colab; instead, load ready (C, T, H, W) tensors.

Target directory:

tensors/...

File per clip, name:

<uid>.pt

3.1. Decide tensor format

Copilot must implement and document:

T (frames per clip), e.g. 16 or 32.

Spatial size (H, W), usually 224 x 224.

Tensor shape: (C, T, H, W) where C = 3.

Value range: [0, 1] (float32). No normalization beyond scaling; VideoMAE processor will do its own normalization later.

3.2. Tensor builder script

Implement build_videomae_tensors.py that:

Takes arguments (or constants at top):

--manifest  as .../ego4d_sta_clips.jsonl

--out_root as .../tensors

--target_frames (e.g. 16)

--target_size (e.g. 224)

For each record in the manifest:

Compute the local video path by reversing the Colab path back to local path

If <out_root>/<uid>.pt already exists → skip (resume-friendly).

Decode all frames or enough frames from the video using ffmpeg / cv2.VideoCapture.

uniformly sample a start index, then take T consecutive frames (a short local window with small stride)

For each sampled frame:

Resize to (target_size, target_size).

Convert to RGB, float32, scale by 1/255.

Stack frames to shape (T, H, W, C) then permute to (C, T, H, W).

Save tensor with torch.save.

Track:

number of tensors created

total disk size of tensors/ folder

number of skipped (already existing) tensors

Save a log file:

local_logs/tensors_summary.txt.

3.3. Update manifest with tensor paths

If desired, Copilot can add to each JSON record a field:

"tensor_path": "/content/drive/MyDrive/videomae_preproc/tensors/94797ff8-a780-4f2b-8e0b-0242f8ffee19.pt"


