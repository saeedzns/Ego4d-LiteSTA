# VideoMAE Local Preprocessing Pipeline

**Purpose**: Preprocess Ego4D clips locally before uploading to Colab for VideoMAE pretraining, avoiding expensive MP4 decoding on cloud GPUs.

---

## 📁 Folder Structure

```
videomae_local_part/
├── scan_clips.py               # Task 1a: Find all MP4 files
├── validate_clips.py           # Task 1b: Validate clip integrity
├── build_videomae_manifest.py  # Task 2: Build JSONL manifest
├── build_videomae_tensors.py   # Task 3: Precompute frame tensors
├── check_clips_manifest.py     # Utility: Cross-check clips vs manifest
├── todo.md                     # Original task specification
├── README.md                   # This documentation
│
├── local_lists/                # Clip discovery outputs
│   ├── sta_clips_all.txt       # All 2323 MP4 paths found
│   └── sta_clips_ok.txt        # Validated clips (2323 OK)
│
├── logs/                       # Processing logs
│   ├── bad_clips.txt           # Clips that failed validation (empty)
│   ├── manifest_summary.txt    # Manifest build statistics
│   ├── tensors_summary.txt     # Tensor build statistics
│   └── validation_report.txt   # Validation run report
│
├── manifests/                  # JSONL manifests for Colab
│   ├── ego4d_sta_clips.jsonl   # Full manifest (2323 records)
│   └── ego4d_sta_clips_test.jsonl  # Test subset (5 records)
│
├── tensors/                    # Precomputed frame tensors
│   └── <uid>.pt                # 2323 files, 20.85 GB total
│
└── tensors_test/               # Test tensors (5 files)
    └── <uid>.pt
```

---

## 📊 Dataset Summary

| Metric | Value |
|--------|-------|
| **Total clips** | 2,323 |
| **Source location** | `I:\My Drive\ego4d_data\clip540s` |
| **Total clip size** | 95.76 GB |
| **Total tensor size** | 20.85 GB |
| **Tensor size (avg)** | 9.4 MB per tensor |

### Video Statistics

| Statistic | Value |
|-----------|-------|
| **Frame count** | min=346, max=9480, mean=8045 |
| **FPS** | 30.0 (all clips) |
| **Resolutions** | 1440×1080 (1386), 1920×1440 (609), 1920×1080 (196), 2560×1920 (130), 2560×1440 (2) |

---

## 🔧 Scripts

### 1. `scan_clips.py` - Clip Discovery

**Purpose**: Recursively scan for MP4 files in the clips folder.

**Usage**:
```bash
python scan_clips.py
```

**Inputs**: Hard-coded path `I:\My Drive\ego4d_data\clip540s`

**Outputs**:
- `local_lists/sta_clips_all.txt` - One absolute path per line

**Key Features**:
- Recursive directory traversal
- Case-insensitive `.mp4` matching
- Summary printed to console

---

### 2. `validate_clips.py` - Integrity Check

**Purpose**: Validate clips are readable via OpenCV (optional random sampling).

**Usage**:
```bash
python validate_clips.py
```

**Inputs**: `local_lists/sta_clips_all.txt`

**Outputs**:
- `local_lists/sta_clips_ok.txt` - Validated clip paths
- `logs/bad_clips.txt` - Failed clips (if any)
- `logs/validation_report.txt` - Run summary

**Key Features**:
- Opens each clip with `cv2.VideoCapture`
- Reads first frame to verify integrity
- Configurable sample size for quick checks
- Reports bad clips with error messages

---

### 3. `build_videomae_manifest.py` - Manifest Builder

**Purpose**: Create JSONL manifest with clip metadata for Colab to read.

**Usage**:
```bash
python build_videomae_manifest.py
```

**Inputs**: `local_lists/sta_clips_ok.txt`

**Outputs**:
- `manifests/ego4d_sta_clips.jsonl` - Full manifest
- `logs/manifest_summary.txt` - Statistics

**JSONL Record Format**:
```json
{
  "uid": "94797ff8-a780-4f2b-8e0b-0242f8ffee19",
  "path": "/content/drive/MyDrive/ego4d_data/clip540s/94797ff8-a780-4f2b-8e0b-0242f8ffee19.mp4",
  "num_frames": 8145,
  "fps": 30.0,
  "width": 1440,
  "height": 1080,
  "tensor_path": "/content/drive/MyDrive/ego4d_data/videomae_tensors/94797ff8-a780-4f2b-8e0b-0242f8ffee19.pt"
}
```

**Key Features**:
- Extracts UID from filename (basename without extension)
- Uses OpenCV to read video metadata (fps, frame count, resolution)
- Maps local paths to Colab mount paths
- Includes `tensor_path` for precomputed tensors
- tqdm progress bar
- Resume-friendly (appends to existing)

**Path Mapping**:
| Location | Path |
|----------|------|
| Local clips | `I:\My Drive\ego4d_data\clip540s\<uid>.mp4` |
| Colab clips | `/content/drive/MyDrive/ego4d_data/clip540s/<uid>.mp4` |
| Colab tensors | `/content/drive/MyDrive/ego4d_data/videomae_tensors/<uid>.pt` |

---

### 4. `build_videomae_tensors.py` - Tensor Builder

**Purpose**: Precompute frame tensors to avoid MP4 decoding on Colab.

**Usage**:
```bash
python build_videomae_tensors.py
```

**Inputs**: `manifests/ego4d_sta_clips.jsonl`

**Outputs**:
- `tensors/<uid>.pt` - PyTorch tensor files
- `logs/tensors_summary.txt` - Build statistics

**Tensor Specification**:
| Property | Value |
|----------|-------|
| **Shape** | `(C=3, T=16, H=224, W=224)` |
| **Dtype** | `float32` |
| **Value range** | `[0.0, 1.0]` |
| **Storage** | `torch.save()` format |
| **Size per tensor** | ~9.4 MB |

**Key Features**:
- Uniformly samples start frame, takes 16 consecutive frames
- Resizes to 224×224 (bilinear interpolation)
- Converts BGR→RGB and scales to [0,1]
- Resume-friendly (skips existing `.pt` files)
- tqdm progress bar with ETA
- Detailed error logging

**Frame Sampling Strategy**:
1. Random start index in range `[0, num_frames - 16]`
2. Extract 16 consecutive frames (small temporal stride)
3. Resize each frame to 224×224
4. Stack into `(3, 16, 224, 224)` tensor

---

### 5. `check_clips_manifest.py` - Utility

**Purpose**: Cross-check that clips folder matches manifest entries.

**Usage**:
```bash
python check_clips_manifest.py
```

**Features**:
- Counts MP4 files in source directory
- Compares against manifest line count
- Reports total disk usage

---

## 🔄 Pipeline Workflow

Execute scripts in order:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: SCAN                                                         │
│   python scan_clips.py                                               │
│   → Finds 2323 MP4 files                                             │
│   → Writes local_lists/sta_clips_all.txt                             │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: VALIDATE                                                     │
│   python validate_clips.py                                           │
│   → Validates clip integrity                                         │
│   → Writes local_lists/sta_clips_ok.txt                              │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: BUILD MANIFEST                                               │
│   python build_videomae_manifest.py                                  │
│   → Extracts video metadata                                          │
│   → Writes manifests/ego4d_sta_clips.jsonl                           │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: BUILD TENSORS                                                │
│   python build_videomae_tensors.py                                   │
│   → Precomputes (3,16,224,224) tensors                               │
│   → Writes tensors/<uid>.pt                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📤 Colab Integration

### Upload to Google Drive

After building tensors locally, sync them to Google Drive:

1. **Tensors folder** → `My Drive/ego4d_data/videomae_tensors/`
2. **Manifest file** → `My Drive/ego4d_data/ego4d_sta_clips.jsonl`

### Load in Colab

```python
import torch
import json

# Load manifest
manifest_path = "/content/drive/MyDrive/ego4d_data/ego4d_sta_clips.jsonl"
with open(manifest_path) as f:
    records = [json.loads(line) for line in f]

# Load a tensor
record = records[0]
tensor = torch.load(record["tensor_path"])  # shape: (3, 16, 224, 224)
print(tensor.shape, tensor.dtype, tensor.min(), tensor.max())
# (3, 16, 224, 224) float32 0.0 1.0
```

### VideoMAE Processing

The tensors are **not normalized** beyond [0,1] scaling. VideoMAE processor applies its own normalization:

```python
from transformers import VideoMAEImageProcessor

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

# Tensor is (C, T, H, W), processor expects list of frames (T, H, W, C) or (T, C, H, W)
# Permute and pass to processor
frames = tensor.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
inputs = processor(list(frames), return_tensors="pt")
```

---

## 💾 Disk Space Requirements

| Item | Size |
|------|------|
| Source clips | 95.76 GB |
| Precomputed tensors | 20.85 GB |
| Manifest files | ~1 MB |
| **Total for Colab upload** | **~21 GB** (tensors + manifest only) |

---

## ⚙️ Dependencies

```bash
pip install opencv-python torch tqdm
```

---

## 📝 Notes

1. **Resume-friendly**: All scripts skip existing files on re-run.
2. **Idempotent**: Safe to run multiple times.
3. **No normalization**: Tensors are [0,1] scaled only; VideoMAE handles normalization.
4. **Frame sampling**: Random start + 16 consecutive frames (not uniform stride).
5. **Path convention**: UIDs are extracted from MP4 filenames (e.g., `<uid>.mp4`).

---

## 📅 Build History

| Date | Action | Result |
|------|--------|--------|
| 2025-12-04 | Initial scan | 2323 clips found |
| 2025-12-04 | Validation | 2323 OK, 0 bad |
| 2025-12-04 | Manifest build | 2323 records |
| 2025-12-04 | Tensor build | 2323 tensors (20.85 GB) |

---

*Generated for Ego4D-LiteSTA thesis project*
