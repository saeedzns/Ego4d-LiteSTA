# Colab Download & Setup Playbook for Ego4D-LiteSTA

This document distills every download- or install-related action in `notebooks/Ego4D_STA_Colab_Starter.ipynb`, clarifying **what** is fetched and **why** it is needed when preparing a Colab Free session. Frame extraction routines are intentionally omitted because they are executed on the local Windows workstation.

---

## 1. Workspace & Git Synchronization

### 1.1 Mount Google Drive
- **Command:** `drive.mount('/content/drive')`
- **Purpose:** Expose personal Drive storage so that checkpoints, logs, and the Git repository persist across Colab sessions.

### 1.2 Configure Git Identity
- **Commands:**
  - `!git config --global user.name "saeedzns"`
  - `!git config --global user.email "zns1992@gmail.com"`
- **Purpose:** Ensure commits from Colab carry the correct author metadata.

### 1.3 Harden Git over SSH
- **Commands:** Set `GIT_SSH_COMMAND` keepalive, configure pack compression/threads.
- **Purpose:** Prevent long-running pulls/pushes from timing out on Colab’s network.

### 1.4 Load Deploy Key from Drive
- **Script:** Bash cell that starts `ssh-agent`, adds the Drive-stored private key (`colab_egolite`), writes `~/.ssh/config`, and preloads GitHub host keys.
- **Purpose:** Allow passwordless Git operations (clone, pull, push) from Colab using a deploy key with repo write access.

### 1.5 Repo Sync Helpers
- **Scripts:** `pull.sh`, `push.sh`, "fast push" path, and raw git commands (`git status`, `git add -A`, `git commit`, `git push`).
- **Purpose:** Reinforce push/pull workflows in Colab so that code edits and generated assets are versioned back to GitHub.

---

## 2. Core Dependencies from Remote Sources

### 2.1 AWS CLI Installation
- **Commands:** Download `awscliv2.zip`, unzip, run `sudo ./aws/install`, configure credentials from environment variables.
- **Purpose:** Enable direct access to Ego4D’s S3 buckets (annotations, checkpoints, video assets) via authenticated AWS CLI calls.

### 2.2 Ego4D Python Package (CLI)
- **Command:** `!pip install ego4d`
- **Purpose:** Provide the `ego4d` command-line tool for dataset discovery and downloading; the package also includes schema helpers for STA tasks.

### 2.3 List Available Ego4D Datasets
- **Command:** `!ego4d --list-datasets`
- **Purpose:** Quick sanity check that the CLI is operational and credentials are valid.

---

## 3. Annotation & Model Asset Downloads

> Frame-extraction commands in the notebook are excluded here because all frame generation is handled on the local machine.

### 3.1 540p Super-Set Annotations (Optional Sandbox)
- **Command:** `!ego4d --output_directory="/content/540" --datasets annotations_540ss -y`
- **Purpose:** Grab a lightweight annotations bundle for rapid prototyping or schema inspection without committing to full-resolution data.

### 3.2 Full STA Bundle (Recommended)
- **Command:**
  ```bash
  !ego4d --output_directory="/content/drive/MyDrive/ego4d_data/" \
        --datasets annotations sta_models omnivore_video_swinl_fp16 \
        --benchmarks FHO \
        --version v2 -y
  ```
- **Purpose:**
  - `annotations`: Required STA train/val/test JSONs.
  - `sta_models`: Provides official detector outputs used in the original benchmark.
  - `omnivore_video_swinl_fp16`: Supplies pretrained clip backbones for optional comparisons.
  - `benchmarks FHO`: Scoped download to the First-Person Hand-Object (FHO) benchmark, which houses STA.
  - `version v2`: Locks to the latest STA annotation schema.

### 3.3 Additional 540p Annotation Sync (Drive Mirror)
- **Command:**
  ```bash
  !ego4d --output_directory="/content/drive/MyDrive/ego4d_data" \
        --datasets annotations_540ss \
        --benchmarks FHO \
        -y
  ```
- **Purpose:** Mirror the 540p subset directly into Drive for persistent access alongside the main v2 dataset.

### 3.4 Clip Downloads (Full & 540p)
- **Command (full-resolution clips):**
  ```bash
  !ego4d --output_directory="/content" \
         --datasets clips \
         --version v2 \
         --video_uid_file /content/drive/MyDrive/ego4d_data/sta_clip_uids_part1.txt \
         -y
  ```
  - Reads a plain-text list of `video_uid` values and pulls only those clip segments, ideal for Stage B prototype runs that need short spans instead of entire videos.
- **Command (540p clips):**
  ```bash
  !ego4d --output_directory="/content/drive/MyDrive/ego4d_data" \
         --datasets clips_540 \
         --version v2 \
         --video_uid_file /content/drive/MyDrive/ego4d_data/sta_clip_uids_part1.txt \
         -y
  ```
  - Mirrors the same `video_uid` list but downloads the bandwidth-friendly 540p versions (`clips_540` dataset) so experiments stay lightweight on Colab Free or local storage.

### 3.5 Annotation Downscaling Utility
- **Reference notebook:** [`facebookresearch/Ego4d/notebooks/transform_annotations.ipynb`](https://github.com/facebookresearch/Ego4d/blob/main/notebooks/transform_annotations.ipynb)
- **Purpose:** Provides a reusable recipe for scaling annotation coordinates when resizing frames (e.g., mapping full-resolution STA boxes to the 540p clips). We adapt this notebook locally to keep YOLO labels and manifest boxes aligned after down-sampling.

---

## 4. Forecasting Repository (Official Evaluators)

### 4.1 Clone Ego4D Forecasting Repo
- **Command:** `git clone https://github.com/EGO4D/forecasting`
- **Purpose:** Fetch the official evaluation scripts (especially `SHORT_TERM_ANTICIPATION.md`) and metric implementations needed to score STA predictions.

---

## 5. Repository Asset Generation (Stored in Drive)

Although not external downloads, the notebook creates helper artifacts using data just downloaded. They’re documented here because they depend on the new assets and become part of the pipeline.

### 5.1 File Inventory CSV
- **Script:** Walk `/content/drive/MyDrive/ego4d_data`, record paths/sizes, and save to `Ego4d-LiteSTA/file_listings/ego4d_file_listing.csv`.
- **Purpose:** Provide a searchable index of all downloaded files for debugging and data audits.

### 5.2 Sample Extracts into `ego4d_samples/`
- **Script:** For every JSON/CSV under `ego4d_data`, capture two representative entries and save the snippets as CSV files in the repo.
- **Purpose:** Offer lightweight previews of large annotation/model files without loading them entirely (useful for schema exploration and QA).

### 5.3 Derived Metadata Sets (Annotations Only)
- **Scripts:** Combine STA train/val annotations into NDJSON, create per-video frame lists, and extract unique video UIDs.
- **Purpose:** Support downstream manifest builders or targeted downloads without re-parsing the full annotation files. (Note: actual video/frame extraction is handled offline and intentionally left out.)

---

## 6. GPU Context Checks (Sanity)

### 6.1 GPU Availability Probe
- **Command:** `torch.cuda.is_available()` plus device name print.
- **Purpose:** Confirm whether the Colab VM exposes a GPU, influencing decisions about running heavier STA inference or pruning experiments remotely.

---

## 7. Why Certain Steps Are Omitted Here

Sections in the notebook titled **“Frame extraction”** and related ffmpeg workflows download full-scale Ego4D videos and generate per-frame JPEGs. Because this process has been migrated to the local Windows environment, those commands are excluded from this summary to avoid duplicate work and wasted Drive quota. Locally we download short clip segments via the Ego4D CLI (`--video_uids` scoped pulls) and only extract the exact frame IDs listed in `tmp_frame_lists/` so that we avoid chewing through storage/bandwidth on frames the Stage B manifest never references. The local pipeline reads `fho_sta_train.json` and `fho_sta_val.json`, groups annotations by `video_uid`, selects frame indices, and, per frame, picks the object with minimum `time_to_contact` as the positive box (mapping noun IDs through the taxonomy when multi-class detection is enabled). We write the sorted unique frames to `tmp_frame_lists/<video_uid>_frames.txt`, run ffmpeg to extract those frames, and emit YOLO `.txt` labels with normalized boxes alongside each saved image.

---

## 8. Quick Reference Table

| Step | Command/Tool | Downloaded Artifact | Reason |
| --- | --- | --- | --- |
| Mount Drive | `drive.mount` | N/A | Persist data & code edits across sessions |
| Git SSH setup | `ssh-agent`, `ssh-add` | SSH config | Allow secure clone/push |
| AWS CLI | `curl`, `aws/install` | `aws` binary | Authenticate to Ego4D S3 |
| Ego4D CLI | `pip install ego4d` | `ego4d` package | Dataset download utility |
| 540ss annotations | `ego4d --datasets annotations_540ss` | 540p STA JSON | Lightweight testing subset |
| Full STA bundle | `ego4d --datasets annotations sta_models omnivore_video_swinl_fp16` | STA annotations + models | Core training/eval data |
| Forecasting repo | `git clone .../forecasting` | Evaluation scripts | Official STA metrics |
| File inventory | `get_directory_structure` script | CSV index | Track downloaded assets |
| Samples export | Custom pandas script | CSV snippets | Rapid schema inspection |

---

## 9. Recommended Colab Session Flow

1. **Mount Drive** and set `WORKDIR`.
2. **Initialize SSH** agent and pull the latest repo state.
3. **Install AWS CLI** and configure credentials (env vars must be set beforehand).
4. **Install Ego4D CLI** and verify access via `--list-datasets`.
5. **Download required datasets** (annotations, models) into Drive.
6. **Clone the forecasting repo** to obtain metric scripts.
7. **Run helper scripts** to catalog files and generate quick-look samples.
8. Proceed to Colab-based analytics or model validation (training and frame extraction remain on local machines).

---

By following this playbook, each Colab session consistently fetches the exact assets needed for Ego4D STA experiments while avoiding redundant downloads and staying aligned with the division of labor between Colab (lightweight analysis, metric tooling) and the local environment (frame extraction, heavy training).
