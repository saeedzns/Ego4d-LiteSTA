## Jupytext-style script; open in VS Code/Jupyter as a notebook or run as a script

# %% [markdown]
# Ego4D Frames — 0→100 (Colab or local Jupyter)
#
# This Python file mirrors the notebook workflow using cell markers.
# You can run it as a script, or open as a notebook in VS Code/Jupyter.

# %%
# Imports and environment detection
import os, csv, ast, glob, subprocess
from pathlib import Path

try:
    from google.colab import drive  # type: ignore
    ON_COLAB = True
except Exception:  # noqa: F401
    ON_COLAB = False

REPO_ROOT = Path('.')
LISTING_CSV = REPO_ROOT / 'file_listings' / 'ego4d_file_listing.csv'
SAMPLES_ROOT = REPO_ROOT / 'ego4d_samples'
V2_ANN = SAMPLES_ROOT / 'v2' / 'annotations'
MAIN_JSON_CSV = SAMPLES_ROOT / 'ego4d_json.csv'
print('ON_COLAB =', ON_COLAB)
print('Listing exists:', LISTING_CSV.exists())
print('Samples present:', V2_ANN.exists())

# %%
# Mount Drive (only on Colab)
if ON_COLAB:
    drive.mount('/content/drive', force_remount=False)
else:
    print('Not on Colab; skipping Drive mount.')

# %%
# Infer dataset roots from the file listing
DATA_V2 = '/content/drive/MyDrive/ego4d_data/v2'
VIDEOS_DIR = DATA_V2 + '/videos'   # fallback; override if listing shows a different folder
FRAMES_DIR = DATA_V2 + '/frames'   # target frames directory

if LISTING_CSV.exists():
    with open(LISTING_CSV, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        paths = [row['Path'] for row in r if 'Path' in row]
    # Prefer explicit v2 root if present
    roots = [p for p in paths if p.rstrip('/').endswith('/ego4d_data/v2')]
    if roots:
        DATA_V2 = roots[0].rstrip('/')
        FRAMES_DIR = DATA_V2 + '/frames'
    # Try to infer a videos folder if .mp4 listed
    vid_candidates = [str(Path(p).parent) for p in paths if p.lower().endswith('.mp4')]
    if vid_candidates:
        from collections import Counter
        VIDEOS_DIR = Counter(vid_candidates).most_common(1)[0][0]

os.makedirs(FRAMES_DIR, exist_ok=True)
print('DATA_V2   =', DATA_V2)
print('VIDEOS_DIR=', VIDEOS_DIR)
print('FRAMES_DIR=', FRAMES_DIR)

# %%
# Helper: parse JSON-like cells safely (for samples)
def parse_jsonish(cell: str):
    t = cell.strip()
    if (t[:1] in ("'", '"')) and (t[-1:] in ("'", '"')):
        t = t[1:-1]
    try:
        return ast.literal_eval(t)
    except Exception:
        return None

# Collect video_uids referenced in v2 sample CSVs (for a small working set)
video_uids = set()
if V2_ANN.exists():
    for p in sorted(V2_ANN.glob('*.csv')):
        with p.open('r', encoding='utf-8') as f:
            header = f.readline().strip()
            r = csv.reader(f)
            for i, row in enumerate(r):
                if not row or len(row) != 1:
                    continue
                d = parse_jsonish(row[0])
                if isinstance(d, dict):
                    if 'video_uid' in d:
                        video_uids.add(d['video_uid'])
                    if 'clips' in d and isinstance(d['clips'], list) and d['clips']:
                        vu = d['clips'][0].get('video_uid')
                        if vu:
                            video_uids.add(vu)
                if i > 400:  # sampling limit per file
                    break
print('Found video_uids:', len(video_uids))
print('Sample:', list(sorted(video_uids))[:5])

# %%
# Verify videos exist under VIDEOS_DIR
existing, missing = [], []
for vu in sorted(list(video_uids))[:50]:  # check a subset for speed
    found = False
    for ext in ('.mp4', '.mov', '.mkv'):
        if os.path.exists(os.path.join(VIDEOS_DIR, vu + ext)):
            found = True
            break
    (existing if found else missing).append(vu)
print('Existing (subset):', len(existing))
print('Missing  (subset):', len(missing))
if missing:
    print('Example missing uid:', missing[0])
    print('If videos are elsewhere, set VIDEOS_DIR accordingly above.')

# %%
# Helpers: frame path builder, preview util, video source resolution
from PIL import Image
import matplotlib.pyplot as plt

def frame_path(video_uid: str, frame_number: int, frames_dir: str = FRAMES_DIR, pad: int = 7, ext: str = 'jpg'):
    return os.path.join(frames_dir, video_uid, str(frame_number).zfill(pad) + '.' + ext)

def show_image(path: str, title: str = ''):
    img = Image.open(path).convert('RGB')
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def find_video_src(video_uid: str):
    for ext in ('.mp4', '.mov', '.mkv'):
        cand = os.path.join(VIDEOS_DIR, video_uid + ext)
        if os.path.exists(cand):
            return cand
    return None

# %%
# Select a sample CSV to drive extraction (keyframes and/or windows)
SAMPLE_CSV = str(V2_ANN / 'fho_sta_val_json.csv')  # e.g., 'fho_hands_val_json.csv', 'fho_oscc-pnr_val_json.csv'
RUN_KEYFRAMES = False   # set True to extract specific frames (images)
RUN_WINDOWS   = False   # set True to extract frame ranges (clips)
MAX_ITEMS     = 50      # limit records to process
print('Using SAMPLE_CSV =', SAMPLE_CSV)

# %%
# Parse SAMPLE_CSV and collect per-video keyframes and windows
from collections import defaultdict

def collect_from_record(d: dict):
    keyframes = set()
    windows = []
    # known start/end frame pairs
    pairs = [
        ('video_start_frame', 'video_end_frame'),
        ('clip_start_frame', 'clip_end_frame'),
        ('action_start_frame', 'action_end_frame'),
        ('interval_start_frame', 'interval_end_frame'),
        ('clip_parent_start_frame', 'clip_parent_end_frame'),
    ]
    # recursive scan
    def rec(x):
        if isinstance(x, dict):
            # keyframes by explicit names
            for k in list(x.keys()):
                v = x[k]
                kl = k.lower()
                if isinstance(v, int) and (
                    kl == 'frame' or kl.endswith('_frame') or kl in (
                        'pnr_frame', 'clip_pnr_frame', 'video_frame_number', 'frame_number', 'clip_frame_number'
                    )
                ):
                    if kl not in ('frame_height', 'frame_width'):
                        keyframes.add(int(v))
            # collect windows
            for a, b in pairs:
                if a in x and b in x and isinstance(x[a], int) and isinstance(x[b], int):
                    s, e = int(x[a]), int(x[b])
                    if e >= s:
                        windows.append((s, e))
            # recurse
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)
    rec(d)
    return keyframes, windows

per_video_frames = defaultdict(set)
per_video_windows = defaultdict(list)
processed = 0
with open(SAMPLE_CSV, 'r', encoding='utf-8') as f:
    header = f.readline().strip()
    r = csv.reader(f)
    for row in r:
        if not row or len(row) != 1:
            continue
        d = parse_jsonish(row[0])
        if not isinstance(d, dict):
            continue
        vu = d.get('video_uid')
        if not vu and 'clips' in d and isinstance(d['clips'], list) and d['clips']:
            vu = d['clips'][0].get('video_uid')
        if not vu:
            continue
        kf, wins = collect_from_record(d)
        per_video_frames[vu].update(kf)
        per_video_windows[vu].extend(wins)
        processed += 1
        if processed >= MAX_ITEMS:
            break

print('Records processed =', processed)
print('Videos with keyframes:', sum(1 for v in per_video_frames if per_video_frames[v]))
print('Videos with windows  :', sum(1 for v in per_video_windows if per_video_windows[v]))

# %%
# Extract specific frames (images) using ffmpeg select='eq(n,FRAME)'
def extract_single_frame(video_uid: str, frame_idx: int):
    src = find_video_src(video_uid)
    if not src:
        return False, 'video not found'
    out_dir = os.path.join(FRAMES_DIR, video_uid)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'f{frame_idx:07d}.jpg')
    if os.path.exists(out_path):
        return True, 'exists'
    expr = f"select='eq(n,{frame_idx})'"
    cmd = ['ffmpeg', '-y', '-i', src, '-vf', expr, '-vsync', 'vfr', '-vframes', '1', out_path]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ok = os.path.exists(out_path)
    return ok, r.returncode

if RUN_KEYFRAMES:
    for vu, frames in per_video_frames.items():
        for fi in sorted(list(frames))[:200]:
            ok, msg = extract_single_frame(vu, fi)
        print('keyframes extracted for', vu)
else:
    print('RUN_KEYFRAMES=False — skipping keyframe extraction')

# %%
# Extract frame windows (ranges) using ffmpeg select='between(n,START,END)'
def extract_window(video_uid: str, start: int, end: int):
    src = find_video_src(video_uid)
    if not src:
        return False, 'video not found'
    out_dir = os.path.join(FRAMES_DIR, video_uid, f'win_{start}_{end}')
    os.makedirs(out_dir, exist_ok=True)
    expr = f"select='between(n,{start},{end})'"
    out_pattern = os.path.join(out_dir, '%07d.jpg')
    cmd = ['ffmpeg', '-y', '-i', src, '-vf', expr, '-vsync', 'vfr', out_pattern]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode == 0, r.returncode

if RUN_WINDOWS:
    for vu, wins in per_video_windows.items():
        for (s, e) in wins[:10]:
            ok, msg = extract_window(vu, s, e)
        print('windows extracted for', vu)
else:
    print('RUN_WINDOWS=False — skipping window extraction')

# %%
# Preview a frame if available
previewed = False
for vu in sorted(video_uids):
    pattern = os.path.join(FRAMES_DIR, vu, '*.jpg')
    imgs = sorted(glob.glob(pattern))
    if imgs:
        show_image(imgs[min(10, len(imgs)-1)], title=f'{vu}')
        previewed = True
        break
if not previewed:
    print('No frames found under FRAMES_DIR. Extract first, then re-run this cell.')

# %%
# Optional: compute last frame index using ego4d_samples/ego4d_json.csv
def get_num_frames_from_main(video_uid: str):
    if not MAIN_JSON_CSV.exists():
        return None
    with MAIN_JSON_CSV.open('r', encoding='utf-8') as f:
        header = f.readline()
        for row in csv.reader(f):
            d = parse_jsonish(row[0])
            if isinstance(d, dict) and d.get('video_uid') == video_uid:
                vm = d.get('video_metadata') or {}
                return vm.get('num_frames')
    return None

for vu in sorted(video_uids)[:5]:
    nf = get_num_frames_from_main(vu)
    if nf:
        p = frame_path(vu, int(nf)-1)
        print('Last-frame path (may or may not exist yet):', p)
        break

# %%
# Report: where "frame" fields appear in v2 sample CSVs
def frame_keys_report(max_rows=60):
    report = {}
    if not V2_ANN.exists():
        return report
    for p in sorted(V2_ANN.glob('*.csv')):
        keys = set()
        with p.open('r', encoding='utf-8') as f:
            header = f.readline().strip()
            for i, row in enumerate(csv.reader(f)):
                if not row or len(row) != 1:
                    continue
                d = parse_jsonish(row[0])
                if isinstance(d, dict):
                    for k, v in d.items():
                        if 'frame' in k.lower():
                            keys.add(k)
                        if isinstance(v, dict):
                            for kk in v.keys():
                                if 'frame' in kk.lower():
                                    keys.add(f'{k}.{kk}')
                        elif isinstance(v, list) and v and isinstance(v[0], dict):
                            for kk in v[0].keys():
                                if 'frame' in kk.lower():
                                    keys.add(f'{k}[].{kk}')
                if i > max_rows:
                    break
        if keys:
            report[str(p)] = sorted(keys)
    return report

rep = frame_keys_report()
for path, keys in rep.items():
    print('FILE:', path)
    for k in keys:
        print('  -', k)

# %% [markdown]
# Done. Frames are configured to be written under `/content/drive/MyDrive/ego4d_data/v2/frames`.
# Adjust `VIDEOS_DIR` above if your videos are stored elsewhere.

# %% [markdown]
# Hints: Where are frames and images?
#
# - Output frames root (this script writes here):
#   - `FRAMES_DIR = {DATA_V2}/frames` → e.g. `/content/drive/MyDrive/ego4d_data/v2/frames`
#   - Single keyframes go to: `FRAMES_DIR/<video_uid>/f0000123.jpg`
#   - Window/range extractions go to: `FRAMES_DIR/<video_uid>/win_<start>_<end>/%07d.jpg`
#
# - Original videos (inputs):
#   - Resolved from your listing `file_listings/ego4d_file_listing.csv` into `VIDEOS_DIR`.
#   - Typical: `/content/drive/MyDrive/ego4d_data/v2/videos/<video_uid>.mp4` (but we infer and print `VIDEOS_DIR`).
#
# - How to decide WHICH frames to extract:
#   - See `ego4d_samples/_auto_fields.md` for a per-file field index.
#   - Common frame sources (by task samples in `ego4d_samples/v2/annotations/`):
#     - STA (fho_sta_*): `frame`, `clip_frame`, `action_*_frame`, `interval_*_frame`.
#     - OSCC/PNR (fho_oscc-pnr_*): `clip_pnr_frame`, `clip_*_frame`, `parent_*_frame`.
#     - Hands/SCOD (fho_hands_*, fho_scod_*): `pre_45`, `pre_30`, `pre_15`, `pnr_frame`, `post_frame`.
#     - NLQ/VQ/Moments/AV: `video_*_frame`, `clip_*_frame` windows.
#   - Use these fields to set `RUN_KEYFRAMES` (discrete frames) or `RUN_WINDOWS` (ranges) above.
#
# - Quick sanity checks:
#   - `print('VIDEOS_DIR =', VIDEOS_DIR)` → where inputs are expected.
#   - `print('FRAMES_DIR =', FRAMES_DIR)` → where outputs land.
#   - After extraction, preview with the built-in `show_image(...)` helper.

print("\n==== HINTS ====")
print("Frames output root:", FRAMES_DIR)
print("Videos input root:", VIDEOS_DIR)
print("Examples:")
print(" - Keyframe path → ", os.path.join(FRAMES_DIR, '<video_uid>', 'f0000123.jpg'))
print(" - Window path   → ", os.path.join(FRAMES_DIR, '<video_uid>', 'win_<start>_<end>', '%07d.jpg'))
print("See ego4d_samples/_auto_fields.md for which *_frame fields to use per sample CSV.")
print("We inferred DATA_V2 and VIDEOS_DIR from file_listings/ego4d_file_listing.csv.")
