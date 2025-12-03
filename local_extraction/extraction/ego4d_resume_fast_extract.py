#!/usr/bin/env python3
# ==============================================
# Ego4D — Resume-safe fast frame extraction (clips or video_540)
# Subset control (priority):
#   1) SPECIFIC_IDS
#   2) USE_PART_FILES + PART_INDEXES or PART_FILES  (reads sta_clip_uids_part{idx}.txt)
#   3) UID_LIST_FILE + optional LIMIT_N (+ SHUFFLE_IDS)
#
# Speed paths:
#   - Decord (GPU/CPU) with adjustable batch size (best for sparse or controlled RAM)
#   - ffmpeg_select_fast (one-call select; good general fallback)
#   - ffmpeg_ranges (collapses contiguous runs; best for "from_lists")
#
# Safety:
#   - Skips already-finished clips; resumes partials (continues numbering)
#   - Writes a CSV log on Drive
#   - Chunked flush to Drive while extracting (caps /content usage)
# ==============================================
import os, sys, json, time, csv, shutil, subprocess, random, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm.auto import tqdm

# ---------------------- USER CONFIG ----------------------
# Root of your Ego4D data on Google Drive
# Default targets a Windows Google Drive path (H:\My Drive\ego4d_data),
# but can be overridden by EGO4D_ROOT env var or auto-detected.
def _detect_ego4d_root() -> str:
    env = os.environ.get("EGO4D_ROOT")
    if env:
        return env
    candidates = [
        # Windows Google Drive for desktop (use forward slashes for Python friendliness)
        "H:/My Drive/ego4d_data",
        "D:/My Drive/ego4d_data",
        # WSL mounts
        "/mnt/h/My Drive/ego4d_data",
        "/mnt/d/My Drive/ego4d_data",
        # Colab mounts
        "/content/drive/MyDrive/ego4d_data",
        "/drive/MyDrive/ego4d_data",
    ]
    for p in candidates:
        try:
            if os.path.isdir(p):
                return p
        except Exception:
            pass
    # Fallback to the requested Windows-style path
    return "H:/My Drive/ego4d_data"

EGO4D_ROOT   = _detect_ego4d_root()
VERSION      = "v2"

# Optionally write extracted frames to a different root (e.g., local D:) while
# keeping the same subdirectory layout (VERSION/extracted_frames/<uid>/...)
def _derive_output_root_from_input(in_root: str) -> str:
    # Explicit override takes precedence
    env = os.environ.get("EGO4D_OUTPUT_ROOT")
    if env:
        return env

    # Prefer storing under the repo's local_extraction folder for local runs
    try:
        here = Path(__file__).resolve().parent  # .../local_extraction
        repo_local = here  # store directly under local_extraction
        if os.path.isdir("/content"):
            pass  # Colab handled below
        else:
            if repo_local.is_dir():
                return str(repo_local)
    except Exception:
        pass

    norm = in_root.replace("\\", "/")
    # Map Windows drive letter to D: if possible (e.g., H:/My Drive/... -> D:/My Drive/...)
    m = re.match(r"^[A-Za-z]:/", norm)
    if m:
        out = "D:" + norm[2:]
        if os.path.isdir(out):
            return out

    # Map WSL /mnt/<letter>/... to /mnt/d/...
    if norm.startswith("/mnt/"):
        parts = norm.split("/")
        if len(parts) > 3 and len(parts[2]) == 1:
            parts[2] = "d"
            out = "/".join(parts)
            if os.path.isdir(out):
                return out

    # Fallback candidates commonly used for local mirrors
    for cand in ("D:/My Drive/ego4d_data", "/mnt/d/My Drive/ego4d_data"):
        if os.path.isdir(cand):
            return cand

    # Default: write alongside the input root
    return in_root

OUTPUT_ROOT = _derive_output_root_from_input(EGO4D_ROOT)

SOURCE_KIND  = "clip_540"            # "clip_540" or "video_540"
EXTRACTION_MODE = "from_lists"       # 'from_lists' | 'first_n' | 'all'
NUM_FRAMES_TO_EXTRACT = 10          # used if EXTRACTION_MODE == 'first_n'

# ---- Subset controls (choose ONE approach, priority is top→down) ----
SPECIFIC_IDS = []         # 1) exact list of clip_uids/video_uids to run (highest priority)

USE_PART_FILES = True                # 2) read from split part files below
PART_INDEXES   = list(range(36,51))                 # e.g., [1,2,3] -> loads part1, part2, part3
PART_PATH_TPL  = os.path.join(EGO4D_ROOT, VERSION, "sta_clip_uids_parts", "sta_clip_uids_part{idx}.txt")
PART_FILES: list[str] = []           # alternatively, explicit paths to part files

LIMIT_N       = 10                    # 3) if not using parts or SPECIFIC_IDS, process only first N (None = all)
SHUFFLE_IDS   = False                # shuffle before limiting

# Speed/quality knobs
SPEED_MODE         = "decord"        # "decord" | "ffmpeg_select_fast" | "ffmpeg_ranges"
USE_GPU_DECORD     = False           # try NVDEC when available
DECORD_BATCH       = 256              # lower = less RAM; typical 32–64; (was 256)
RESIZE_TO_540      = True
JPG_QUALITY        = 6               # 2..7 (7 = fastest/smallest files)
N_EXTRACT_WORKERS  = 4               # default; may be overridden dynamically below
FFMPEG_TIMEOUT_SECS= 1800
PRINT_PER_ITEM_INFO= True

# Environment detection
IS_COLAB = os.path.isdir("/content")

# Chunked flush to Drive during extraction
# - On Colab: periodically flush from /content to Drive to cap local usage
# - Local: write directly into final destination (no extra flush/move)
FLUSH_EVERY_IMAGES = 350 if IS_COLAB else 0

# ---------------------- PATHS ----------------------
if SOURCE_KIND == "clip_540":
    DATASET          = "clips"
    MEDIA_DIR        = os.path.join(EGO4D_ROOT, VERSION, "clips_540", "clips")   # <clip_uid>.mp4
    UID_LIST_FILE    = os.path.join(EGO4D_ROOT, "sta_clip_uids.txt")             # master list of clip_uids
    FRAME_LISTS_DIR  = os.path.join(EGO4D_ROOT, VERSION, "tmp_frame_lists", "clips")
    FRAME_FILE_TPL   = "{uid}_clip_frames.txt"
elif SOURCE_KIND == "video_540":
    DATASET          = "video_540ss"
    MEDIA_DIR        = os.path.join(EGO4D_ROOT, VERSION, "video_540ss")         # <video_uid>.mp4
    UID_LIST_FILE    = os.path.join(EGO4D_ROOT, "sta_uids.txt")
    FRAME_LISTS_DIR  = os.path.join(EGO4D_ROOT, VERSION, "tmp_frame_lists", "videos")
    FRAME_FILE_TPL   = "{uid}_frames.txt"
else:
    raise SystemExit("SOURCE_KIND must be 'clip_540' or 'video_540'")

FRAMES_DRIVE_ROOT = os.path.join(OUTPUT_ROOT, VERSION, "extracted_frames")
# Use a cross-platform scratch dir: prefer system temp on Windows, /content on Colab if present
_default_tmp = os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp"
if os.path.isdir("/content"):
    FRAMES_TMP_ROOT = "/content/extract_tmp"
else:
    FRAMES_TMP_ROOT = os.path.join(_default_tmp, "ego4d_extract_tmp")
LOG_CSV           = os.path.join(OUTPUT_ROOT, VERSION, "extraction_log.csv")

# ---------------------- PREP ----------------------
for p in (MEDIA_DIR, FRAME_LISTS_DIR, FRAMES_DRIVE_ROOT, FRAMES_TMP_ROOT, os.path.dirname(LOG_CSV)):
    os.makedirs(p, exist_ok=True)

def which(cmd: str) -> bool:
    import shutil as _sh; return _sh.which(cmd) is not None

def list_existing_jpgs(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")]) if os.path.isdir(folder) else []

def list_existing_indices(folder):
    idxs = set()
    if not os.path.isdir(folder):
        return idxs
    for f in os.listdir(folder):
        if not f.lower().endswith('.jpg'):
            continue
        stem, _ = os.path.splitext(f)
        try:
            idxs.add(int(stem))
        except Exception:
            continue
    return idxs

def ensure_log_header(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","uid","source_kind","mode","speed_mode","requested","new_written","total_now","note"])

def append_log(uid, requested, new_written, total_now, note):
    ensure_log_header(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.now().isoformat(timespec="seconds"), uid, SOURCE_KIND,
                    EXTRACTION_MODE, SPEED_MODE, requested, new_written, total_now, note])

def _read_uid_file(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_ids():
    # Priority 1: SPECIFIC_IDS
    if SPECIFIC_IDS:
        return list(dict.fromkeys([x for x in SPECIFIC_IDS if x]))

    # Priority 2: USE_PART_FILES
    if USE_PART_FILES:
        parts = PART_FILES if PART_FILES else [PART_PATH_TPL.format(idx=i) for i in PART_INDEXES]
        if not parts:
            raise SystemExit("USE_PART_FILES=True but neither PART_FILES nor PART_INDEXES provided.")
        all_ids = []
        for p in parts:
            if not os.path.exists(p):
                raise SystemExit(f"Part file missing: {p}")
            all_ids.extend(_read_uid_file(p))
        return list(dict.fromkeys(all_ids))  # de-dup, keep order

    # Priority 3: UID_LIST_FILE (+ optional limit)
    if not os.path.exists(UID_LIST_FILE):
        raise SystemExit(f"UID list not found: {UID_LIST_FILE}")
    ids = _read_uid_file(UID_LIST_FILE)
    if SHUFFLE_IDS: random.shuffle(ids)
    if LIMIT_N is not None: ids = ids[:int(LIMIT_N)]
    return ids

def load_frames_for(uid):
    p = os.path.join(FRAME_LISTS_DIR, FRAME_FILE_TPL.format(uid=uid))
    if not os.path.exists(p): return []
    with open(p) as f:
        return [int(x.strip()) for x in f if x.strip()]

def ffprobe_nb_frames(path):
    cmd = ["ffprobe","-v","error","-select_streams","v:0",
           "-show_entries","stream=nb_frames,r_frame_rate,width,height:format=duration",
           "-of","json", path]
    try:
        out = subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=60)
        data = json.loads(out.stdout or "{}")
        nb = None
        if data.get("streams"):
            s = data["streams"][0]
            nb = s.get("nb_frames")
            if nb is not None:
                try: nb = int(nb)
                except: nb = None
        if nb is None and "format" in data and data["format"].get("duration"):
            fps = None
            s = data["streams"][0]
            if s.get("r_frame_rate"):
                a,b = s["r_frame_rate"].split("/")
                fps = float(a)/max(float(b),1.0)
            if fps:
                nb = int(float(data["format"]["duration"])*fps)
        return nb
    except:
        return None

def to_ranges(indices):
    if not indices: return []
    xs = sorted(set(int(i) for i in indices))
    runs, s, p = [], xs[0], xs[0]
    for k in xs[1:]:
        if k == p+1: p = k
        else: runs.append((s, p)); s = p = k
    runs.append((s,p))
    return runs

# ---------------------- Decord path (GPU-aware, resume-safe, chunked flush) ----------------------
def ensure_decord():
    try:
        import decord  # noqa
        import imageio, PIL  # noqa
        return True
    except Exception:
        print("[setup] Installing decord, pillow, imageio ...")
        rc = os.system(f"{sys.executable} -m pip install -q decord pillow imageio")
        return rc == 0

def _resize_short_side(arr, short=540):
    from PIL import Image
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0: return arr
    if h <= w:
        new_h = short; new_w = int(round(w * (short/float(h))))
    else:
        new_w = short; new_h = int(round(h * (short/float(w))))
    return Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR)

def _choose_decord_ctx():
    try:
        from decord import gpu, cpu
        if USE_GPU_DECORD:
            has_gpu_runtime = os.path.isdir("/usr/local/cuda") or os.path.exists("/proc/driver/nvidia/version")
            if has_gpu_runtime:
                try:
                    return gpu(0)
                except Exception:
                    pass
        return cpu(0)
    except Exception:
        from decord import cpu
        return cpu(0)

def _flush_chunk_to_drive(tmp_dir, final_dir, upto_index):
    """Move files <= upto_index (inclusive) from tmp to Drive. Keeps numbering stable."""
    os.makedirs(final_dir, exist_ok=True)
    moved = 0
    for f in sorted(os.listdir(tmp_dir)):
        if not f.lower().endswith(".jpg"): continue
        try:
            idx = int(os.path.splitext(f)[0])
        except:
            continue
        if idx <= upto_index:
            src = os.path.join(tmp_dir, f)
            dst = os.path.join(final_dir, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1
    return moved

def extract_with_decord(media_path, indices, out_tmp, out_final, start_number):
    ok = ensure_decord()
    if not ok:
        return 0, 0.0, "no_decord"
    import imageio.v2 as imageio
    from decord import VideoReader

    ctx = _choose_decord_ctx()
    t0 = time.time()
    try:
        vr = VideoReader(media_path, ctx=ctx)
    except Exception as e:
        return 0, 0.0, f"decord_open_failed:{str(e)[:80]}"

    n_total = len(vr)
    if EXTRACTION_MODE == "all":
        picks = list(range(n_total))
    elif EXTRACTION_MODE == "first_n":
        picks = list(range(min(NUM_FRAMES_TO_EXTRACT, n_total)))
    else:
        picks = sorted(set([i for i in indices if 0 <= i < n_total]))
    if not picks:
        return 0, 0.0, "no_frames"

    os.makedirs(out_tmp, exist_ok=True)
    written = 0
    last_flushed = start_number - 1  # last index moved to Drive

    B = max(1, int(DECORD_BATCH))
    for i in range(0, len(picks), B):
        batch_idx = picks[i:i+B]
        try:
            batch_nd = vr.get_batch(batch_idx)   # NDArray
            batch_np = batch_nd.asnumpy()        # -> numpy (host)
        except Exception as e:
            return written, (time.time()-t0), f"decord_batch_failed:{str(e)[:80]}"

        # write frames
        for j in range(batch_np.shape[0]):
            arr = batch_np[j]
            if RESIZE_TO_540:
                arr = _resize_short_side(arr, short=540)
            if EXTRACTION_MODE == "from_lists":
                outname = f"{int(batch_idx[j]):07d}.jpg"
            else:
                outname = f"{start_number+written:07d}.jpg"
            outpath = os.path.join(out_tmp, outname)
            try:
                imageio.imwrite(outpath, arr, quality=max(1, 100 - JPG_QUALITY*12))
            except Exception as e:
                return written, (time.time()-t0), f"image_write_failed:{str(e)[:80]}"
            written += 1

        # chunked flush to Drive every FLUSH_EVERY_IMAGES
        if FLUSH_EVERY_IMAGES and (written % FLUSH_EVERY_IMAGES == 0):
            upto = start_number + written - 1
            moved = _flush_chunk_to_drive(out_tmp, out_final, upto)
            last_flushed = upto if moved > 0 else last_flushed

    # final flush for any remaining files (only when tmp and final differ)
    if os.path.isdir(out_tmp) and out_tmp != out_final:
        upto = start_number + written - 1
        _flush_chunk_to_drive(out_tmp, out_final, upto)

    # Clean tmp dir if empty
    try:
        if out_tmp != out_final and os.path.isdir(out_tmp) and not os.listdir(out_tmp):
            shutil.rmtree(out_tmp, ignore_errors=True)
    except:
        pass

    return written, (time.time()-t0), ("ok_gpu" if "gpu" in str(ctx).lower() else "ok_cpu")

# ---------------------- FFMPEG PATHS (resume-safe) ----------------------
def ffmpeg_select_fast(media_path, indices, out_tmp, out_final, start_number):
    os.makedirs(out_tmp, exist_ok=True)
    if EXTRACTION_MODE == "all":
        vf = []
    elif EXTRACTION_MODE == "first_n":
        vf = ["-vframes", str(int(NUM_FRAMES_TO_EXTRACT))]
    else:
        frames = sorted(set(int(x) for x in indices))
        if not frames: return 0, 0.0, "no_frames"
        eqn = "+".join([f"eq(n\\,{int(n)})" for n in frames])
        filt = f"select='{eqn}',setpts=N/FRAME_RATE/TB"
        if RESIZE_TO_540:
            filt += ",scale='if(gt(iw,ih),-1,540)':'if(gt(iw,ih),540,-1)'"
        vf = ["-vf", filt]
    q = ["-q:v", str(int(JPG_QUALITY))]
    t0 = time.time()
    before = len(list_existing_jpgs(out_tmp))
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y",
           "-threads","2","-i", media_path] + vf + q + \
          ["-vsync","0","-start_number", str(int(start_number)),
           os.path.join(out_tmp, "%07d.jpg")]
    subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=FFMPEG_TIMEOUT_SECS)
    after = len(list_existing_jpgs(out_tmp))

    # Move everything produced in this run to Drive (skip when writing directly to final)
    if out_tmp != out_final:
        os.makedirs(out_final, exist_ok=True)
        for f in list_existing_jpgs(out_tmp):
            src = os.path.join(out_tmp, f); dst = os.path.join(out_final, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        try:
            if not os.listdir(out_tmp): shutil.rmtree(out_tmp, ignore_errors=True)
        except: pass
    return max(0, after-before), (time.time()-t0), "ok"

def ffmpeg_ranges(media_path, indices, out_tmp, out_final, start_number):
    runs = to_ranges(indices)
    if not runs:
        return 0, 0.0, "no_frames"
    written_total, t0 = 0, time.time()
    for (a,b) in runs:
        filt = f"select='between(n\\,{a}\\,{b})',setpts=N/FRAME_RATE/TB"
        if RESIZE_TO_540:
            filt += ",scale='if(gt(iw,ih),-1,540)':'if(gt(iw,ih),540,-1)'"
        cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y",
               "-threads","2","-i", media_path,
               "-vf", filt, "-q:v", str(int(JPG_QUALITY)),
               # Name outputs by true frame index using -start_number a
               "-vsync","0","-start_number", str(int(a)),
               os.path.join(out_tmp, "%07d.jpg")]
        subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=FFMPEG_TIMEOUT_SECS)
        written_total += (b - a + 1)

        # Move this run immediately to Drive to limit /content size (skip when writing directly)
        if out_tmp != out_final:
            os.makedirs(out_final, exist_ok=True)
            for f in list_existing_jpgs(out_tmp):
                src = os.path.join(out_tmp, f); dst = os.path.join(out_final, f)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
    try:
        if out_tmp != out_final and os.path.isdir(out_tmp) and not os.listdir(out_tmp): shutil.rmtree(out_tmp, ignore_errors=True)
    except: pass
    return written_total, (time.time()-t0), "ok"

# ---------------------- REQUEST SIZE & RESUME LOGIC ----------------------
def requested_count_for(uid, media_path):
    if EXTRACTION_MODE == "from_lists":
        return len(load_frames_for(uid))
    elif EXTRACTION_MODE == "first_n":
        return int(NUM_FRAMES_TO_EXTRACT)
    else:
        nb = ffprobe_nb_frames(media_path)
        return nb if isinstance(nb, int) and nb > 0 else 0

def remaining_indices_for_from_lists(uid, out_final_dir):
    """Return the subset of annotated indices that are not yet present on disk.
    Uses filename existence (e.g., 0001234.jpg) rather than a simple count.
    """
    idx_all = sorted(set(load_frames_for(uid)))
    if not idx_all:
        return []
    existing = list_existing_indices(out_final_dir)
    missing = [i for i in idx_all if int(i) not in existing]
    return missing

# ---------------------- MAIN EXTRACTOR ----------------------
def extract_one(uid):
    media_path = os.path.join(MEDIA_DIR, f"{uid}.mp4")
    if not os.path.exists(media_path):
        note = "media_missing"
        append_log(uid, 0, 0, 0, note)
        return uid, 0, 0.0, note

    out_final = os.path.join(FRAMES_DRIVE_ROOT, uid)
    # For local runs, write directly into final directory; on Colab use a tmp dir then move
    out_tmp   = os.path.join(FRAMES_TMP_ROOT, uid)
    if not os.path.isdir('/content'):
        out_tmp = out_final

    req_ct = requested_count_for(uid, media_path)
    existing_final_ct = len(list_existing_jpgs(out_final))
    if existing_final_ct >= req_ct > 0:
        note = "skip_already_done"
        if PRINT_PER_ITEM_INFO:
            print(f"  {uid}: already complete ({existing_final_ct}/{req_ct}) → skip")
        append_log(uid, req_ct, 0, existing_final_ct, note)
        return uid, 0, 0.0, note

    # Determine indices still needed and base numbering
    if EXTRACTION_MODE == "from_lists":
        indices_needed = remaining_indices_for_from_lists(uid, out_final)
        base_start = 0  # not used for naming in from_lists mode
    elif EXTRACTION_MODE == "first_n":
        indices_needed, base_start = [], existing_final_ct
    else:  # 'all'
        indices_needed, base_start = [], existing_final_ct

    try:
        if SPEED_MODE == "decord":
            new_ct, secs, note = extract_with_decord(
                media_path, indices_needed, out_tmp, out_final, start_number=base_start
            )
            if (note not in ("ok", "ok_gpu", "ok_cpu")) or new_ct == 0:
                # fallback to ffmpeg
                if EXTRACTION_MODE == "from_lists" and indices_needed:
                    new_ct, secs, note = ffmpeg_ranges(media_path, indices_needed, out_tmp, out_final, start_number=base_start)
                else:
                    new_ct, secs, note = ffmpeg_select_fast(media_path, indices_needed, out_tmp, out_final, start_number=base_start)

        elif SPEED_MODE == "ffmpeg_ranges" and EXTRACTION_MODE == "from_lists":
            new_ct, secs, note = ffmpeg_ranges(media_path, indices_needed, out_tmp, out_final, start_number=base_start)

        else:
            # Enforce correct naming for from_lists
            if EXTRACTION_MODE == "from_lists" and indices_needed:
                new_ct, secs, note = ffmpeg_ranges(media_path, indices_needed, out_tmp, out_final, start_number=base_start)
            else:
                new_ct, secs, note = ffmpeg_select_fast(media_path, indices_needed, out_tmp, out_final, start_number=base_start)

        total_now = len(list_existing_jpgs(out_final))
        if PRINT_PER_ITEM_INFO:
            nb = ffprobe_nb_frames(media_path) or "unknown"
            print(f"  {uid}: wrote {new_ct} jpgs → now={total_now}/{req_ct} | total≈{nb} | {note} in {secs:.1f}s")
        append_log(uid, req_ct, new_ct, total_now, note)
        return uid, new_ct, secs, note

    except subprocess.CalledProcessError:
        note = "ffmpeg_failed"
        append_log(uid, req_ct, 0, len(list_existing_jpgs(out_final)), note)
        return uid, 0, 0.0, note
    except Exception as e:
        note = f"error:{str(e)[:120]}"
        append_log(uid, req_ct, 0, len(list_existing_jpgs(out_final)), note)
        return uid, 0, 0.0, note

if __name__ == "__main__":
    # ---------------------- RUN ----------------------
    ids_all = load_ids()
    print(f"Source: {SOURCE_KIND} (dataset={DATASET})")
    print(f"Speed mode: {SPEED_MODE} | Use GPU: {USE_GPU_DECORD} | Resize to 540: {RESIZE_TO_540} | JPG quality: {JPG_QUALITY}")
    sel_msg = f"Selected {len(ids_all)} item(s)"
    if USE_PART_FILES:
        sel_msg += f" from parts {PART_INDEXES or '[custom files]'}"
    print(f"{sel_msg}; mode='{EXTRACTION_MODE}' | FLUSH_EVERY_IMAGES={FLUSH_EVERY_IMAGES} | DECORD_BATCH={DECORD_BATCH}")

    t0 = time.time()
    rows = []
    if ids_all:
        # Tune parallelism: more workers locally, fewer on Colab; allow env override
        cpu = os.cpu_count() or 4
        default_workers = 4 if os.path.isdir('/content') else max(1, min(8, max(1, cpu // 2)))
        workers = int(os.environ.get('N_EXTRACT_WORKERS', default_workers))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [ex.submit(extract_one, u) for u in ids_all]
            for fut in tqdm(as_completed(futs), total=len(futs)):
                rows.append(fut.result())

    t1 = time.time()
    total_new = sum(r[1] for r in rows)
    print("\n========== SUMMARY ==========")
    print(f"Items processed: {len(rows)}")
    print(f"Frames extracted (new): {total_new}")
    print(f"Local scratch: {FRAMES_TMP_ROOT}  → merged into: {FRAMES_DRIVE_ROOT}")
    print(f"Log CSV: {LOG_CSV}")
    print(f"Total wall time: {timedelta(seconds=int(t1-t0))}")
    if rows:
        print("\n[uid | new | note]")
        for uid, new_ct, secs, note in rows[:60]:
            print(f"{uid} | {new_ct:4d} | {note}")
    print("=============================================")
