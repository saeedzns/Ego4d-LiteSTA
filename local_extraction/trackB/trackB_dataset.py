"""Track B Dataset & Utilities

Provides:
 - Manifest loading (JSON list or JSONL).
 - Flexible schema parsing for candidate boxes, labels, and TTC.
 - Caching of parsed manifest records and per-UID frame listings.
 - Invalid record logging (CSV) for QA.
 - Optional on-the-fly tokenization (image + temporal window) with FGTP + fusion left to caller.
 - TTC normalization (z-score) with fallback if insufficient data.

Configuration loaded from configs/trackB.yaml

Schema Assumptions (will lock once final schema provided):
Record keys (any combination accepted):
  uid | video_uid | video_id : str
  frame | frame_idx | frame_index : int (frame index) OR
  frame_name | image : str (filename in UID directory)
  candidates | boxes | objects : list of candidate entries
Each candidate entry (dict or list):
  Dict keys for boxes: x1|xmin|left, y1|ymin|top, x2|xmax|right, y2|ymax|bottom
  Class label: cls | label (bool/str/int mapped to int)
  TTC: ttc | time_to_contact (float seconds)

If manifests are empty, dataset yields synthetic candidates.
"""
from __future__ import annotations

import sys
import json
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image

# ================== YAML CONFIG LOADING ==================
def _load_yaml_config():
    """Load YAML config for Track B."""
    _THIS_DIR = Path(__file__).resolve().parent
    _LOCAL_EXTRACTION = _THIS_DIR.parent
    for p in [str(_LOCAL_EXTRACTION), str(_LOCAL_EXTRACTION.parent), str(_THIS_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from core import load_config
        return load_config('trackB')
    except Exception:
        return None

_cfg = _load_yaml_config()
# =========================================================

# We defer torch imports until needed so importing dataset does not require torch
import torch
from trackB_tokenizer import (
    TokenizerConfig,
    build_backbone,
    build_transform,
    list_uid_frames,
    sample_window_ending_at,
    image_grid_tokens,
    video_grid_tokens,
    get_tokens,  # Unified tokenizer that handles both ResNet18 and VideoMAE
)

@dataclass
class ManifestStats:
    total_records: int = 0
    total_candidates: int = 0
    ttc_mean: float = 1.0
    ttc_std: float = 1.0
    ttc_min: float = 0.0
    ttc_max: float = 0.0


# TTC bin thresholds from YAML or default
def _get_ttc_thresholds() -> Tuple[float, float, float]:
    if _cfg:
        bins = _cfg.get('multi_task.ttc_thresholds', [0.5, 1.0, 2.0])
        if isinstance(bins, (list, tuple)) and len(bins) >= 3:
            return (float(bins[0]), float(bins[1]), float(bins[2]))
    return (0.5, 1.0, 2.0)

TTC_BIN_THRESHOLDS: Tuple[float, float, float] = _get_ttc_thresholds()

def ttc_to_bin(ttc: float, thresholds: Tuple[float, float, float] = TTC_BIN_THRESHOLDS) -> int:
    """
    Map a TTC value in seconds to a coarse bin index.

    Default bins (t in seconds):
      0: [0.0, 0.5)
      1: [0.5, 1.0)
      2: [1.0, 2.0)
      3: [2.0, +inf)
    """
    try:
        x = float(ttc)
    except Exception:
        x = 0.0
    if x < thresholds[0]:
        return 0
    if x < thresholds[1]:
        return 1
    if x < thresholds[2]:
        return 2
    return 3


def _safe_float(v: Any, default: float = 0.0) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return default


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    # Try JSON array
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # JSONL fallback
    records: List[Dict[str, Any]] = []
    try:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except Exception:
                    continue
    except Exception:
        pass
    return records


def discover_manifest(root: Path) -> Optional[Path]:
    candidates = [
        'head_train.jsonl',
        'head_train.json',
        'head_train_clip.json',
        'head_train_clip.jsonl',
        'head_train_video.json',
        'head_train_video.jsonl',
    ]
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return None


def latest_stageB_run(trackA_runs_root: Path) -> Optional[Path]:
    """Return the most recent Track A Stage B run directory that produced head manifests."""
    if not trackA_runs_root.exists():
        return None
    candidates: List[Path] = []
    for item in trackA_runs_root.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if not name.startswith('trackA_stageB_'):
            continue
        has_manifest = any((item / fname).exists() for fname in ('head_train.jsonl', 'head_train.json'))
        if has_manifest:
            candidates.append(item)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def resolve_stageB_manifest(stageB_run: Optional[Path], base_name: str) -> Optional[Path]:
    """Return manifest path inside a Stage B run for the given base name (e.g., 'head_train')."""
    if stageB_run is None:
        return None
    for suffix in ('.jsonl', '.json'):
        candidate = stageB_run / f"{base_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def discover_stageB_manifest(trackA_runs_root: Path, base_name: str) -> Optional[Path]:
    """Find the latest Stage B manifest of the requested type, if available."""
    stageB_run = latest_stageB_run(trackA_runs_root)
    return resolve_stageB_manifest(stageB_run, base_name)


class TrackBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frames_root: Path,
        manifests_root: Path,
        manifest_path: Optional[Path] = None,
        tokenizer_cfg: Optional[TokenizerConfig] = None,
        time_len: Optional[int] = None,
        candidate_limit: int = 32,
        normalize_ttc: bool = True,
        synthetic_if_empty: bool = True,
        cache_dir: Optional[Path] = None,
        seed: int = 0,
        tokens_root: Optional[Path] = None,  # Path to pre-extracted ResNet18 tokens
    ):
        super().__init__()
        random.seed(seed)
        self.frames_root = frames_root
        self.manifests_root = manifests_root
        self.manifest_path = manifest_path or discover_manifest(manifests_root)
        self.tokenizer_cfg = tokenizer_cfg or TokenizerConfig()
        if time_len is not None:
            self.tokenizer_cfg.time_len = time_len
        self.candidate_limit = candidate_limit
        self.normalize_ttc = normalize_ttc
        self.synthetic_if_empty = synthetic_if_empty
        
        # Pre-extracted tokens directory (for ~120x faster training)
        self.tokens_root = tokens_root
        if self.tokens_root is not None:
            print(f"[TrackBDataset] Using pre-extracted tokens from: {self.tokens_root}")
        
        # Default cache directory: local_extraction/runs/Track_B/cache
        if cache_dir is None:
            root_local_extraction = frames_root.parent.parent  # .../local_extraction
            self.cache_dir = root_local_extraction / 'runs' / 'Track_B' / 'cache'
        else:
            self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Backbone & transform (single instance; DataLoader num_workers should stay 0 unless reworked)
        self.backbone = build_backbone(self.tokenizer_cfg)
        self.transform = build_transform(self.tokenizer_cfg)

        # Load manifest
        self.records: List[Dict[str, Any]] = []
        if self.manifest_path is not None:
            self.records = load_manifest(self.manifest_path)
        self._maybe_convert_stageB_records()
        if not self.records and synthetic_if_empty:
            # We'll fabricate synthetic records (one per UID last frame)
            self._fabricate_synthetic_records()

        # Parse, cache per-uid frame lists
        self.uid_to_frames: Dict[str, List[Path]] = {}
        self.invalid_records: List[Dict[str, Any]] = []
        self.parsed: List[Dict[str, Any]] = []

        for r in tqdm(self.records, desc='[TrackBDataset] parse records'):
            parsed = self._parse_record(r)
            if parsed is None:
                self.invalid_records.append(r)
                continue
            uid = parsed['uid']
            if uid not in self.uid_to_frames:
                fdir = self.frames_root / uid
                if fdir.exists():
                    self.uid_to_frames[uid] = list_uid_frames(fdir)
                else:
                    self.uid_to_frames[uid] = []
            self.parsed.append(parsed)

        # Compute TTC stats
        self.stats = self._compute_ttc_stats(self.parsed)
        if self.normalize_ttc:
            for p in self.parsed:
                for c in p['candidates']:
                    c['ttc_norm'] = (c['ttc'] - self.stats.ttc_mean) / (self.stats.ttc_std if self.stats.ttc_std > 1e-6 else 1.0)

        # Write invalid log
        if self.invalid_records:
            inv_path = self.cache_dir / 'invalid_records.csv'
            with inv_path.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['raw_record_json'])
                for r in self.invalid_records:
                    w.writerow([json.dumps(r)])

    def _maybe_convert_stageB_records(self) -> None:
        if not self.records:
            return
        sample = self.records[0]
        if not isinstance(sample, dict):
            return
        if 'image_path' not in sample or 'candidate_box' not in sample:
            return

        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        cwd = Path.cwd()

        for rec in self.records:
            if not isinstance(rec, dict):
                continue
            image_path = rec.get('image_path')
            box = rec.get('candidate_box')
            if not isinstance(image_path, str) or not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            img_path = Path(image_path.replace('\\', '/'))
            if not img_path.is_absolute():
                img_path = (cwd / img_path).resolve()
            frame_name = img_path.name
            uid = img_path.parent.name if img_path.parent else None
            if not uid:
                continue
            try:
                frame_idx = int(Path(frame_name).stem)
            except Exception:
                frame_idx = None

            key = (uid, frame_name)
            entry = grouped.get(key)
            if entry is None:
                entry = {
                    'uid': uid,
                    'frame_name': frame_name,
                    'frame_idx': frame_idx,
                    'image': str(img_path),
                    'candidates': [],
                }
                grouped[key] = entry

            is_pos = rec.get('is_positive')
            if isinstance(is_pos, bool):
                cls = 1 if is_pos else 0
            else:
                cls = 1 if str(is_pos).lower() in {'1', 'true', 'pos', 'positive'} else 0

            ttc_val = _safe_float(rec.get('ttc', 0.0), 0.0)
            verb_id = rec.get('verb_id')
            noun_id = rec.get('noun_id')
            try:
                verb_id = int(verb_id) if verb_id is not None else None
            except Exception:
                verb_id = None
            try:
                noun_id = int(noun_id) if noun_id is not None else None
            except Exception:
                noun_id = None

            candidate = {
                'x1': float(box[0]),
                'y1': float(box[1]),
                'x2': float(box[2]),
                'y2': float(box[3]),
                'cls': cls,
                'ttc': ttc_val,
                'is_positive': bool(is_pos) if is_pos is not None else bool(cls),
                'verb_id': verb_id,
                'noun_id': noun_id,
            }

            entry['candidates'].append(candidate)

        if grouped:
            self.records = list(grouped.values())

    def _fabricate_synthetic_records(self):
        # Each UID: create a fake record with a single candidate
        if not self.frames_root.exists():
            return
        for uid_dir in self.frames_root.iterdir():
            if not uid_dir.is_dir():
                continue
            frames = list_uid_frames(uid_dir)
            if not frames:
                continue
            last_frame = frames[-1].name
            self.records.append({
                'uid': uid_dir.name,
                'frame_name': last_frame,
                'candidates': [
                    {
                        'x1': 0,
                        'y1': 0,
                        'x2': 32,
                        'y2': 32,
                        'cls': 0,
                        'ttc': 1.0,
                        'is_positive': False,
                        'verb_id': None,
                        'noun_id': None,
                    }
                ]
            })

    def _parse_record(self, r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        uid = r.get('uid') or r.get('video_uid') or r.get('video_id')
        frame_abs = None
        if 'image' in r and isinstance(r['image'], str):
            frame_abs = r['image']
            # derive uid from parent folder if not provided
            if uid is None:
                try:
                    uid = Path(frame_abs).parent.name
                except Exception:
                    uid = None
        if uid is None:
            return None
        frame_idx = r.get('frame') or r.get('frame_idx') or r.get('frame_index')
        frame_name = r.get('frame_name') or r.get('image')
        # Candidate list
        raw_cands = r.get('candidates') or r.get('boxes') or r.get('objects') or []
        candidates: List[Dict[str, Any]] = []
        for c in raw_cands:
            if isinstance(c, dict):
                x1 = c.get('x1') or c.get('xmin') or c.get('left')
                y1 = c.get('y1') or c.get('ymin') or c.get('top')
                x2 = c.get('x2') or c.get('xmax') or c.get('right')
                y2 = c.get('y2') or c.get('ymax') or c.get('bottom')
                if None in (x1, y1, x2, y2):
                    continue
                cls = c.get('cls')
                if cls is None:
                    cls = c.get('label')
                    if isinstance(cls, bool):
                        cls = 1 if cls else 0
                    elif isinstance(cls, str):
                        cls = 1 if cls.lower() in ('1', 'true', 'active', 'pos', 'positive') else 0
                if not isinstance(cls, int):
                    cls = 0
                ttc_raw = c.get('ttc') or c.get('time_to_contact') or 0.0
                ttc_val = _safe_float(ttc_raw, 0.0)
                is_pos_raw = c.get('is_positive')
                if is_pos_raw is None:
                    # Fallback: treat cls>0 as positive when explicit flag is missing
                    is_pos_flag = 1 if int(cls) == 1 else 0
                else:
                    is_pos_flag = 1 if bool(is_pos_raw) else 0
                verb_id = c.get('verb_id')
                noun_id = c.get('noun_id')
                try:
                    verb_id_int = int(verb_id) if verb_id is not None else -1
                except Exception:
                    verb_id_int = -1
                try:
                    noun_id_int = int(noun_id) if noun_id is not None else -1
                except Exception:
                    noun_id_int = -1
                candidates.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'cls': int(cls),
                    'ttc': ttc_val,
                    'is_positive': int(is_pos_flag),
                    'verb_id': verb_id_int,
                    'noun_id': noun_id_int,
                    'ttc_bin': ttc_to_bin(ttc_val),
                })
            elif isinstance(c, (list, tuple)) and len(c) >= 4:
                candidates.append({
                    'bbox': (float(c[0]), float(c[1]), float(c[2]), float(c[3])),
                    'cls': 0,
                    'ttc': 0.0,
                    'is_positive': 0,
                    'verb_id': -1,
                    'noun_id': -1,
                    'ttc_bin': ttc_to_bin(0.0),
                })
        # If no candidates but gt_box exists, build a single candidate
        if not candidates:
            gt = r.get('gt_box')
            if isinstance(gt, (list, tuple)) and len(gt) >= 4:
                x1, y1, x2, y2 = gt[:4]
                cls = r.get('verb_id')
                if cls is None:
                    cls = r.get('noun_id', 0)
                if not isinstance(cls, int):
                    try:
                        cls = int(cls)
                    except Exception:
                        cls = 0
                ttc_rec = _safe_float(r.get('ttc', 0.0), 0.0)
                verb_id = r.get('verb_id')
                noun_id = r.get('noun_id')
                try:
                    verb_id_int = int(verb_id) if verb_id is not None else -1
                except Exception:
                    verb_id_int = -1
                try:
                    noun_id_int = int(noun_id) if noun_id is not None else -1
                except Exception:
                    noun_id_int = -1
                candidates.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'cls': int(cls),
                    'ttc': ttc_rec,
                    'is_positive': 1,
                    'verb_id': verb_id_int,
                    'noun_id': noun_id_int,
                    'ttc_bin': ttc_to_bin(ttc_rec),
                })

        return {
            'uid': str(uid),
            'frame_idx': frame_idx if isinstance(frame_idx, int) else None,
            'frame_name': frame_name if isinstance(frame_name, str) else None,
            'frame_abs': frame_abs,
            'candidates': candidates,
        }

    def _compute_ttc_stats(self, parsed: List[Dict[str, Any]]) -> ManifestStats:
        ttcs: List[float] = []
        for p in parsed:
            for c in p['candidates']:
                ttcs.append(c['ttc'])
        stats = ManifestStats()
        stats.total_records = len(parsed)
        stats.total_candidates = len(ttcs)
        if ttcs:
            stats.ttc_min = min(ttcs)
            stats.ttc_max = max(ttcs)
            mean = sum(ttcs) / len(ttcs)
            var = sum((x - mean) ** 2 for x in ttcs) / max(1, len(ttcs) - 1)
            std = var ** 0.5
            stats.ttc_mean = mean
            stats.ttc_std = std if std > 1e-6 else 1.0
        return stats

    def __len__(self) -> int:
        return len(self.parsed)

    def _resolve_frame_path(self, uid: str, frame_idx: Optional[int], frame_name: Optional[str], frame_abs: Optional[str]) -> Optional[Path]:
        """Resolve an absolute frame path with robust fallbacks.

        Priority:
        1) Use provided absolute path if it exists.
        2) Fallback to frames_root/uid/<basename(frame_name or frame_abs)> if available.
        3) Use frame_idx within the uid directory if available.
        4) Use the last frame for the uid as a final fallback.
        """
        # 1) Direct absolute path
        if frame_abs:
            p = Path(frame_abs)
            if p.exists():
                return p
            # If absolute path doesn't exist on this machine, keep basename for fallback
            frame_basename = p.name
        else:
            frame_basename = None

        # Prepare uid directory
        fdir = self.frames_root / uid
        if not fdir.exists():
            return None

        # 2) Try by explicit frame_name (use basename to avoid accidental absolute paths)
        if frame_name:
            try:
                name_only = Path(frame_name).name
            except Exception:
                name_only = frame_name
            cand = fdir / name_only
            if cand.exists():
                return cand

        # 2b) Try basename derived from missing absolute frame path
        if frame_basename:
            cand = fdir / frame_basename
            if cand.exists():
                return cand

        # 3) Use indexed frame if available
        frames = self.uid_to_frames.get(uid)
        if not frames:
            frames = list_uid_frames(fdir)
            self.uid_to_frames[uid] = frames
        if frame_idx is not None and 0 <= frame_idx < len(frames):
            return frames[frame_idx]

        # 4) Last frame fallback
        return frames[-1] if frames else None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.parsed[idx]
        uid = rec['uid']
        frame_path = self._resolve_frame_path(uid, rec['frame_idx'], rec['frame_name'], rec.get('frame_abs'))
        if frame_path is None:
            # Debug hint for first few failures
            if idx < 5:
                print(f"[TrackBDataset][debug] missing frame path uid={uid} frame_name={rec.get('frame_name')} frame_abs={rec.get('frame_abs')}")
        if frame_path is None:
            # return empty sample
            return {'uid': uid, 'valid': False}

        # Tokenize using unified function (handles both ResNet18 and VideoMAE)
        # Supports pre-extracted tokens for ~120x faster training
        img_tokens, vid_tokens, hw = get_tokens(
            frame_path, self.frames_root, self.tokenizer_cfg, self.backbone, self.transform,
            tokens_root=self.tokens_root
        )

        # Candidate subset
        cands = rec['candidates']
        if self.candidate_limit and len(cands) > self.candidate_limit:
            sel = list(range(len(cands)))
            random.shuffle(sel)
            sel = sel[:self.candidate_limit]
            cands = [cands[i] for i in sel]

        # Load image for width/height (for ROI -> grid mapping, if needed upstream)
        with Image.open(str(frame_path)) as im:
            W, H = im.size

        bboxes = [c['bbox'] for c in cands]
        labels = [c['cls'] for c in cands]
        ttcs = [c['ttc'] for c in cands]
        ttcs_norm = [c.get('ttc_norm', c['ttc']) for c in cands]
        is_pos_list = [c.get('is_positive', 1 if c.get('cls', 0) == 1 else 0) for c in cands]
        verb_ids = [c.get('verb_id', -1) for c in cands]
        noun_ids = [c.get('noun_id', -1) for c in cands]
        ttc_bins = [c.get('ttc_bin', ttc_to_bin(c.get('ttc', 0.0))) for c in cands]

        if not bboxes:
            # Provide early debug sample (rare) to help diagnose manifest mismatch
            if idx < 3:
                print(f"[TrackBDataset][debug] zero candidates after parsing uid={uid} raw_record_keys={list(rec.keys())}")

        return {
            'uid': uid,
            'frame_path': frame_path,
            'img_tokens': img_tokens,         # (N,512)
            'vid_tokens': vid_tokens,         # (T,N,512)
            'hw': hw,
            'image_size': (W, H),
            'bboxes': bboxes,
            # Primary classification label (currently next-active vs not for Stage B manifests)
            'labels': torch.tensor(labels, dtype=torch.long),
            # Explicit multi-task supervision fields per candidate
            'is_positive': torch.tensor(is_pos_list, dtype=torch.long),
            'gt_noun_id': torch.tensor(noun_ids, dtype=torch.long),
            'gt_verb_id': torch.tensor(verb_ids, dtype=torch.long),
            'gt_ttc': torch.tensor(ttcs, dtype=torch.float32),
            'gt_ttc_bin': torch.tensor(ttc_bins, dtype=torch.long),
            # Legacy TTC fields (kept for backward-compatibility)
            'ttc': torch.tensor(ttcs, dtype=torch.float32),
            'ttc_norm': torch.tensor(ttcs_norm, dtype=torch.float32),
            'valid': True,
        }


def trackB_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = [b for b in batch if b.get('valid')]
    if not batch:
        return {'valid': False}
    # For simplicity we keep variable-length candidate sets; downstream can iterate per sample.
    return {
        'samples': batch,
        'valid': True,
    }


__all__ = [
    'TrackBDataset',
    'trackB_collate',
    'ManifestStats',
]
