"""Consistency checker for the YOLO noun fine-tuning dataset.

Adjust the CONFIG dictionary below if your directory layout changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Update values here rather than wiring command-line flags everywhere.
CONFIG = {
    "data_root": Path("D:/Thesis/Ego4d-LiteSTA/local_extraction/v2"),
    "frames_subdir": Path("extracted_frames"),
    "labels_subdir": Path("yolo_labels_540/clips"),
    "report_limit": 20,  # cap on the number of problematic clips to display
    "example_limit": 8,  # per-clip cap when showing missing/orphaned files
}


@dataclass
class ClipDelta:
    clip_id: str
    frame_count: int
    label_count: int
    missing_labels: list[str]
    orphan_labels: list[str]


def _iter_file_stems(folder: Path, suffixes: Iterable[str]) -> set[str]:
    suffixes = tuple(suffix.lower() for suffix in suffixes)
    return {
        item.stem
        for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in suffixes
    }


def validate_clips(frames_root: Path, labels_root: Path) -> tuple[list[ClipDelta], dict[str, int]]:
    clip_reports: list[ClipDelta] = []
    total_frames = total_labels = total_missing = total_orphaned = 0

    frame_clips = {entry.name: entry for entry in frames_root.iterdir() if entry.is_dir()}
    label_clips = {entry.name: entry for entry in labels_root.iterdir() if entry.is_dir()}

    shared_clips = sorted(frame_clips.keys() & label_clips.keys())

    for clip_id in shared_clips:
        frame_dir = frame_clips[clip_id]
        label_dir = label_clips[clip_id]

        frame_ids = _iter_file_stems(frame_dir, suffixes=(".jpg", ".jpeg", ".png"))
        label_ids = _iter_file_stems(label_dir, suffixes=(".txt",))

        missing = sorted(frame_ids - label_ids)
        orphan = sorted(label_ids - frame_ids)

        total_frames += len(frame_ids)
        total_labels += len(label_ids)
        total_missing += len(missing)
        total_orphaned += len(orphan)

        if missing or orphan:
            clip_reports.append(
                ClipDelta(
                    clip_id=clip_id,
                    frame_count=len(frame_ids),
                    label_count=len(label_ids),
                    missing_labels=missing,
                    orphan_labels=orphan,
                )
            )

    summary = {
        "clips_with_frames": len(frame_clips),
        "clips_with_labels": len(label_clips),
        "clips_in_both": len(shared_clips),
        "total_frames": total_frames,
        "total_labels": total_labels,
        "missing_labels": total_missing,
        "orphan_labels": total_orphaned,
    }

    return clip_reports, summary


def main() -> None:
    data_root: Path = CONFIG["data_root"]
    frames_root = data_root / CONFIG["frames_subdir"]
    labels_root = data_root / CONFIG["labels_subdir"]

    if not frames_root.exists():
        raise SystemExit(f"Frames directory not found: {frames_root}")
    if not labels_root.exists():
        raise SystemExit(f"Labels directory not found: {labels_root}")

    frame_clips = {entry.name for entry in frames_root.iterdir() if entry.is_dir()}
    label_clips = {entry.name for entry in labels_root.iterdir() if entry.is_dir()}

    missing_label_dirs = sorted(frame_clips - label_clips)
    orphan_label_dirs = sorted(label_clips - frame_clips)

    clip_reports, summary = validate_clips(frames_root, labels_root)

    print("=== YOLO Dataset Consistency Report ===")
    print(f"Frames root: {frames_root}")
    print(f"Labels root: {labels_root}")
    print()

    print("Top-level clip directories:")
    print(f"  Clips with frames : {summary['clips_with_frames']}")
    print(f"  Clips with labels : {summary['clips_with_labels']}")
    print(f"  Clips in both     : {summary['clips_in_both']}")
    print(f"  Frame-only clips  : {len(missing_label_dirs)}")
    print(f"  Label-only clips  : {len(orphan_label_dirs)}")
    print()

    if missing_label_dirs:
        print("Clips missing label directories (sample):")
        for clip_id in missing_label_dirs[: CONFIG["report_limit"]]:
            print(f"  - {clip_id}")
        if len(missing_label_dirs) > CONFIG["report_limit"]:
            print("  …")
        print()

    if orphan_label_dirs:
        print("Clips without matching frame directories (sample):")
        for clip_id in orphan_label_dirs[: CONFIG["report_limit"]]:
            print(f"  - {clip_id}")
        if len(orphan_label_dirs) > CONFIG["report_limit"]:
            print("  …")
        print()

    print("File-level summary across shared clips:")
    print(f"  Total frames examined     : {summary['total_frames']}")
    print(f"  Total labels found        : {summary['total_labels']}")
    print(f"  Frames missing labels     : {summary['missing_labels']}")
    print(f"  Labels without frame image: {summary['orphan_labels']}")
    print()

    if clip_reports:
        print("Clips with mismatched file counts (sample):")
        for report in clip_reports[: CONFIG["report_limit"]]:
            print(f"  - {report.clip_id} (frames={report.frame_count}, labels={report.label_count})")
            if report.missing_labels:
                sample_missing = ", ".join(report.missing_labels[: CONFIG["example_limit"]])
                suffix = " …" if len(report.missing_labels) > CONFIG["example_limit"] else ""
                print(f"      missing labels: {sample_missing}{suffix}")
            if report.orphan_labels:
                sample_orphan = ", ".join(report.orphan_labels[: CONFIG["example_limit"]])
                suffix = " …" if len(report.orphan_labels) > CONFIG["example_limit"] else ""
                print(f"      orphan labels : {sample_orphan}{suffix}")
        if len(clip_reports) > CONFIG["report_limit"]:
            print("    …")
    else:
        print("No per-frame mismatches detected in shared clips.")


if __name__ == "__main__":
    main()
