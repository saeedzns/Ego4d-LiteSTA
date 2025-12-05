#!/usr/bin/env python3
"""
Scan for STA clip MP4 files and record their paths.

This script is idempotent - running it multiple times will produce the same result.

Usage:
    python scan_clips.py [--clips_root PATH]
    
Output:
    local_lists/sta_clips_all.txt - All found MP4 paths (one per line)
"""

import argparse
from pathlib import Path
from datetime import datetime


def scan_clips(clips_root: Path, output_file: Path) -> int:
    """
    Recursively scan for *.mp4 files and write paths to output file.
    
    Args:
        clips_root: Root directory to scan
        output_file: Output file path
        
    Returns:
        Number of clips found
    """
    print(f"Scanning for MP4 files in: {clips_root}")
    
    if not clips_root.exists():
        print(f"ERROR: Clips root does not exist: {clips_root}")
        return 0
    
    # Find all MP4 files
    mp4_files = sorted(clips_root.rglob("*.mp4"))
    
    print(f"Found {len(mp4_files)} MP4 files")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write paths to file (overwrite for idempotency)
    with open(output_file, 'w', encoding='utf-8') as f:
        for mp4_path in mp4_files:
            f.write(str(mp4_path.resolve()) + '\n')
    
    print(f"Wrote {len(mp4_files)} paths to: {output_file}")
    
    return len(mp4_files)


def main():
    parser = argparse.ArgumentParser(description="Scan for STA clip MP4 files")
    parser.add_argument(
        "--clips_root",
        type=str,
        default=r"I:\My Drive\ego4d_data\clip540s",
        help="Root directory containing MP4 clips"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for lists (default: ./local_lists)"
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    clips_root = Path(args.clips_root)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / "local_lists"
    
    output_file = output_dir / "sta_clips_all.txt"
    
    # Print header
    print("=" * 60)
    print("STA CLIPS SCANNER")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    print(f"Clips root: {clips_root}")
    print(f"Output file: {output_file}")
    print()
    
    # Scan clips
    count = scan_clips(clips_root, output_file)
    
    # Summary
    print()
    print("=" * 60)
    print(f"SCAN COMPLETE: {count} clips found")
    print("=" * 60)
    
    return 0 if count > 0 else 1


if __name__ == "__main__":
    exit(main())
