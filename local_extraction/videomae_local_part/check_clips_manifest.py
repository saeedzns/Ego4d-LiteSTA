"""
Check and summarize the contents of the ego4d clips folder and manifest.
"""

import os
import csv
from pathlib import Path
from collections import Counter

# Paths
CLIPS_ROOT = Path(r"I:\My Drive\ego4d_data\clip540s")
MANIFEST_PATH = CLIPS_ROOT / "manifest.csv"


def check_manifest():
    """Read and summarize the manifest.csv file."""
    print("=" * 60)
    print("MANIFEST.CSV SUMMARY")
    print("=" * 60)
    
    if not MANIFEST_PATH.exists():
        print(f"❌ Manifest not found: {MANIFEST_PATH}")
        return None
    
    print(f"✓ Manifest found: {MANIFEST_PATH}")
    print(f"  File size: {MANIFEST_PATH.stat().st_size / 1024:.2f} KB")
    print()
    
    # Read manifest
    rows = []
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    print(f"Columns: {fieldnames}")
    print(f"Total rows: {len(rows)}")
    print()
    
    # Show first few rows
    print("First 5 rows:")
    print("-" * 60)
    for i, row in enumerate(rows[:5]):
        print(f"  [{i+1}] {row}")
    print()
    
    # Analyze columns if present
    if fieldnames:
        print("Column Statistics:")
        print("-" * 60)
        for col in fieldnames:
            values = [r.get(col, '') for r in rows]
            unique_count = len(set(values))
            print(f"  {col}: {unique_count} unique values")
            
            # Show sample values
            sample = list(set(values))[:5]
            print(f"    Samples: {sample}")
        print()
    
    return rows


def check_clips_folder():
    """Summarize the contents of the clips folder."""
    print("=" * 60)
    print("CLIPS FOLDER SUMMARY")
    print("=" * 60)
    
    if not CLIPS_ROOT.exists():
        print(f"❌ Clips folder not found: {CLIPS_ROOT}")
        return
    
    print(f"✓ Clips folder found: {CLIPS_ROOT}")
    print()
    
    # Count files by extension
    extensions = Counter()
    total_size = 0
    file_count = 0
    mp4_files = []
    
    for item in CLIPS_ROOT.iterdir():
        if item.is_file():
            ext = item.suffix.lower()
            extensions[ext] += 1
            total_size += item.stat().st_size
            file_count += 1
            
            if ext == '.mp4':
                mp4_files.append(item.name)
    
    print(f"Total files: {file_count}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print()
    
    print("Files by extension:")
    for ext, count in extensions.most_common():
        print(f"  {ext or '(no extension)'}: {count} files")
    print()
    
    # MP4 analysis
    if mp4_files:
        print(f"MP4 files: {len(mp4_files)}")
        print("Sample MP4 filenames:")
        for f in sorted(mp4_files)[:10]:
            print(f"  - {f}")
        if len(mp4_files) > 10:
            print(f"  ... and {len(mp4_files) - 10} more")
        print()
        
        # Analyze naming pattern
        print("Filename pattern analysis:")
        first_mp4 = mp4_files[0]
        print(f"  Example: {first_mp4}")
        parts = first_mp4.replace('.mp4', '').split('_')
        print(f"  Parts (split by '_'): {parts}")
        print(f"  Number of parts: {len(parts)}")
    
    return mp4_files


def cross_check(manifest_rows, mp4_files):
    """Cross-check manifest entries against actual files."""
    print("=" * 60)
    print("CROSS-CHECK: MANIFEST vs FILES")
    print("=" * 60)
    
    if not manifest_rows or not mp4_files:
        print("Cannot cross-check - missing data")
        return
    
    mp4_set = set(mp4_files)
    
    # Use exported_clip_uid column (which matches filenames without .mp4)
    matched_col = 'exported_clip_uid'
    manifest_files = set()
    
    if matched_col in manifest_rows[0]:
        print(f"Using column '{matched_col}' for filenames")
        for row in manifest_rows:
            val = row.get(matched_col, '')
            # Add .mp4 extension to match actual filenames
            if val and not val.endswith('.mp4'):
                val = val + '.mp4'
            if val:
                manifest_files.add(val)
    else:
        print(f"Column '{matched_col}' not found in manifest")
        print("Available columns:", list(manifest_rows[0].keys()))
        return
    
    # Compare
    in_manifest_only = manifest_files - mp4_set
    in_folder_only = mp4_set - manifest_files
    in_both = manifest_files & mp4_set
    
    print(f"\nFiles in both manifest and folder: {len(in_both)}")
    print(f"Files in manifest only (missing from folder): {len(in_manifest_only)}")
    print(f"Files in folder only (not in manifest): {len(in_folder_only)}")
    
    if in_manifest_only:
        print("\nSample files in manifest but missing from folder:")
        for f in list(in_manifest_only)[:5]:
            print(f"  - {f}")
    
    if in_folder_only:
        print("\nSample files in folder but not in manifest:")
        for f in list(in_folder_only)[:5]:
            print(f"  - {f}")


def main():
    print("\n" + "=" * 60)
    print("EGO4D CLIPS FOLDER & MANIFEST CHECK")
    print(f"Path: {CLIPS_ROOT}")
    print("=" * 60 + "\n")
    
    # Check manifest
    manifest_rows = check_manifest()
    
    # Check clips folder
    mp4_files = check_clips_folder()
    
    # Cross-check
    if manifest_rows and mp4_files:
        cross_check(manifest_rows, mp4_files)
    
    print("\n" + "=" * 60)
    print("CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
