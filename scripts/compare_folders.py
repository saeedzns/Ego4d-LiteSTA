"""
Compare two folders and find missing/different files.
"""
import os
from pathlib import Path

# Folders to compare
FOLDER_A = r"H:\My Drive\ego4d_data\v2\clips_540\clips"
FOLDER_B = r"I:\My Drive\ego4d_data\clip540s"

def get_files(folder: str) -> set:
    """Get set of filenames in a folder."""
    if not os.path.exists(folder):
        print(f"❌ Folder does not exist: {folder}")
        return set()
    
    files = set()
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)):
            files.add(f)
    return files

def compare_folders(folder_a: str, folder_b: str):
    """Compare two folders and report differences."""
    print("=" * 60)
    print("FOLDER COMPARISON")
    print("=" * 60)
    print(f"\nFolder A: {folder_a}")
    print(f"Folder B: {folder_b}\n")
    
    files_a = get_files(folder_a)
    files_b = get_files(folder_b)
    
    if not files_a and not files_b:
        print("❌ Both folders are empty or don't exist!")
        return
    
    print(f"📁 Folder A: {len(files_a)} files")
    print(f"📁 Folder B: {len(files_b)} files")
    print()
    
    # Files only in A (missing from B)
    only_in_a = files_a - files_b
    # Files only in B (missing from A)
    only_in_b = files_b - files_a
    # Files in both
    in_both = files_a & files_b
    
    print(f"✅ Files in both: {len(in_both)}")
    print(f"🔴 Only in Folder A (missing from B): {len(only_in_a)}")
    print(f"🔵 Only in Folder B (missing from A): {len(only_in_b)}")
    print()
    
    # Show missing files
    if only_in_a:
        print("-" * 60)
        print("🔴 FILES ONLY IN FOLDER A (missing from B):")
        print("-" * 60)
        for f in sorted(only_in_a)[:50]:  # Show first 50
            print(f"  {f}")
        if len(only_in_a) > 50:
            print(f"  ... and {len(only_in_a) - 50} more")
        print()
    
    if only_in_b:
        print("-" * 60)
        print("🔵 FILES ONLY IN FOLDER B (missing from A):")
        print("-" * 60)
        for f in sorted(only_in_b)[:50]:  # Show first 50
            print(f"  {f}")
        if len(only_in_b) > 50:
            print(f"  ... and {len(only_in_b) - 50} more")
        print()
    
    # Save full lists to files
    if only_in_a or only_in_b:
        output_dir = Path(__file__).parent
        
        if only_in_a:
            with open(output_dir / "missing_from_B.txt", "w") as f:
                f.write(f"# Files in A but missing from B\n")
                f.write(f"# A: {folder_a}\n")
                f.write(f"# B: {folder_b}\n\n")
                for filename in sorted(only_in_a):
                    f.write(f"{filename}\n")
            print(f"📄 Saved full list: {output_dir / 'missing_from_B.txt'}")
        
        if only_in_b:
            with open(output_dir / "missing_from_A.txt", "w") as f:
                f.write(f"# Files in B but missing from A\n")
                f.write(f"# A: {folder_a}\n")
                f.write(f"# B: {folder_b}\n\n")
                for filename in sorted(only_in_b):
                    f.write(f"{filename}\n")
            print(f"📄 Saved full list: {output_dir / 'missing_from_A.txt'}")
    
    if not only_in_a and not only_in_b:
        print("✅ FOLDERS ARE IDENTICAL! All files match.")

if __name__ == "__main__":
    compare_folders(FOLDER_A, FOLDER_B)
