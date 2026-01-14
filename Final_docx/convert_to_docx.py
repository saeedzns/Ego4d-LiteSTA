#!/usr/bin/env python3
"""
Convert thesis_fixed.md to Word document (.docx)

Requirements:
    pip install pypandoc

Usage:
    python convert_to_docx.py
"""

import os
import sys
from pathlib import Path

def check_pandoc():
    """Check if pandoc is installed"""
    try:
        import pypandoc
        pandoc_version = pypandoc.get_pandoc_version()
        print(f"✓ Pandoc found (version {pandoc_version})")
        return True
    except (ImportError, OSError):
        print("✗ Pandoc not found")
        print("\nInstall instructions:")
        print("  Windows: choco install pandoc  (or download from https://pandoc.org)")
        print("  macOS:   brew install pandoc")
        print("  Linux:   sudo apt install pandoc")
        print("\nThen install Python wrapper:")
        print("  pip install pypandoc")
        return False

def convert_thesis():
    """Convert markdown thesis to DOCX"""
    import pypandoc
    
    # Paths
    input_file = Path("Final_docx/thesis_fixed.md")
    output_file = Path("Final_docx/thesis_final.docx")
    
    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        return False
    
    print(f"\n📄 Converting: {input_file}")
    print(f"📝 Output: {output_file}")
    
    # Conversion options
    extra_args = [
        '--number-sections',           # Number chapters/sections
        '--toc',                        # Table of contents
        '--toc-depth=3',               # Include subsections in TOC
        '--standalone',                # Complete document
        '--highlight-style=tango',     # Code syntax highlighting
    ]
    
    # Check if reference template exists
    template = Path("Final_docx/template.docx")
    if template.exists():
        extra_args.append(f'--reference-doc={template}')
        print(f"✓ Using template: {template}")
    else:
        print("ℹ No template found, using default formatting")
    
    try:
        # Convert
        print("\n⏳ Converting (this may take 30-60 seconds)...")
        pypandoc.convert_file(
            str(input_file),
            'docx',
            outputfile=str(output_file),
            extra_args=extra_args
        )
        
        # Check output
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"\n✓ Conversion successful!")
            print(f"✓ Output size: {size_mb:.2f} MB")
            print(f"\n📁 Open: {output_file.absolute()}")
            
            # Post-conversion checklist
            print("\n📋 Manual fixes needed in Word:")
            print("  1. Verify all 20 figures loaded from figs/ folder")
            print("  2. Adjust table column widths for readability")
            print("  3. Add page numbers (Insert → Page Number)")
            print("  4. Add headers/footers (Insert → Header/Footer)")
            print("  5. Set line spacing (Home → Line Spacing → 1.5 or Double)")
            print("  6. Format title page with author, date, institution")
            print("  7. Check LaTeX equations rendered correctly")
            print("  8. Review figure/table captions formatting")
            print("  9. Set page margins (Layout → Margins → Normal or Custom)")
            print("  10. Generate final PDF (File → Save As → PDF)")
            
            return True
        else:
            print("✗ Conversion failed - output file not created")
            return False
            
    except Exception as e:
        print(f"\n✗ Conversion error: {e}")
        return False

def main():
    print("=" * 60)
    print("Ego4D-LiteSTA Thesis Converter")
    print("=" * 60)
    
    # Check pandoc installation
    if not check_pandoc():
        return 1
    
    # Convert thesis
    if convert_thesis():
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
