# Add Table of Contents in Word (30 seconds)

## Steps:

1. **Open:** `build/thesis.docx` in Word
2. **Click** at the beginning of the document (before "Chapter 1")
3. **Press Enter** a few times to create space
4. **Go to:** References tab → Table of Contents
5. **Choose:** "Automatic Table 1" or "Automatic Table 2"
6. **Done!** TOC is generated from all headings

---

## To Update TOC Later:

- **Click anywhere in the TOC**
- **Press F9** (or right-click → Update Field)
- **Choose:** "Update entire table"

---

## Why Manual TOC?

Pandoc's automatic `--toc` flag creates field references that cause update prompts and duplicate numbering issues. Word's native TOC generation is cleaner and more reliable for DOCX output.

---

## TOC Settings (Optional):

In References → Table of Contents → Custom Table of Contents:
- **Show levels:** 3 (to show Chapter, Section, Subsection)
- **Show page numbers:** Yes
- **Right align page numbers:** Yes
- **Tab leader:** Dotted line

