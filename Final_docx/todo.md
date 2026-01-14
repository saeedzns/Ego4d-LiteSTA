# Thesis MD → DOCX Workflow (VS Code + Pandoc) — To-Do List

## 0) Folder structure (do this first)
- [ ] Create folders:
  - [ ] `build/` (generated output)
  - [ ] `figs/` or `images/` (all figures)
  - [ ] `.vscode/` (VS Code tasks)
- [ ] Ensure your thesis source is:
  - [ ] Single file: `thesis.md`
  - [ ] OR multi-file: a `main.md` that includes/links chapters consistently (keep a single “entry” file for Pandoc).

---

## 1) Install local tools (no websites)
- [ ] Install **Pandoc** (system install).
- [ ] Verify Pandoc works:
  - [ ] Run: `pandoc --version`

---

## 2) Create a Word style template (reference DOCX)
- [ ] Generate default reference doc:
  - [ ] Run:
    - [ ] `pandoc -o custom-reference.docx --print-default-data-file reference.docx`
- [ ] Open `custom-reference.docx` in Word and edit **styles only**:
  - [ ] Normal (body font, size, line spacing)
  - [ ] Heading 1/2/3 (fonts, spacing before/after)
  - [ ] Caption (figure/table caption style)
  - [ ] Bibliography (if used)
  - [ ] Code/Preformatted style (if you use code blocks)
- [ ] Save `custom-reference.docx` in the thesis root folder.

---

## 3) Standardize Markdown for images and math (fix fidelity)
### Images
- [ ] Make sure image paths are relative and exist:
  - [ ] `![](figs/your_image.png)`
- [ ] Add width attributes where needed:
  - [ ] `![](figs/plot.png){width=90%}`
- [ ] Move all images into `figs/` or `images/` so `--resource-path` can find them.

### Math / formulas
- [ ] Inline math uses `$...$`
- [ ] Block math uses `$$...$$`
- [ ] Avoid heavy custom LaTeX macros if Word equations look wrong.
- [ ] If you must use macros, consider expanding them manually.

---

## 4) Build DOCX locally with Pandoc (baseline command)
- [ ] Create `build/` folder (if not already).
- [ ] Run a baseline build command (Windows PowerShell):
  - [ ] 
    ```powershell
    pandoc thesis.md `
      -o build/thesis.docx `
      --reference-doc=custom-reference.docx `
      --resource-path=.:figs:images `
      --standalone
    ```
- [ ] Check output in Word:
  - [ ] Fonts, headings, spacing
  - [ ] Images present + sized correctly
  - [ ] Equations render as Word equations

---

## 5) If you have citations (optional)
- [ ] Ensure you have:
  - [ ] `refs.bib` (or other bibliography file)
  - [ ] `ieee.csl` (or your required style)
- [ ] Build with citeproc:
  - [ ] 
    ```powershell
    pandoc thesis.md `
      -o build/thesis.docx `
      --reference-doc=custom-reference.docx `
      --resource-path=.:figs:images `
      --citeproc `
      --bibliography=refs.bib `
      --csl=ieee.csl
    ```

---

## 6) Add a VS Code build task (one-command build inside VS Code)
- [ ] Create `.vscode/tasks.json`
- [ ] Paste (edit filenames if needed):
  - [ ] 
    ```json
    {
      "version": "2.0.0",
      "tasks": [
        {
          "label": "Build thesis.docx (pandoc)",
          "type": "shell",
          "command": "pandoc",
          "args": [
            "thesis.md",
            "-o", "build/thesis.docx",
            "--reference-doc=custom-reference.docx",
            "--resource-path=.:figs:images",
            "--standalone"
          ],
          "problemMatcher": []
        }
      ]
    }
    ```

---

## 7) Auto-build DOCX on save (fast iteration loop)
- [ ] Install ONE extension (choose one):
  - [ ] “Trigger Task on Save” (preferred if it runs VS Code tasks)
  - [ ] OR “Run On Save”
- [ ] Configure it to run:
  - [ ] Task: `Build thesis.docx (pandoc)`
  - [ ] Trigger: saving `thesis.md` (and chapter files if multi-file)

---

## 8) Preview DOCX inside VS Code (optional but recommended)
- [ ] Install ONE DOCX preview extension:
  - [ ] “Office Viewer” OR “Docx Renderer”
- [ ] Open `build/thesis.docx` in VS Code to quickly spot layout issues.

---

## 9) Debug checklist (when output still looks wrong)
### Styles/fonts wrong
- [ ] Confirm build includes:
  - [ ] `--reference-doc=custom-reference.docx`
- [ ] Fix styles in `custom-reference.docx` (not in Markdown), rebuild.

### Images missing
- [ ] Confirm `--resource-path=.:figs:images`
- [ ] Confirm image paths in Markdown match actual folders.
- [ ] Avoid absolute paths.

### Images too big/small
- [ ] Add width attributes:
  - [ ] `{width=60%}` or `{width=90%}`

### Equations look broken
- [ ] Simplify LaTeX (remove custom macros).
- [ ] Convert `\[ ... \]` to `$$ ... $$` if you use that style.
- [ ] Check for unsupported LaTeX packages/commands (Word output won’t support all).

---

## 10) How Copilot (AI agent) should help (copy/paste tasks)
- [ ] Ask Copilot to:
  - [ ] Add `{width=...}` to all figures matching a pattern.
  - [ ] Normalize all math blocks to `$$...$$`.
  - [ ] Find and replace unsupported LaTeX macros with expanded versions.
  - [ ] Generate/adjust `.vscode/tasks.json` for multi-file builds.
  - [ ] Refactor Markdown headings to consistent levels (`#`, `##`, `###`).
  - [ ] Create a repeatable command for your exact file layout.

---

## 11) Done definition (final acceptance)
- [ ] `build/thesis.docx` matches required formatting:
  - [ ] Correct fonts and line spacing
  - [ ] Correct headings and TOC-ready structure
  - [ ] All figures present and properly sized
  - [ ] Equations readable and correct
  - [ ] References/citations correct (if used)
- [ ] One-save workflow works:
  - [ ] Save MD → DOCX auto-regenerates → preview quickly
