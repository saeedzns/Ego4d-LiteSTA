# TODO — Auto-insert generated PNG figures into `thesis_fixed.md` (filename-driven)

## Global rules (agent behavior)
- Read all `.png` files in `figs/incoming/` (original names are your “topic names”).
- For each image:
  1) Create a sanitized copy in `figs/` (safe path for Markdown/Pandoc):
     - lowercase  
     - spaces → `_`  
     - remove `()[],:` and duplicate `_`  
     - keep `.png`
  2) Route the figure to a thesis section using the mapping below (match on filename keywords).
  3) Insert or replace in `thesis_fixed.md` using the specified **anchor heading**.
  4) Use relative paths: `figs/<sanitized_name>.png`
- Replacement policy:
  - If the target section already has a figure serving the same purpose, **replace the existing figure file** (preferred: preserves numbering/captions).
  - Otherwise, **insert a new figure block** right after the anchor heading (and mark caption as NEW so you can renumber later).

## Figure block template
Use exactly this block when **INSERTING** a new figure:

```md
![<ALT TEXT>](figs/<sanitized_filename>.png)

<!-- TODO: renumber figure index if you keep manual Figure numbers -->
**Figure (NEW) - <Short caption>.**
```

When **REPLACING** an existing figure:
- Do NOT edit the thesis text.
- Copy/overwrite the file in `figs/` so the existing Markdown image link stays valid.

---

## 1) Big picture pipeline diagram (Track A/B/C overview) (png name: "Big picture pipeline diagram (Track A/B/C overview).png")
Insert location:
- Chapter 4 start (system-level overview).

Find (anchor in thesis):
- `# Chapter 4: System Overview (Tracks A / B / C)`

Action:
- INSERT new figure block immediately after the anchor heading.

ALT + caption:
- ALT: `Ego4D-LiteSTA three-track pipeline overview`
- Caption: `Three-track STA pipeline: proposals, fusion head, pruning knob`

---

## 2) Data/manifest flow diagram (reproducibility pipeline) (png name: "Data/manifest flow diagram (reproducibility pipeline).png")
Insert location:
- Section 3.4, because it formally defines manifests and label alignment.

Find (anchor in thesis):
- `## 3.4 Data Manifests and Label Alignment Decisions`

Action:
- INSERT new figure block immediately after the anchor heading (before 3.4.1).

ALT + caption:
- ALT: `Manifest-driven workflow from clips to metrics`
- Caption: `Manifest-driven data and artifact flow across Track A/B/C`

---

## 3) Evaluation protocol diagram (Top-5 scoring and metric computation) (png name: "Evaluation protocol diagram (Top-5 scoring and metric computation).png")
Insert location:
- Section 7.3 (evaluation protocol).

Find (anchor in thesis):
- `## 7.3 Evaluation Protocol`

Action:
- INSERT new figure block immediately after the anchor heading (before 7.3.1).

ALT + caption:
- ALT: `Top-5 evaluation protocol for candidate ranking`
- Caption: `Top-5 ranking protocol and metric computation steps`

---

## 4) Track B architecture schematic (FGTP + dual cross-attention fusion) (png name: "Track B architecture schematic (FGTP + dual cross-attention fusion).png")
Insert location:
- Section 5.2 (Track B methodology).

Find (anchor in thesis):
- `## 5.2 Track B: Tokenization, Fusion (FGTP + Dual Cross-Attention), and Heads`

Action:
- INSERT new figure block immediately after the anchor heading (before 5.2.1).

ALT + caption:
- ALT: `Track B architecture: FGTP and dual cross-attention fusion`
- Caption: `Track B fusion head: FGTP + dual cross-attention + heads`

---

## 5) Token pruning visualization (rollout-guided) (png name: "Token pruning visualization (rollout-guided).png")
Insert location:
- Section 5.3 (Track C methodology).

Find (anchor in thesis):
- `## 5.3 Track C: Rollout-Guided Token Pruning (RGTP)`

Action:
- INSERT new figure block immediately after the anchor heading (before 5.3.1).

ALT + caption:
- ALT: `Rollout-guided token pruning concept`
- Caption: `Track C rollout-guided token pruning (training-free efficiency knob)`

---

## 6) Class imbalance + collapse vs diversity visuals (png name: "Class imbalance + collapse vs diversity visuals.png")
Insert location:
- Section 7.2.5 (class weighting and training methodology).

Find (anchor in thesis):
- `### 7.2.5 Class weighting and training methodology`

Action (preferred: REPLACE to preserve numbering):
- Replace the existing diversity figure by overwriting this file:
  - `figs/class_diversity_comparison.png`
- If your new infographic also includes the “weighted vs unweighted performance” table/chart visually,
  optionally also overwrite:
  - `figs/weighted_vs_unweighted.png`

Fallback action (if you do NOT want replacement):
- INSERT new figure block directly after the paragraph that introduces “Class Diversity Analysis:”.

ALT + caption (only if inserting):
- ALT: `Class imbalance and prediction collapse: weighted vs unweighted`
- Caption: `Class imbalance and diversity: weighting prevents prediction collapse`

---

## 7) Backbone comparison (ResNet18 ImageNet vs VideoMAE ego-pretrained) (png name: "Backbone comparison (ResNet18 ImageNet vs VideoMAE ego-pretrained).png")
Insert location:
- Section 7.2.8 (backbone selection).

Find (anchor in thesis):
- `### 7.2.8 Backbone selection: Exo-Transfer vs Ego-Pretrained`

Action (preferred: REPLACE to preserve numbering):
- If your infographic focuses on performance comparison:
  - overwrite: `figs/backbone_metrics_comparison.png`
- If your infographic also includes architecture/pretraining distribution:
  - overwrite: `figs/backbone_pretraining.png` as well.

Fallback action (if you do NOT want replacement):
- INSERT new figure block right after the line:
  - `**Performance Comparison:**`

ALT + caption (only if inserting):
- ALT: `Backbone comparison: ResNet18 exo-transfer vs VideoMAE ego-pretrained`
- Caption: `Backbone comparison: spatial exo-transfer vs temporal ego-pretraining`

---

## 8) Qualitative prediction examples grid (success + failures) (png name: "Qualitative prediction examples grid (success + failures).png")
Insert location:
- Chapter 9.5 (qualitative galleries) as a single “overview” grid before individual examples.

Find (anchor in thesis):
- `## 9.5 Error Analysis and Qualitative Galleries`

Action:
- INSERT new figure block immediately after the anchor heading (before 9.5.1).

ALT + caption:
- ALT: `Qualitative STA results: success and failure overview grid`
- Caption: `Qualitative overview: common successes and failure modes`

---

## Post-run checks (agent must do)
- Verify all inserted `figs/<...>.png` paths exist on disk.
- Verify no duplicate insertions:
  - If the same topic is re-run, prefer REPLACE behavior instead of adding another copy.
- Run a Markdown preview check:
  - images render
  - captions stay directly under images
  - headings remain unchanged

## Determinism option (recommended)
- Use **exact heading matches** for all anchors above.
- If an anchor heading is missing, **fail loudly** (do not guess insertion locations).
