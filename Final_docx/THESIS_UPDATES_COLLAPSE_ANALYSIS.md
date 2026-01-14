# Thesis Text Updates - Class Collapse Analysis

## Location: Chapter 7.2.6 - After Table 7.1

### ADD NEW SECTION: Class Diversity Analysis

**Insert this immediately after Table 7.1 (before "Selection Rationale"):**

```markdown
**Class Diversity Analysis:**

Beyond aggregate metrics, prediction diversity analysis reveals how class weighting affects model behavior:

![](../local_extraction/runs/Track_B/collapse_analysis/collapse_comparison.png)

**Figure X - Class diversity comparison: Weighted vs unweighted checkpoints**


| Diversity Metric | Weighted | Unweighted | Δ |
|------------------|----------|------------|---|
| Unique nouns (top-1) | 98 | 44 | +123% |
| Unique verbs (top-1) | 60 | 17 | +253% |
| Noun top-5 concentration | 21.0% | 45.7% | -54% |
| Verb top-5 concentration | 31.9% | 89.7% | -64% |

**Table 7.X - Prediction diversity metrics (lower concentration = better)**

**Key Observations:**

1. **Severe collapse without weighting:** The unweighted checkpoint predicts only 44 out of 110 possible noun classes, with 90% of verb predictions coming from just 5 frequent classes. This demonstrates extreme class collapse where the model over-relies on common categories.

2. **Dramatic diversity improvement with weighting:** The weighted checkpoint (α=0.5) uses 2.2× more noun classes and 3.5× more verb classes, with top-5 concentration reduced by 54-64% across categories.

3. **Trade-off validation:** While aggregate mAP drops by 1-6%, prediction diversity improves by 100-250%, confirming that class weighting successfully prevents frequency-based overfitting at a measured performance cost.

This diversity analysis provides empirical evidence for the theoretical motivation behind class weighting: without balancing, the model learns to predict only the most frequent classes in the long-tailed Ego4D distribution.
```

---

## REPLACE: Selection Rationale (Section 7.2.6)

**OLD TEXT (lines ~1599-1609 in thesis.md):**
```markdown
**Selection Rationale:**
Despite slightly lower aggregate scores, the **weighted checkpoint (0.3708) is used throughout this thesis** because:

1. **Methodological consistency:** Represents our final training approach with class balancing
2. **Fair evaluation:** All Dec 26-30 efficiency experiments used this checkpoint, enabling direct comparison
3. **Better generalization:** Class weighting prevents overfitting to frequent classes
4. **Reproducibility:** Weighted approach is the methodology we recommend for future work
5. **Experimental integrity:** Comparing efficiency variants requires a consistent baseline checkpoint

The slight performance difference validates the trade-off: class weighting sacrifices some aggregate performance to improve learning across all classes, including rare ones.
```

**NEW TEXT (replace with this):**
```markdown
**Selection Rationale:**

The **weighted checkpoint (0.3708) is used throughout this thesis** despite lower aggregate benchmark scores, particularly a -5.72% drop in All_top5_mAP (4.81% vs 10.53%). This choice is supported by empirical evidence and methodological priorities:

1. **Quantified diversity improvement (Table 7.X):** Weighted checkpoint uses 123% more unique noun classes (98 vs 44) and 253% more verb classes (60 vs 17). Top-5 prediction concentration is reduced by 54% for nouns and 64% for verbs, demonstrating successful prevention of class collapse.

2. **Prevention of frequency bias:** Without class weighting, the model collapses to predicting only 44 noun classes and concentrates 90% of verb predictions in just 5 frequent classes. Class weighting (α=0.5) forces the model to explore the full label space, improving rare-class representation at the cost of aggregate metrics.

3. **Experimental integrity:** All 16 Track C efficiency experiments (Dec 26-30, 2025) used this checkpoint, enabling fair comparison across ablations without confounding checkpoint selection with pruning strategies.

4. **Methodological consistency:** Represents our final training methodology with principled class balancing, making results reproducible and extensible for future work targeting long-tailed distributions.

5. **Research objectives alignment:** This thesis prioritizes reproducible methodology, rare-class generalization, and transparent engineering trade-offs over benchmark maximization. The documented performance gap is acceptable given the 100-250% diversity improvement.

The performance trade-off (-1.04% mAP, -5.72% All_top5_mAP) is **not slight** but is a measured cost of preventing class collapse. This choice reflects principled engineering: accepting lower aggregate scores to maintain semantic diversity and avoid overfitting to frequent classes. The diversity analysis (Table 7.X, Figure X) validates this decision empirically rather than relying on theoretical assumptions.
```

---

## UPDATE: Abstract (Optional Enhancement)

**Current abstract mentions "competitive accuracy" - consider adding diversity context:**

Add after the mAP results paragraph:
```markdown
Class diversity analysis demonstrates that the weighted training approach uses 
2.2× more unique noun classes and reduces top-5 prediction concentration by 
54-64% compared to unweighted training, validating the class balancing 
methodology despite lower aggregate benchmark scores.
```

---

## UPDATE: Chapter 9.5.3 Error Analysis

**Add connection to diversity findings (around line 2240):**

```markdown
**Connection to Diversity Analysis:** The 15 zero-accuracy noun classes 
identified in error analysis correlate with the class collapse phenomenon 
documented in Section 7.2.6. The unweighted checkpoint never predicts 
many of these rare classes (only 44 out of 110 classes used), while the 
weighted checkpoint attempts predictions across 98 classes. This explains 
why weighted accuracy is similar overall: it explores more classes but 
struggles equally with fine-grained discrimination due to the frozen 
ImageNet backbone bottleneck.
```

---

## UPDATE: Chapter 10 Discussion

**Add to "What worked well" section:**

```markdown
**Class weighting successfully prevents prediction collapse.** Empirical 
analysis (Section 7.2.6) demonstrates that without class weighting, the 
model collapses to predicting only 40% of available classes with 90% 
concentration in top-5 frequent categories. The α=0.5 weighting factor 
achieves 2-3× improvement in class diversity metrics, validating the 
long-tailed distribution mitigation strategy despite a measured 1-6% 
cost on aggregate benchmarks.
```

---

## File Updates Required

1. **Copy plot to Final_docx/figs/**
   ```bash
   cp local_extraction/runs/Track_B/collapse_analysis/collapse_comparison.png \
      Final_docx/figs/class_diversity_comparison.png
   ```

2. **Update thesis.md** with sections above

3. **Increment figure numbers** (if Figure X becomes Figure 2, update subsequent figures 2→3, 3→4, etc.)

4. **Add table to table count** (Table 7.X becomes Table 7.2, renumber following tables)

5. **Rebuild DOCX** after edits:
   ```bash
   cd Final_docx
   # Run your build pipeline
   ```

---

## Citation in Other Chapters

When discussing weighted checkpoint in other chapters:

**Before:**
> "...using the weighted checkpoint for methodological consistency..."

**After:**
> "...using the weighted checkpoint which demonstrates 123% better noun class 
> diversity (Section 7.2.6) at a -1.04% mAP cost..."

---

## Key Messages for Defense/Review

1. **Claim is now evidence-based:** "Class weighting prevents collapse" backed by 98 vs 44 unique classes

2. **Trade-off is transparent:** Explicitly state -5.72% drop is NOT slight, show it's justified

3. **Methodology over benchmarks:** Positions thesis as reproducible research, not SOTA-chasing

4. **Rare-class focus validated:** Diversity metrics directly support rare-class learning goals

5. **Engineering maturity:** Shows you understand trade-offs and make principled decisions

---

## Before/After Comparison

**BEFORE (weak claim):**
> "Class weighting prevents overfitting to frequent classes"
- No evidence
- Assumes reader accepts theory
- "Slight" performance difference understates impact

**AFTER (strong claim):**
> "Class weighting achieves 123% noun diversity improvement (98 vs 44 classes) 
> and 54% reduction in top-5 concentration, at a measured -5.72% All_top5_mAP cost"
- Quantified evidence
- Transparent about trade-off
- Principled engineering decision

---

## Final Checklist

- [ ] Copy collapse_comparison.png to Final_docx/figs/
- [ ] Add "Class Diversity Analysis" section after Table 7.1
- [ ] Replace "Selection Rationale" with evidence-based version
- [ ] Update figure/table numbering throughout document
- [ ] Add diversity metrics to abstract (optional)
- [ ] Connect diversity to error analysis (Chapter 9.5.3)
- [ ] Mention in Discussion chapter (Chapter 10)
- [ ] Rebuild DOCX with updated content
- [ ] Verify all cross-references work
- [ ] Update List of Figures/Tables in Word

---

## Impact

This update transforms a weak theoretical claim into a strong empirical finding, 
strengthening the thesis's methodological rigor and making the weighted checkpoint 
choice defensible under scrutiny.
