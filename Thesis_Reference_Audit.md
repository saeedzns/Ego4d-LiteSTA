# Thesis Reference & Citation Audit (from `thesis_fixed.md` + provided notes)

Generated: 2026-01-10 (Europe/Rome)  
Style target: **IEEE numeric** (recommended for CS theses)

This file consolidates **all methods/papers/repos explicitly mentioned** in the attached material and your notes, and provides **ready-to-paste citations where I can confidently specify complete metadata**.  
Where complete metadata (especially full author lists) could not be validated within tool constraints, the entry is marked **NEEDS VERIFICATION** and includes exactly what to look up.

---

## Action required (your checklist)

Replace placeholder entries with full citations including:
- Authors (full names)
- Paper title
- Venue (conference/journal)
- Year
- DOI or arXiv ID
- Page numbers where applicable

Verification steps:
1. Search each method name + **“paper”** on Google Scholar.
2. Cross-reference with **Ego4D workshop / challenge proceedings** and the **Ego4D Forecasting** benchmark docs/repo.
3. Check **arxiv.org** for preprints (prefer arXiv IDs + arXiv DOI `10.48550/arXiv.<id>` when no publisher DOI exists).
4. Verify all DOIs resolve.
5. Keep formatting consistent (IEEE / ACM / APA). This file uses **IEEE-like** formatting.

---

## Required citations (must appear in your thesis)

### [R1] Ego4D dataset and benchmark paper (NEEDS VERIFICATION: full author list + CVPR pages)
- **Title:** *Ego4D: Around the World in 3,000 Hours of Egocentric Video*  
- **Venue/Year:** CVPR 2022 (as cited in the Ego4D project documentation) citeturn24view0  
- **arXiv:** 2110.07058 (arXiv DOI: `10.48550/arXiv.2110.07058`) (verify on arXiv)  
- **Authors:** Kristen Grauman **et al.** (**replace with full consortium author list from the paper**)

### [R2] VideoMAE (NEEDS VERIFICATION: full author list + NeurIPS pages)
- **Title:** *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*  
- **Venue/Year:** NeurIPS 2022 (per paper header; verify proceedings)  
- **arXiv:** 2203.12602 (arXiv DOI: `10.48550/arXiv.2203.12602`)  
- **Authors:** NEEDS VERIFICATION (pull from arXiv page)

### [R3] YOLOv8 (Ultralytics) (NEEDS VERIFICATION: exact version you used)
- **Software/Docs:** Ultralytics. *YOLOv8 Documentation / Model Card.*  
- **Version/date:** Use your exact package version (e.g., `ultralytics==8.x.y`) + accessed date.  
- **Release evidence:** Ultralytics docs show YOLOv8 release timing (verify exact tag/date). citeturn23search1  
- **Authors:** Ultralytics (organization)

### [R4] CLIP (ready to paste)
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning Transferable Visual Models From Natural Language Supervision,” in *Proc. ICML*, 2021. arXiv:2103.00020 (DOI: 10.48550/arXiv.2103.00020). citeturn4view0

### [R5] StillFast (Ego4D baseline) (NEEDS VERIFICATION: full author list + venue pages)
- **Title:** *StillFast: An End-to-End Approach for Short-Term Object Interaction Anticipation*  
- **arXiv:** 2304.03959 (DOI: `10.48550/arXiv.2304.03959`) (verify on arXiv)  
- **Context:** Listed among short-term anticipation resources in the Ego4D Forecasting benchmark repo/docs. citeturn24view0turn24view1  
- **Authors/Venue/Year:** NEEDS VERIFICATION (pull from arXiv + any workshop proceedings)

### [R6] GANO v2 (NEEDS VERIFICATION: confirm it is v2 + authors/venue)
- **Likely paper:** “Guided Attention for Next-Active Object Detection” (arXiv likely).  
- **Source hint:** Mentioned as a prior STA method in SOIA-DOD related-work text (your notes).  
- **Action:** Search “GANOv2 Ego4D STA arXiv” and verify exact title, author list, and whether a v2 version exists.

### [R7] STAformer / AFF-ttention (NEEDS VERIFICATION: authors + arXiv ID)
- **Title (from your notes):** *ZARRIO @ Ego4D Short Term Object Interaction Anticipation Challenge: Leveraging Affordances and Attention-based models for STA*  
- **arXiv:** 2407.04369 (as stated in your notes; verify on arXiv)  
- **Code:** `https://github.com/lmur98/AFFttention` (verify permanent commit/tag)  
- **Authors/Venue/Year:** NEEDS VERIFICATION (pull from arXiv)

### [R8] FRCNN+SF baseline (recommended citation set)
This “baseline” is best cited as **its component architectures**, unless you find a specific Ego4D report explicitly naming “FRCNN+SF”:
- S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” in *NeurIPS*, 2015. arXiv:1506.01497 (DOI: 10.48550/arXiv.1506.01497).  
- C. Feichtenhofer, “SlowFast Networks for Video Recognition,” in *ICCV*, 2019. arXiv:1812.03982 (DOI: 10.48550/arXiv.1812.03982).  
- **And** cite the Ego4D Forecasting benchmark docs/repo for where this baseline is defined. citeturn24view0turn24view1

### [R9] Transformer architecture (ready to paste)
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention Is All You Need,” in *NeurIPS*, 2017. arXiv:1706.03762 (DOI: 10.48550/arXiv.1706.03762). citeturn26view2

### [R10] Standard mAP metrics (ready to paste)
- M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The Pascal Visual Object Classes (VOC) Challenge,” *Int. J. Comput. Vis.*, vol. 88, no. 2, pp. 303–338, 2010. DOI: 10.1007/s11263-009-0275-4.  
- T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft COCO: Common Objects in Context,” in *ECCV*, 2014, pp. 740–755. DOI: 10.1007/978-3-319-10602-1_48.

---

## Ego4D STA task & baselines (directly referenced in your notes)

### Ego4D Forecasting benchmark docs and repo
- Ego4D, “Forecasting Benchmark (Ego4D),” documentation page. citeturn24view0  
- EGO4D, “forecasting” benchmark repository (GitHub). citeturn24view2  
- EGO4D, “SHORT_TERM_ANTICIPATION.md” (dataset/annotation instructions). citeturn24view1  

### SOIA-DOD (NEEDS VERIFICATION: authors from arXiv)
- **Title:** *Short-term Object Interaction Anticipation with Decomposition-oriented Detection (SOIA-DOD)*  
- **arXiv:** 2407.05713 (as stated in your notes; verify on arXiv)  
- **Code:** `https://github.com/KeenyJin/SOIA-DOD` (verify release/commit)

---

## Efficient backbones & pretraining (from your notes)

### MViTv2 (NEEDS VERIFICATION: authors/venue/pages)
- **Title:** *MViTv2: Improved Multiscale Vision Transformers for Classification and Detection*  
- **arXiv:** 2112.01526 (verify on arXiv)  
- **Code:** `https://github.com/facebookresearch/mvit`

### VideoMAE (see [R2])

---

## Token pruning / streaming efficiency (from your notes)

### PruneVid (NEEDS VERIFICATION: arXiv ID + authors)
- **Title:** *PruneVid: Visual Token Pruning for Efficient Video Large Language Models*  
- **arXiv:** Your notes imply an arXiv preprint (verify exact ID on arXiv).  
- **Action:** search “PruneVid Visual Token Pruning arXiv” and paste arXiv ID + full author list.

### Rollout-Guided Token Pruning (ICIP 2025) (NEEDS VERIFICATION: authors)
- **Title:** *Rollout-Guided Token Pruning for Efficient Video Understanding*  
- **Venue/Year:** IEEE ICIP 2025 (per your pasted header)  
- **DOI:** 10.1109/ICIP55913.2025.11084634  
- **Code:** `https://github.com/RGTPdyn/RGTP`

### EgoPrune (NEEDS VERIFICATION: authors/arXiv)
- **Title:** *EgoPrune: Efficient Token Pruning for Egomotion Video Reasoning in Embodied Agent*  
- **arXiv:** 2507.15428 (as stated in your notes; verify on arXiv)

---

## Hand–object anticipation / affordances (from your notes)

### PEAR (NEEDS VERIFICATION: authors/venue; arXiv in notes)
- **Title:** *PEAR: Phrase-Based Hand-Object Interaction Anticipation*  
- **arXiv:** 2407.21510 (as stated in your notes; verify on arXiv)

### Fine-grained affordance annotation (NEEDS VERIFICATION: authors list; arXiv in notes)
- **Title:** *Fine-grained Affordance Annotation for Egocentric Hand-Object Interaction Videos*  
- **arXiv:** 2302.03292 citeturn9view2

---

## Ego–Exo transfer & surveys (from your notes)

### Ego–Exo survey (NEEDS VERIFICATION: authors/arXiv)
- **Title:** *Egocentric and Exocentric Methods: A Short Survey*  
- **arXiv:** 2410.20621 (as stated in your notes; verify on arXiv)

### Synchronization is All You Need (NEEDS VERIFICATION: authors; arXiv in notes)
- **Title:** *Synchronization is All You Need: Exocentric-to-Egocentric Transfer for Temporal Action Segmentation with Unlabeled Synchronized Video Pairs*  
- **arXiv:** 2312.02638 (as stated in your notes; verify on arXiv)

### Ego-Only (NEEDS VERIFICATION: authors; arXiv in notes)
- **Title:** *Ego-Only: Egocentric Action Detection without Exocentric Transferring*  
- **arXiv:** 2301.01380 (as stated in your notes; verify on arXiv)

### EgoTransfer (NEEDS VERIFICATION: exact venue/year/DOI)
- **Title:** *EgoTransfer: Transferring Motion Across Egocentric and Exocentric Domains using Deep Neural Networks*  
- **Action:** confirm venue/year (likely CVPR-era) and obtain DOI/arXiv.

---

## Bonus hubs & lists (from your notes)

- EgoVis Workshop @ CVPR 2024: workshop page (use as a web reference if cited).  
- “Awesome Egocentric Action Understanding” GitHub list: `https://github.com/Lyman-Smoker/Awesome_Ego_Action`

---

## Practical “citation hygiene” notes

- For **repos/software**, cite: organization/author, project name, version tag/commit hash, release date (if available), and accessed date.
- For **arXiv**, cite both **arXiv ID** and **arXiv DOI** (`10.48550/arXiv.<id>`).
- For **challenge baselines**, cite the **challenge technical report** (if any) + the **official benchmark repo/docs**.

