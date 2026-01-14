# References From Your Provided List (Used in `Final_docx/thesis_fixed.md`)

This file maps the references you pasted in chat to where they are used in your thesis (`Final_docx/thesis_fixed.md`) and provides **fill‑in‑ready citations** (authors, title, venue, year, DOI/arXiv, and URLs).

Sources for citation metadata:
- arXiv API (titles, authors, dates, comments, DOIs when present)
- Crossref API (DOIs/venues/years for items registered with Crossref)
- Local workspace artifacts (YOLO training args + pinned `ultralytics==...` versions in notebooks)

Items still marked **VERIFY** either lack a stable paper record (e.g., software) or were not resolvable from the sources above.

---

## A) Ego4D STA task & challenge baselines

### A1) Ego4D Forecasting benchmark + STA task definition (website + repo)
- **Used in thesis:** Yes — Section **2.1** (`Final_docx/thesis_fixed.md:266`)
- **Links:**
  - https://ego4d-data.org/docs/benchmarks/forecasting/
  - https://github.com/EGO4D/forecasting
- **Draft citation:**
  - Ego4D Consortium. *Ego4D Forecasting Benchmark Documentation (including Short‑Term Anticipation / STA).* https://ego4d-data.org/docs/benchmarks/forecasting/ (accessed 2026‑01‑10).
  - Ego4D Consortium. *EGO4D/forecasting (benchmark code and evaluation).* https://github.com/EGO4D/forecasting (accessed 2026‑01‑10).

### A2) Ego4D dataset paper (task framing + benchmark suite)
- **Used in thesis:** Yes — Section **2.1.2** (`Final_docx/thesis_fixed.md:305`) notes “3,670 hours… 931… 74…”
- **Link:** https://ego4d-data.org/
- **Full citation:**
  - Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhongcong Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Cartillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli, Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Abrham Gebreselasie, Cristina Gonzalez, James Hillis, Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, Jachym Kolar, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li, Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Modhugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will Price, Paola Ruiz Puentes, Merey Ramazanova, Leda Sari, Kiran Somasundaram, Audrey Southerland, Yusuke Sugano, Ruijie Tao, Minh Vo, Yuchen Wang, Xindi Wu, Takuma Yagi, Ziwei Zhao, Yunyi Zhu, Pablo Arbelaez, David Crandall, Dima Damen, Giovanni Maria Farinella, Christian Fuegen, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar, Hanbyul Joo, Kris Kitani, Haizhou Li, Richard Newcombe, Aude Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi, Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei Yan, Jitendra Malik, “Ego4D: Around the World in 3,000 Hours of Egocentric Video,” arXiv:2110.07058, 2021. (Comment: to appear CVPR 2022.) https://arxiv.org/abs/2110.07058
  - Ego4D Consortium. *Ego4D project page.* https://ego4d-data.org/ (accessed 2026‑01‑10).

### A3) STAformer / AFF‑ttention (ZARRIO @ Ego4D STA)
- **Used in thesis:** Yes — Section **2.2.1** (`Final_docx/thesis_fixed.md:348`)
- **Paper:** arXiv:2407.04369 (as in your notes)
- **Code:** https://github.com/lmur98/AFFttention
- **Draft citation:**
  - Lorenzo Mur‑Labadia, Ruben Martinez‑Cantin, Josechu Guerrero‑Campo, Giovanni Maria Farinella, “ZARRIO @ Ego4D Short Term Object Interaction Anticipation Challenge: Leveraging Affordances and Attention‑based models for STA,” arXiv:2407.04369, 2024. https://arxiv.org/abs/2407.04369
  - AFFttention code. https://github.com/lmur98/AFFttention (accessed 2026‑01‑10).

### A4) SOIA‑DOD (disentangled detection → anticipate verb/noun/TTC)
- **Used in thesis:** Yes — Section **2.2.2** (`Final_docx/thesis_fixed.md:394`)
- **Paper:** arXiv:2407.05713 (as in your notes)
- **Code:** https://github.com/KeenyJin/SOIA-DOD
- **Draft citation:**
  - Hyunjin Cho, Dong Un Kang, Se Young Chun, “Short‑term Object Interaction Anticipation with Disentangled Object Detection @ Ego4D Short Term Object Interaction Anticipation Challenge,” arXiv:2407.05713, 2024. https://arxiv.org/abs/2407.05713
  - SOIA‑DOD code. https://github.com/KeenyJin/SOIA-DOD (accessed 2026‑01‑10).

### A5) Ego4D‑STA baselines in your tables (FRCNN+SF, StillFast, GANO v2, STAformer variants)
- **Used in thesis:** Yes — literature tables (`Final_docx/thesis_fixed.md:1935`)
- **Where they appear:** Table **8.6** / Table **8.7** area (top‑5 mAP baselines)
- **Full citations (baseline sources):**
  - Francesco Ragusa, Giovanni Maria Farinella, Antonino Furnari, “StillFast: An End‑to‑End Approach for Short‑Term Object Interaction Anticipation,” arXiv:2304.03959, 2023. https://arxiv.org/abs/2304.03959
  - Sanket Thakur, Cigdem Beyan, Pietro Morerio, Vittorio Murino, Alessio Del Bue, “Guided Attention for Next Active Object @ EGO4D STA Challenge (GANOv2),” arXiv:2305.16066, 2023. https://arxiv.org/abs/2305.16066
  - FRCNN+SF baseline (paper components + benchmark definition):
    - Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, “Faster R‑CNN: Towards Real‑Time Object Detection with Region Proposal Networks,” arXiv:1506.01497, 2015. https://arxiv.org/abs/1506.01497
    - Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He, “SlowFast Networks for Video Recognition,” arXiv:1812.03982, 2018. https://arxiv.org/abs/1812.03982
    - Ego4D Consortium. *Ego4D Forecasting Benchmark Documentation (STA protocol and metrics).* https://ego4d-data.org/docs/benchmarks/forecasting/ (accessed 2026‑01‑10).

---

## B) Efficient backbones & pretraining

### B1) MViTv2 (multiscale vision transformer)
- **Used in thesis:** Yes — Section **2.3.1** (`Final_docx/thesis_fixed.md:435`)
- **Link:** https://github.com/facebookresearch/mvit
- **Full citation:**
  - Yanghao Li, Chao‑Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer, “MViTv2: Improved Multiscale Vision Transformers for Classification and Detection,” in *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 4794–4804. DOI: 10.1109/CVPR52688.2022.00476
  - Code: https://github.com/facebookresearch/mvit (accessed 2026‑01‑10).

### B2) VideoMAE (masked autoencoders for video pretraining)
- **Used in thesis:** Yes — Section **2.3.2** (`Final_docx/thesis_fixed.md:471`), plus later evaluation discussion
- **Paper:** arXiv:2203.12602 (as in your notes)
- **Code:** https://github.com/MCG-NJU/VideoMAE
- **Draft citation:**
  - Zhan Tong, Yibing Song, Jue Wang, Limin Wang, “VideoMAE: Masked Autoencoders are Data‑Efficient Learners for Self‑Supervised Video Pre‑Training,” arXiv:2203.12602, 2022. (Comment: NeurIPS 2022 camera‑ready.) https://arxiv.org/abs/2203.12602
  - VideoMAE code. https://github.com/MCG-NJU/VideoMAE (accessed 2026‑01‑10).

---

## C) Token pruning / streaming efficiency

### C1) PruneVid (training‑free token pruning for video LLMs)
- **Used in thesis:** Yes — Section **2.4.1** (`Final_docx/thesis_fixed.md:516`)
- **Full citation:**
  - Xiaohu Huang, Hao Zhou, Kai Han, “PruneVid: Visual Token Pruning for Efficient Video Large Language Models,” *Findings of the Association for Computational Linguistics: ACL 2025*, 2025, pp. 19959–19973. DOI: 10.18653/v1/2025.findings-acl.1024

### C2) Rollout‑Guided Token Pruning (RGTP) for video transformers
- **Used in thesis:** Yes — Section **2.4.2** (`Final_docx/thesis_fixed.md:548`) and Track‑C design motivation
- **Code:** https://github.com/RGTPdyn/RGTP
- **Paper:** ICIP 2025 (as in your notes; DOI shown in your pasted text)
- **Draft citation:**
  - Yonatan Dinai, Ishay Goldin, Avraham Raviv, Niv Zehngut, “Rollout‑Guided Token Pruning for Efficient Video Understanding,” in *2025 IEEE International Conference on Image Processing (ICIP)*, 2025, pp. 37–42. DOI: 10.1109/ICIP55913.2025.11084634
  - RGTP code. https://github.com/RGTPdyn/RGTP (accessed 2026‑01‑10).

### C3) EgoPrune (egomotion‑aware token pruning)
- **Used in thesis:** Yes — Section **2.4.3** (`Final_docx/thesis_fixed.md:573`)
- **Paper:** arXiv:2507.15428 (as in your notes)
- **Draft citation:**
  - Jiaao Li, Kaiyuan Li, Chen Gao, Yong Li, Xinlei Chen, “EgoPrune: Efficient Token Pruning for Egomotion Video Reasoning in Embodied Agent,” arXiv:2507.15428, 2025. https://arxiv.org/abs/2507.15428

---

## D) Hand–object anticipation / affordances

### D1) PEAR (phrase‑based hand‑object interaction anticipation)
- **Used in thesis:** Yes — Section **2.5.1** (`Final_docx/thesis_fixed.md:612`)
- **Paper:** arXiv:2407.21510 (as in your notes)
- **Draft citation:**
  - Zichen Zhang, Hongchen Luo, Wei Zhai, Yang Cao, Yu Kang, “PEAR: Phrase‑Based Hand‑Object Interaction Anticipation,” *Science China Information Sciences*, 2025. DOI: 10.1007/s11432-024-4405-4 (also on arXiv:2407.21510). https://arxiv.org/abs/2407.21510

### D2) Fine‑grained affordance annotation for egocentric HOI (EPIC‑KITCHENS)
- **Used in thesis:** Yes — Section **2.5.2** (`Final_docx/thesis_fixed.md:652`)
- **Paper:** arXiv:2302.03292 (as in your notes)
- **Code:** https://github.com/zch-yu/epic-affordance-annotation
- **Draft citation:**
  - Zecheng Yu, Yifei Huang, Ryosuke Furuta, Takuma Yagi, Yusuke Goutsu, Yoichi Sato, “Fine‑grained Affordance Annotation for Egocentric Hand‑Object Interaction Videos,” arXiv:2302.03292, 2023. (Comment: WACV 2023.) https://arxiv.org/abs/2302.03292
  - EPIC affordance annotations. https://github.com/zch-yu/epic-affordance-annotation (accessed 2026‑01‑10).

---

## E) Ego ↔ Exo transfer & surveys

### E1) Ego–exo joint learning survey (“Egocentric and Exocentric Methods: A Short Survey”)
- **Used in thesis:** Yes — Section **2.6.1** (survey is described but not named explicitly) (`Final_docx/thesis_fixed.md:697`)
- **Paper:** arXiv:2410.20621 (as in your notes)
- **Draft citation:**
  - Anirudh Thatipelli, Shao‑Yuan Lo, Amit K. Roy‑Chowdhury, “Egocentric and Exocentric Methods: A Short Survey,” arXiv:2410.20621, 2024–2025. (Comment: accepted in *Computer Vision and Image Understanding (CVIU)*, 2025.) https://arxiv.org/abs/2410.20621

### E2) Synchronization‑based exo→ego transfer (“Synchronization is All You Need”)
- **Used in thesis:** Yes — Section **2.6.2** (`Final_docx/thesis_fixed.md:716`)
- **Paper:** arXiv:2312.02638 (as in your notes)
- **Code:** https://github.com/fpv-iplab/synchronization-is-all-you-need
- **Draft citation:**
  - Camillo Quattrocchi, Antonino Furnari, Daniele Di Mauro, Mario Valerio Giuffrida, Giovanni Maria Farinella, “Synchronization is All You Need: Exocentric‑to‑Egocentric Transfer for Temporal Action Segmentation with Unlabeled Synchronized Video Pairs,” arXiv:2312.02638, 2023–2024. https://arxiv.org/abs/2312.02638
  - Code. https://github.com/fpv-iplab/synchronization-is-all-you-need (accessed 2026‑01‑10).

### E3) Ego‑Only (ego‑centric training without exocentric transferring)
- **Used in thesis:** Yes — Section **2.6.3** (`Final_docx/thesis_fixed.md:732`)
- **Paper:** arXiv:2301.01380 (as in your notes)
- **Draft citation:**
  - Huiyu Wang, Mitesh Kumar Singh, Lorenzo Torresani, “Ego‑Only: Egocentric Action Detection without Exocentric Transferring,” arXiv:2301.01380, 2023. https://arxiv.org/abs/2301.01380

### E4) EgoTransfer (early ego↔exo motion transfer)
- **Used in thesis:** Yes — Section **2.6.4** (`Final_docx/thesis_fixed.md:765`)
- **Draft citation (VERIFY):**
  - **[VERIFY AUTHORS/VENUE]** *EgoTransfer: Transferring Motion Across Egocentric and Exocentric Domains using Deep Neural Networks.* **[add official link / venue / year]**

### E5) REAR (retrieval‑augmented egocentric action recognition)
- **Used in thesis:** Yes — Section **2.6.5** (`Final_docx/thesis_fixed.md:784`)
- **Draft citation (VERIFY):**
  - **[VERIFY AUTHORS/LINK]** *REAR: Retrieval‑Augmented Egocentric Action Recognition.* **[add official link / arXiv / venue]** (noted in your pasted text as “under review ICLR 2026”).

---

## F) Community hubs & lists

### F1) EgoVis workshop (CVPR 2024)
- **Used in thesis:** Yes — Section **2.7.1** (`Final_docx/thesis_fixed.md:822`)
- **Draft citation (VERIFY):**
  - **[VERIFY URL]** *First Joint Egocentric Vision (EgoVis) Workshop, CVPR 2024.* **[add official workshop URL]**

### F2) “Awesome Egocentric Action Understanding” list
- **Used in thesis:** Yes — Section **2.7.1** (`Final_docx/thesis_fixed.md:848`)
- **Link:** https://github.com/Lyman-Smoker/Awesome_Ego_Action
- **Draft citation:**
  - Lyman‑Smoker (GitHub). *Awesome Egocentric Action Understanding.* https://github.com/Lyman-Smoker/Awesome_Ego_Action (accessed 2026‑01‑10).

---

## Required Citations (for the placeholder list in `Final_docx/thesis_fixed.md#L2719`)

These correspond 1:1 to the placeholder “VERIFY” items at the end of your thesis (`Final_docx/thesis_fixed.md:2719`).

1. **Ego4D dataset and benchmark paper**
   - Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhongcong Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Cartillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli, Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Abrham Gebreselasie, Cristina Gonzalez, James Hillis, Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, Jachym Kolar, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li, Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Modhugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will Price, Paola Ruiz Puentes, Merey Ramazanova, Leda Sari, Kiran Somasundaram, Audrey Southerland, Yusuke Sugano, Ruijie Tao, Minh Vo, Yuchen Wang, Xindi Wu, Takuma Yagi, Ziwei Zhao, Yunyi Zhu, Pablo Arbelaez, David Crandall, Dima Damen, Giovanni Maria Farinella, Christian Fuegen, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar, Hanbyul Joo, Kris Kitani, Haizhou Li, Richard Newcombe, Aude Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi, Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei Yan, Jitendra Malik, “Ego4D: Around the World in 3,000 Hours of Egocentric Video,” arXiv:2110.07058, 2021 (to appear CVPR 2022). https://arxiv.org/abs/2110.07058

2. **VideoMAE (Tong et al., NeurIPS 2022)**
   - Zhan Tong, Yibing Song, Jue Wang, Limin Wang, “VideoMAE: Masked Autoencoders are Data‑Efficient Learners for Self‑Supervised Video Pre‑Training,” arXiv:2203.12602, 2022 (NeurIPS 2022 camera‑ready). https://arxiv.org/abs/2203.12602

3. **YOLOv8 (Ultralytics; specify version/date)**
   - **Software citation (recommended):** Ultralytics. *YOLOv8 (Ultralytics) object detection framework.* https://github.com/ultralytics/ultralytics (accessed 2026‑01‑10). **VERIFY:** pin exact version used in your detector training run.
   - **Local version evidence (this repo):**
     - `Thesis_main/thesis_plan_tracks_ABC_v2.md` pins `ultralytics==8.3.0`.
     - `notebooks/Ego4D_STA_Full_Colab.ipynb` pins `ultralytics==8.3.36`.
     - `local_extraction/toolkit_yolo/runs/sta_yolov8s_singlecls_20251113_002330/args.yaml` is the detector training config snapshot for the Colab run on 2025‑11‑13.

4. **CLIP (Radford et al.)**
   - Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever, “Learning Transferable Visual Models From Natural Language Supervision,” arXiv:2103.00020, 2021. https://arxiv.org/abs/2103.00020

5. **StillFast (Ego4D baseline)**
   - Francesco Ragusa, Giovanni Maria Farinella, Antonino Furnari, “StillFast: An End‑to‑End Approach for Short‑Term Object Interaction Anticipation,” arXiv:2304.03959, 2023. https://arxiv.org/abs/2304.03959

6. **GANO v2 (Ego4D baseline)**
   - Sanket Thakur, Cigdem Beyan, Pietro Morerio, Vittorio Murino, Alessio Del Bue, “Guided Attention for Next Active Object @ EGO4D STA Challenge (GANOv2),” arXiv:2305.16066, 2023. https://arxiv.org/abs/2305.16066

7. **STAformer / AFF‑ttention (ZARRIO)**
   - Lorenzo Mur‑Labadia, Ruben Martinez‑Cantin, Josechu Guerrero‑Campo, Giovanni Maria Farinella, “ZARRIO @ Ego4D Short Term Object Interaction Anticipation Challenge: Leveraging Affordances and Attention‑based models for STA,” arXiv:2407.04369, 2024. https://arxiv.org/abs/2407.04369

8. **FRCNN+SF baseline**
   - Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, “Faster R‑CNN: Towards Real‑Time Object Detection with Region Proposal Networks,” arXiv:1506.01497, 2015. https://arxiv.org/abs/1506.01497
   - Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He, “SlowFast Networks for Video Recognition,” arXiv:1812.03982, 2018. https://arxiv.org/abs/1812.03982
   - Ego4D Consortium. *Ego4D Forecasting Benchmark Documentation (baseline definition + evaluation protocol).* https://ego4d-data.org/docs/benchmarks/forecasting/ (accessed 2026‑01‑10).

9. **Transformer architecture**
   - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, “Attention Is All You Need,” arXiv:1706.03762, 2017. https://arxiv.org/abs/1706.03762

10. **Standard mAP metrics (COCO / VOC)**
   - Tsung‑Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár, “Microsoft COCO: Common Objects in Context,” arXiv:1405.0312, 2014. https://arxiv.org/abs/1405.0312
   - Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, Andrew Zisserman, “The Pascal Visual Object Classes (VOC) Challenge,” *International Journal of Computer Vision*, vol. 88, no. 2, pp. 303–338, 2009. DOI: 10.1007/s11263-009-0275-4

---

## Verification Checklist (what to do before final submission)

- Confirm each arXiv citation’s final venue/journal (when applicable) from the official PDF first page.
- For software citations (YOLOv8 / Ultralytics), pin the exact version used in training and record the date/environment.
- Ensure every DOI resolves and matches the intended paper title.
- Keep one consistent style (IEEE or APA) across all references in your thesis `# References` section.
