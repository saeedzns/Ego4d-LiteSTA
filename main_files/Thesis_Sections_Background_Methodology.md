# 1. Background

Egocentric video understanding has become a central topic in computer vision due to the increasing availability of wearable cameras and AR/VR devices. Unlike third‑person (exocentric) recordings, egocentric video provides a direct view of human–object interactions, hand trajectories, and gaze‑aligned visual information. These properties make it uniquely suited for studying short‑term predictions, assistive technologies, and embodied AI systems.  
The Ego4D dataset is currently the largest and most diverse egocentric dataset, spanning thousands of hours of daily‑life activities and providing detailed annotations for forecasting tasks such as Short-Term Object Interaction Anticipation (STA). STA requires predicting the next active object, the verb describing the upcoming interaction, and the time‑to‑contact (TTC) before the interaction begins.  
Recent STA solutions tend to rely on heavy transformer models or multimodal fusion approaches. Although accurate, these systems introduce high computational cost, long inference times, and practical limitations for real‑time deployment on wearable devices. This motivates the need for lightweight anticipation frameworks that preserve competitive accuracy while significantly reducing computational demand.

---

# 1.2 Problem Statement

Short‑Term Object Interaction Anticipation is a challenging problem that combines spatial detection, temporal reasoning, and semantic understanding. The STA task requires identifying the object the camera wearer is most likely to interact with, predicting the associated verb, and estimating the time to contact—all from a short observation window.  
Existing high‑performance STA models face several limitations:

- **High computational cost:** Transformer‑based video encoders typically require substantial GPU resources, limiting real‑time use.  
- **Redundant visual tokens:** Egocentric video contains spatial and temporal redundancies that increase processing load without improving predictions.  
- **Complex multi‑task coupling:** Object detection, verb prediction, and TTC estimation are often trained jointly, leading to optimization conflicts.  
- **Deployment constraints:** Many models are too large to operate efficiently on mobile or embedded hardware used in wearable AR systems.

The problem addressed in this thesis is the development of a **lightweight, modular, and efficient STA pipeline** that maintains reasonable predictive accuracy while targeting reduced computational overhead. The goal is to retain essential anticipation capabilities while eliminating unnecessary processing.

---

# 1.3 Objectives of the Study

The thesis aims to design and evaluate an efficient framework—Ego4D‑LiteSTA—for short‑term object interaction anticipation in egocentric video. The study pursues the following objectives:

- To analyze and decompose the STA task into modular stages that can be optimized independently.  
- To develop a high‑recall proposal mechanism capable of capturing potential active objects reliably.  
- To design a lightweight temporal–spatial fusion model that predicts the next active object, verb, and time‑to‑contact with reduced computational complexity.  
- To integrate a training‑free token‑pruning technique that decreases inference cost while preserving accuracy.  
- To evaluate the full pipeline on the Ego4D STA benchmark and document the trade‑offs between accuracy, efficiency, and architectural simplicity.  
- To demonstrate that resource‑aware egocentric anticipation can approach state‑of‑the‑art accuracy using significantly reduced compute.

---

# 1.5 Research Methodology

The research methodology follows a structured, multi‑stage approach aligned with the modular design of the Ego4D‑LiteSTA pipeline:

### **Stage 1 — Dataset Preparation and Exploration**
- Use the Ego4D STA v2 dataset with its annotations for nouns, verbs, bounding boxes, and TTC values.  
- Analyze frame sequences, object distributions, and annotation structures to guide model design.

### **Stage 2 — Proposal Generation (Track A)**
- Implement a high‑recall candidate extraction mechanism operating on the final observed frame.  
- Validate recall against ground‑truth boxes to ensure downstream modules receive accurate candidates.

### **Stage 3 — Lightweight Fusion Model (Track B)**
- Build a compact neural head that fuses spatial cues from the last frame with short‑term temporal features.  
- Train the module to output object scores, verb probabilities, and TTC predictions.  
- Measure accuracy using STA metrics (N mAP, N+V mAP, N+δ mAP, Overall).

### **Stage 4 — Efficiency Module (Track C)**
- Apply training‑free token‑pruning using rollout‑guided or redundancy‑based techniques.  
- Evaluate the effect of pruning ratios on FLOPs, latency, and accuracy.

### **Stage 5 — Experimental Evaluation**
- Conduct systematic experiments to compare baselines, pruned models, and ablated variants.  
- Report detailed metrics and complexity trends to illustrate trade‑offs.

### **Stage 6 — Interpretation and Discussion**
- Analyze strengths, limitations, and failure cases.  
- Translate empirical findings into architectural recommendations for egocentric forecasting systems.

---

This structured methodology ensures that each component of the pipeline is rigorously analyzed, optimized, and validated, resulting in a coherent and efficient short‑term anticipation framework.
