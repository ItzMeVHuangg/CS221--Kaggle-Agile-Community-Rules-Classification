# Jigsaw - Agile Community Rules Classification Framework

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License" /></a>
  <img src="https://img.shields.io/badge/Python-3.8+-3c873a?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Models-Transformers_&_Prompt_Eng-blue?style=flat-square" alt="Model" />
  <img src="https://img.shields.io/badge/Dataset-Kaggle_Jigsaw-orange?style=flat-square" alt="Dataset" />
</p>

This repository presents an end-to-end framework for detecting community rule violations on Reddit, rigorously evaluated on the **Jigsaw - Agile Community Rules Classification** dataset. 

To solve this unique "Agile" challenge—where a model must not merely classify text in isolation but dynamically assess the contextual relationship between a comment (`body`) and a specific community rule (`rule`)—our research pipeline is systematically split into two methodology phases:
1. **Architectural Exploration (Phase 1):** Building strong foundational benchmarks utilizing classical machine learning (TF-IDF + Logistic Regression) and lightweight deep learning (DistilBERT) architectures paired with dynamic classification threshold search.
2. **Advanced Prompt Engineering Framework (Phase 2):** Crafting and evaluating three discrete input restructuring strategies (Sentence-Pair, Few-Shot, and Context-Aware) across high-capacity pre-trained text encoders (DeBERTa-v3 and RoBERTa) to optimize cross-attention reasoning.

---

## 1. Phase 1: Foundational Benchmarks & Architectural Exploration

Before scaling up to heavy architectures, a comprehensive text preprocessing pipeline (`preprocess_text`) was built to unify the corpus by stripping HTML formatting, standardizing hyperlinks using a global `URL` token, mapping contractions via the `contractions` library, removing punctuation noise, and applying lowercasing.

- **Statistical Baseline:** Leveraged TF-IDF text vectorization to capture word-level frequencies coupled with a linear Logistic Regression classifier.
- **Lightweight Deep Learning Baseline:** Integrated a `DistilBERT` backbone (66M parameters) to assess structural cross-attention efficiency with low computational overhead under a dual-sequence paradigm.
- **Dynamic Thresholding:** Replaced the rigid $0.50$ classification barrier with a fine-grained localized threshold sweep ($\tau \in [0.30, 0.70]$) optimized on validation distributions to maximize the target F1-score.

---

## 2. Phase 2: Advanced Prompt Engineering Framework

To transcend keyword-matching limitations and enable rich deep semantic inference, we engineered three structural prompt templates, mapping the task directly onto a Natural Language Inference (NLI) configuration across `DeBERTa-v3-small` (utilizing Disentangled Attention) and `RoBERTa-base` backbones.

- **Strategy 1: Sentence-Pair (SP):** The classic NLI structure feeding the raw segments contiguously into the network encoder: `[CLS] + body_clean + [SEP] + rule_clean + [SEP]`. This design forces the cross-attention layers to mathematically compute token-to-token alignments across strings.
- **Strategy 2: Few-Shot Guidance (FS):** Dynamically prepends explicit ground-truth positive and negative exemplars retrieved from the corpus metadata: `body_clean [SEP] rule_clean [SEP] POSITIVE: ... [SEP] NEGATIVE: ...`, expanding maximum block context lengths to 256 tokens to guide model pattern logic.
- **Strategy 3: Context-Aware Template (CA):** Injects spatial community-level metadata by wrapping the source comment into an explicit location structure: `In r/SUBREDDIT, this comment: body_clean`, enabling the backbone to adapt to distinct forum sub-cultures.
- **Optimization Strategy:** Implemented **Layer-wise Learning Rate Decay (LLRD)** to protect pre-trained low-level features within encoder weights, allowing the custom binary projection head to converge securely using `BCEWithLogitsLoss` under stable gradient clipping.

---

## 3. Dataset Summary

Experiments are driven by the official dataset from the **Jigsaw - Agile Community Rules Classification** Kaggle competition.
- **Data Split Configuration:** Comprises a consolidated set of 2,029 training instances. To protect generalization boundaries, a rigorous **Stratified Split (80/20)** separated the data into 1,623 training entries and 406 distinct validation entries using fixed seed control (`random_state=42`).
- **Label Distribution:** Exhibits a near-perfect balanced distribution containing 50.81% violation instances (Label 1: 1,031 samples) and 49.19% valid entries (Label 0: 998 samples).
- **Rule Demographics:** Concentrates entirely on exactly 2 distinct community regulations: `No Advertising` (1,012 samples) and `No Legal Advice` (1,017 samples).
- **Metadata Context:** Covers over 100 unique subreddits showing steep behavioral variances (e.g., `r/legaladvice` maintains a heavy 78.9% violation rate, whereas `r/soccerstreams` reports a low 2.9%).

---

## 4. Experimental Results

The table below provides an exhaustive performance breakdown of all 12 structural configurations (4 core model backbones across 3 prompt variations) tested on the isolated validation split (406 samples):

| Configuration ID | Model Backbone | Prompt Engineering Strategy | F1-Score | ROC-AUC | Optimal Threshold ($\tau$) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | Logistic Regression | Sentence-Pair (SP) | 0.7638 | 0.8081 | 0.50 |
| 2 | Logistic Regression | Few-Shot Guidance (FS) | 0.7076 | 0.6346 | 0.40 |
| 3 | Logistic Regression | Context-Aware Template (CA) | 0.7586 | 0.8076 | 0.48 |
| 4 | DistilBERT | Sentence-Pair (SP) | 0.8047 | 0.8566 | 0.58 |
| 5 | DistilBERT | Few-Shot Guidance (FS) | 0.7982 | 0.8309 | 0.50 |
| 6 | DistilBERT | Context-Aware Template (CA) | 0.8219 | 0.8729 | 0.52 |
| 7 | DeBERTa-v3-small | Sentence-Pair (SP) | 0.7807 | 0.8232 | 0.64 |
| 8 | DeBERTa-v3-small | Few-Shot Guidance (FS) | 0.7920 | 0.8272 | 0.46 |
| 9 | DeBERTa-v3-small | Context-Aware Template (CA) | 0.8219 | 0.8654 | 0.46 |
| 10 | 🏆 **RoBERTa-base** | **Sentence-Pair (SP)** | **0.8436** | 0.8838 | **0.40** |
| 11 | RoBERTa-base | Few-Shot Guidance (FS) | 0.8341 | **0.8977** | 0.52 |
| 12 | RoBERTa-base | Context-Aware Template (CA) | 0.8315 | 0.8880 | 0.30 |

---

## 5. Discussion & Analysis

The comparative matrix uncovers critical empirical behaviors within modern moderation architectures:
- **Transformer Dominance:** Deep learning encoders significantly outclass classic Bag-of-Words methods, introducing absolute absolute improvements ranging from **+2.7 to +8.0 F1 points** above the baseline. This validates the capacity of Self-Attention networks to process complex contextual negation over simple token frequencies.
- **The Dual Effect of Few-Shotting:** In traditional machine learning, Few-Shotting drastically damages performance (F1 collapses to 0.7076) due to superficial lexical overlap noise introduced from standard templates. Conversely, high-capacity Transformer backbones successfully digest this long context, enabling **RoBERTa-FS** to secure the highest overall discriminatory ranking with a **peak ROC-AUC of 0.8977**.
- **Context-Aware Acceleration:** Explicit forum localization via Context-Aware templates acts as a powerful catalyst for mid-sized networks, boosting **DistilBERT-CA** and **DeBERTa-CA** up to an identical high-tier F1-score of **0.8219** (a localized leap of +2.2 and +4.1 points respectively).
- **The SOTA Setup:** **RoBERTa-base (SP)** established the definitive **SOTA F1-Score profile of 0.8436** at an adaptive threshold of $\tau = 0.40$, signaling that its massive internal representation capacity is self-sufficient at mapping rule boundaries natively without secondary prompt overhead.

---

## 6. Conclusion, Limitations & Future Directions

**Core Achievements:**
This project establishes an empirical framework for contextual moderation, transitioning the text-matching problem into a robust NLI pipeline. By dynamically mapping adaptive thresholds, the solution provides fine-grained control to minimize false-positive bans in active online communities.

**Limitations:**
- Lightweight architectures like DistilBERT experience attention scattering and data overhead when processing lengthy 256-token few-shot prompt patterns.
- High-capacity encoders like DeBERTa show slight indicators of validation overfitting on densely distributed subreddits when wrapped in Context-Aware strings.
- Qualitative analysis over 60 stratified audit validation entries revealed label ambiguity regarding counter-intuitive or highly subjective moderator ground-truth labels.

**Future Directions:**
- **Hybrid Multi-Modal Ensemble:** Implementing soft-voting logit blending combining the strengths of RoBERTa-SP, RoBERTa-FS, and DeBERTa-CA to stabilize prediction borders.
- **Pre-trained NLI Checkpoint Bootstrapping:** Transferring specialized weights from networks already fine-tuned on large-scale textual entailment data (e.g., `cross-encoder/nli-deberta-v3-small`) to eliminate the pre-training task gap.
- **Granular Fragmented Thresholding:** Transitioning from a universal threshold $\tau$ toward dedicated, decoupled dynamic boundaries tailored separately per rule or per individual subreddit.
- **Hard-Negative Mining Integration:** Explicitly augmenting training loops with adversarial borderline samples (e.g., text blocks with legitimate hyperlinks that are not advertisements, or legal terminology devoid of advisory intent) to refine decision margins.

---

## 7. Team Members

This project was engineered and analyzed under the guidance of **Dr. Nguyen Trong Chinh** at the Faculty of Information Science and Engineering, University of Information Technology (UIT) - VNU-HCM.

| Member Name | Student ID |
| :--- | :--- |
| **Vu Viet Hoang** | 23520548 |
| **Trinh Nguyen Thanh Binh** | 23520162 |

---
*For extensive mathematical pool configuration documentation or a full review of the 60 stratified data auditing matrices, please refer directly to the compiled project report (`Report.pdf`) and the localized training execution scripts (`Codespace.ipynb`).*
