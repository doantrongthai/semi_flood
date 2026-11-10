# Reproduction Documentation: PFENet (Few-Shot Baseline)

## 1. Paper & Official Repository
* **Paper**: Tian, Z., Zhao, H., Shu, M., Yang, Z., Li, R., & Jia, J. (2020). *Prior Guided Feature Enrichment Network for Few-Shot Segmentation*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / CVPR 2020.
* **Official Repository**: `https://github.com/JIA-Lab-research/PFENet` (Commit: `5c34e89`).
* **Source Files Ported**:
  - `model/PFENet.py`: Prior Generation Module (cosine similarity map) and Multi-scale Feature Enrichment Module (FEM).

## 2. Architecture & Binary Flood Adaptation
* **Episode Construction**: 1 support image-mask pair $(x_{sp}, y_{sp})$ + query image $x_q$.
* **Prior Map Generation**:
  $$p = \cos(\text{masked\_avg\_pool}(f_{sp}, y_{sp}), f_q)$$
* **Feature Enrichment**: Fuses multiscale query features with the spatial prior map to guide dense flood boundary localization.
* **Support Selection**: Supports random selection, fixed exemplar, and SSIM-guided candidate selection from the labeled training split.

## 3. Hyperparameter Specification

| Hyperparameter | Official Reference (`PFENet`) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **Backbone** | ResNet50 / VGG | 4-stage Conv-U-Net Feature Extractor | Unified benchmark capacity |
| **Shots** | 1-shot | 1-shot (single flood class) | Verified |
| **Optimizer** | SGD ($\text{lr}=2.5\text{e-}3$) | Adam ($\text{lr}=1\text{e-}4$) | Standardized benchmark optimizer |
