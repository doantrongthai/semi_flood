# Reproduction Documentation: HMNet (Few-Shot Baseline)

## 1. Paper & Official Repository
* **Paper**: Huang, Z., et al. (2024). *Hybrid Mamba for Few-Shot Segmentation*. Advances in Neural Information Processing Systems (NeurIPS 2024).
* **Official Repository**: `https://github.com/Sam1224/HMNet` (Commit: `a314bc7`).
* **Source Files Ported**:
  - Hybrid Mamba cross-sequence interaction module.
  - Support-query state space matching mechanism.

## 2. Architecture & Binary Flood Adaptation
* **Core SSM Matching**: Flattens support feature tokens $S \in \mathbb{R}^{L \times D}$ and query feature tokens $Q \in \mathbb{R}^{L \times D}$ into an interleaved compound sequence $[S, Q]$ and runs 2D Selective Scan to dynamically exchange contextual tokens with linear complexity.
* **Pure PyTorch Implementation**: Utilizes the chunked pure-PyTorch selective scan engine for full platform portability.
* **Binary Flood Adaptation**: Final linear classification head predicts 2-class logits (Channel 0 = Background, Channel 1 = Flood).

## 3. Hyperparameter Specification

| Hyperparameter | Official Reference (`HMNet`) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **SSM State Dimension ($N$)** | 16 | 16 | Verified |
| **Shots** | 1-shot | 1-shot | Verified |
| **Optimizer** | AdamW ($\text{lr}=1\text{e-}4$) | AdamW ($\text{lr}=1\text{e-}4$) | Standardized Mamba optimizer |
