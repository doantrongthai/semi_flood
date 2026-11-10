# Reproduction Documentation: UniMatch V2 (Semi-Supervised Baseline)

## 1. Paper & Official Repository
* **Paper**: Yang, L., Qi, L., Feng, L., Zhang, W., & Shi, J. (2024). *UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / arXiv:2403.07889.
* **Official Repository**: `https://github.com/LiheYoung/UniMatch-V2` (Commit: `f8a29b1`).
* **Source Files Ported**:
  - Unified learnable stream combining image and feature space consistency.
  - Dynamic adaptive thresholding.

## 2. Architecture & Binary Flood Adaptation
* **Core Update**: Replaces decoupled streams with an integrated end-to-end learnable strong stream. Incorporates adaptive thresholding $\tau_t$ that ramps up with training iterations.
* **Loss Function**:
  $$\tau_t = \max(0.85, \min(0.95, 0.85 + 0.1 \times \frac{t}{T}))$$
  $$\mathcal{L}_u = \mathcal{L}_{ce}(M(x_u^{s}), \hat{y}_w) \cdot (\max(p_w) \ge \tau_t)$$

## 3. Hyperparameter Specification

| Hyperparameter | Official Reference (`UniMatch-V2`) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **Adaptive Threshold ($\tau$)** | Dynamic ($0.85 \to 0.95$) | Dynamic ($0.85 \to 0.95$) | Verified |
| **Unsupervised Weight ($\lambda_u$)**| $1.0$ | $0.5$ | Tuned for binary flood balance |
| **Optimizer** | AdamW ($\text{lr}=1\text{e-}4$) | Adam ($\text{lr}=1\text{e-}4$) | Standardized benchmark optimizer |
