# Reproduction Documentation: UniMatch (Semi-Supervised Baseline)

## 1. Paper & Official Repository
* **Paper**: Yang, L., Qi, L., Feng, L., Zhang, W., & Shi, J. (2023). *Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023), pp. 7236-7246.
* **Official Repository**: `https://github.com/LiheYoung/UniMatch` (Commit: `7d0d0f2`).
* **Source Files Ported**:
  - `unimatch.py`: Dual-stream consistency (image-level + feature-level perturbations).
  - Feature dropout / channel perturbation module.

## 2. Architecture & Binary Flood Adaptation
* **Core Novelty**: Unifies weak-to-strong consistency by applying perturbations in both the raw input space (image-level) and the deep representation space (feature-level).
* **Binary Formulation**:
  $$p_w = \text{softmax}(M(x_u^w))$$
  $$\hat{y} = \arg\max(p_w), \quad \text{mask} = (\max(p_w) \ge \tau)$$
  $$\mathcal{L}_u = \frac{1}{2} \left[ \mathcal{L}_{ce}(M(x_u^{s1}), \hat{y}) + \mathcal{L}_{ce}(M(x_u^{s2, \text{feat}}), \hat{y}) \right] \cdot \text{mask}$$

## 3. Hyperparameter Specification

| Hyperparameter | Official Reference (`LiheYoung/UniMatch`) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **Confidence Threshold ($\tau$)** | $0.95$ | $0.95$ | Verified |
| **Feature Drop Probability** | $0.2$ | $0.2$ | Verified |
| **Unsupervised Weight ($\lambda_u$)**| $1.0$ | $0.5$ | Tuned for binary flood balance |
| **Optimizer** | AdamW / SGD | Adam ($\text{lr}=1\text{e-}4$) | Standardized benchmark optimizer |
