# Reproduction Documentation: FixMatch (Semi-Supervised Baseline)

## 1. Paper & Official Repository
* **Paper**: Sohn, K., Berthelot, D., Carlini, N., Zhang, Z., Zhang, H., Raffel, C. A., Cubuk, E. D., Kurakin, A., & Li, C.-L. (2020). *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*. Advances in Neural Information Processing Systems (NeurIPS 2020), 33, pp. 596-608.
* **Semantic Segmentation Port**: French et al. (BMVC 2020) / Standard FixMatch segmentation protocol (`LiheYoung/UniMatch:model/fixmatch.py`).
* **Source Files Ported**:
  - Weak-to-strong consistency framework.
  - Confidence thresholding ($\tau = 0.95$).
  - CutMix spatial blending on strongly augmented views.

## 2. Architecture & Binary Flood Adaptation
* **Backbone**: U-Net architecture.
* **Unsupervised Loss Formulation**:
  $$p_w = \text{softmax}(M(x_u^w))$$
  $$\text{conf\_mask} = (\max_{c \in \{0, 1\}} p_w^c \ge \tau)$$
  $$\hat{y} = \arg\max_{c \in \{0, 1\}} p_w^c$$
  $$\mathcal{L}_u = \frac{\sum_{i,j} \text{conf\_mask}_{i,j} \cdot \mathcal{L}_{ce}(M(x_u^s)_{i,j}, \hat{y}_{i,j})}{\sum_{i,j} \text{conf\_mask}_{i,j} + \epsilon}$$

## 3. Hyperparameter Specification

| Hyperparameter | Official Reference (Sohn et al. / UniMatch) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **Confidence Threshold ($\tau$)** | $0.95$ | $0.95$ | Verified |
| **Unsupervised Weight ($\lambda_u$)**| $1.0$ | $0.5$ | Tuned for binary flood balance |
| **Strong Augmentation** | Color Jitter + Grayscale + Blur + CutMix | Identical (`data/augmentations.py`) | Verified |
| **Optimizer** | Adam ($\text{lr}=1\text{e-}4$) | Adam ($\text{lr}=1\text{e-}4$) | Standardized benchmark optimizer |
