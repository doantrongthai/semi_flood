# Reproduction Documentation: Semi-Mamba-UNet (Semi-Supervised Baseline)

## 1. Paper Reference & Description
* **Reference**: Derived from Nguyen et al. (BSPC 2026) Section 4.5.1 / 4.5.2 (Ablation Baseline).
* **Concept**: Integrates the Visual State Space (Mamba) U-Net backbone with the Teacher-Student EMA consistency framework and the Adaptive Self-Supervised (ASS) Loss, operating in pure semi-supervised mode without few-shot support conditioning.

## 2. Architecture & Binary Flood Adaptation
* **Backbone**: `MambaUNetModel` featuring 4-way 2D selective scan bottleneck.
* **Loss Function**:
  - Supervised loss: $\mathcal{L}_{sup} = \mathcal{L}_{dice} + \mathcal{L}_{ce}$.
  - Adaptive Self-Supervised (ASS) Loss:
    $$E_w = -\sum_{c \in \{0, 1\}} p_w^c \log(p_w^c + \epsilon)$$
    $$\text{uncertain} = (E_w > \tau)$$
    $$\text{adjust} = 1.0 \times (E_w \le \tau) + 0.3 \times (E_w > \tau)$$
    $$\mathcal{L}_{self} = \text{mean}\left[ \text{adjust} \cdot \mathcal{L}_{ce}(p_s, \arg\max(p_w)) + \text{uncertain} \cdot E_s \right]$$

## 3. Hyperparameter Specification

| Hyperparameter | Reference Value (BSPC 2026) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **Uncertainty Threshold ($\tau$)** | $0.4$ | $0.4$ | Verified |
| **Low-confidence weight ($b$)** | $0.3$ | $0.3$ | Verified |
| **EMA Decay** | $0.996$ | $0.996$ | Verified |
| **Unsupervised Weight ($\lambda_u$)**| $0.2$ | $0.2$ | Verified |
| **Optimizer** | AdamW ($\text{lr}=1\text{e-}4$) | AdamW ($\text{lr}=1\text{e-}4$) | Standardized Mamba optimizer |
