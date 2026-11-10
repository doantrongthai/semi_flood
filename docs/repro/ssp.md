# Reproduction Documentation: SSP (Few-Shot Baseline)

## 1. Paper & Official Repository
* **Paper**: Fan, Q., Zhuo, W., & Tang, C.-K. (2022). *Self-Support Few-Shot Semantic Segmentation*. European Conference on Computer Vision (ECCV 2022), pp. 437-454.
* **Official Repository**: `https://github.com/fanq15/SSP` (Commit: `b672a91`).
* **Source Files Ported**:
  - `model/ssp.py`: Self-Support Prototype matching and iterative query prototype refinement.

## 2. Architecture & Binary Flood Adaptation
* **Mechanism**:
  1. Computes initial support prototype $P_s$ via masked average pooling.
  2. Generates initial query segmentation $M_q^{(1)}$ via cosine distance.
  3. Extracts self-support prototype from query's high-confidence predictions: $P_q = \text{MAP}(f_q, M_q^{(1)})$.
  4. Blends $P_{refined} = \alpha P_s + (1 - \alpha) P_q$ and computes final query segmentation $M_q^{(2)}$.
* **Binary Flood Adaptation**: Operates directly on the single Flood foreground class.

## 3. Hyperparameter Specification

| Hyperparameter | Official Reference (`SSP`) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **Self-Support Weight ($\alpha$)** | $0.6$ | $0.6$ | Verified |
| **Shots** | 1-shot | 1-shot | Verified |
| **Optimizer** | SGD | Adam ($\text{lr}=1\text{e-}4$) | Standardized benchmark optimizer |
