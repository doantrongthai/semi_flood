# Reproduction Documentation: Cross Teaching / CPS (Semi-Supervised Baseline)

## 1. Paper & Official Repository
* **Paper**: Chen, X., Yuan, Y., Zeng, G., & Wang, J. (2021). *Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021), pp. 2657-2666.
* **Official Repository**: `https://github.com/charlesCXK/TorchSemiSeg` (`exp_cityscapes/train_cps.py`).
* **Source Files Ported**:
  - Dual-branch network architecture (Branch 1 and Branch 2).
  - Cross pseudo-supervision loss logic.

## 2. Architecture & Binary Flood Adaptation
* **Structure**: Two parallel U-Net models ($M_1, M_2$) initialized with different random seeds.
* **Loss Function**:
  - Labeled Loss: $\mathcal{L}_{sup} = \mathcal{L}_{sup}(M_1(x_l), y_l) + \mathcal{L}_{sup}(M_2(x_l), y_l)$.
  - Cross Pseudo Supervision on Unlabeled Data:
    $$y_1^* = \arg\max(M_1(x_u)), \quad y_2^* = \arg\max(M_2(x_u))$$
    $$\mathcal{L}_{cps} = \mathcal{L}_{ce}(M_1(x_u), y_2^*) + \mathcal{L}_{ce}(M_2(x_u), y_1^*)$$
* **Binary Adaptation**: Cross-entropy applied over 2 classes (Channel 0 = Background, Channel 1 = Flood).

## 3. Hyperparameter Specification

| Hyperparameter | Official Reference (`TorchSemiSeg`) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **CPS Weight ($\lambda_{cps}$)** | $1.5$ | $0.5$ | Scaled for binary aerial stability |
| **Supervised Loss** | Cross Entropy | Cross Entropy + Dice | Enhanced for flood class imbalance |
| **Optimizer** | Adam / SGD | Adam ($\text{lr}=1\text{e-}4$) | Standardized benchmark optimizer |
