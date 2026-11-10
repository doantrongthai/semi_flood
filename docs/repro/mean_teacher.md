# Reproduction Documentation: Mean Teacher (Semi-Supervised Baseline)

## 1. Paper & Official Repository
* **Paper**: Tarvainen, A., & Valpola, H. (2017). *Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, pp. 1195-1204.
* **Official Segmentation Reference**: `https://github.com/charlesCXK/TorchSemiSeg` (`exp_cityscapes/train_mean_teacher.py`).
* **Source Files Ported**:
  - Dual Student-Teacher network synchronization.
  - Mean Squared Error (MSE) probability consistency on unlabeled views.
  - Exponential Moving Average (EMA) teacher parameter update.

## 2. Architecture & Binary Flood Adaptation
* **Backbone**: U-Net architecture. Student and Teacher share identical architecture and initial weights.
* **Loss Function**:
  $$\mathcal{L} = \mathcal{L}_{sup}(p_{student}^l, y^l) + \lambda_u \cdot \text{MSE}(\text{softmax}(p_{student}^u), \text{softmax}(p_{teacher}^u))$$
* **Evaluation Network**: Teacher model (evaluated on validation and test sets as in official paper).

## 3. Hyperparameter Specification

| Hyperparameter | Official Reference (`TorchSemiSeg`) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **Consistency Loss** | MSE on Softmax | MSE on Softmax | Verified |
| **EMA Decay** | $0.99$ to $0.999$ | $0.996$ (warmup 200 iters) | Verified, matches BSPC protocol |
| **Unsupervised Weight ($\lambda_u$)** | $1.0$ | $0.5$ | Tuned for aerial flood stability |
| **Optimizer** | Adam / SGD | Adam ($\text{lr}=1\text{e-}4$) | Standardized benchmark optimizer |
