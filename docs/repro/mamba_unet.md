# Reproduction Documentation: Mamba-UNet (Supervised Baseline)

## 1. Paper & Official Repository
* **Paper**: Wang, Z., Zheng, J.-Q., Zhang, Y., Cui, G., & Li, L. (2024). *Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation*. arXiv:2402.05079.
* **Official Repository**: `https://github.com/ziyangwang007/Mamba-Unet` (Commit: `8a7b6c3`).
* **Source Files Ported**:
  - Visual State Space Block (VSS Block) with 2D Selective Scan (SS2D).
  - Encoder-Decoder hierarchy with Mamba bottleneck.

## 2. Architecture & Binary Flood Adaptation
* **Core SSM Operator**: Pure PyTorch 2D Selective Scan (`selective_scan_pure_pytorch`). Scans feature sequences along four complementary directional paths (horizontal forward/backward, vertical forward/backward) to capture bidirectional spatial context with linear complexity.
* **Binary Adaptation**: Final classification head maps feature representation to 2-class logits (Channel 0 = Background, Channel 1 = Flood).
* **Supervised Loss**: $\mathcal{L}_{sup} = \mathcal{L}_{dice} + \mathcal{L}_{ce}$.

## 3. Hyperparameter Specification

| Hyperparameter | Reference Value (`ziyangwang007/Mamba-Unet`) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **d_state** | 16 | 16 | Verified |
| **ssm_ratio** | 2.0 | 2.0 | Verified |
| **Selective Scan Engine**| CUDA (`mamba_ssm`) / Pure PyTorch | Pure PyTorch Chunked Fallback | Verified for cross-platform stability |
| **Optimizer** | AdamW ($\text{lr}=1\text{e-}4$, $\text{wd}=1\text{e-}2$) | AdamW ($\text{lr}=1\text{e-}4$, $\text{wd}=1\text{e-}2$) | Verified |
| **Input Resolution**| $224 \times 224$ (Synapse) | $256 \times 256$ | Unified aerial resolution |
