# Reproduction Documentation: U-Net (Supervised Baseline)

## 1. Paper & Official Repository
* **Paper**: Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 234-241. [DOI: 10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28).
* **Official / Reference Repository**: `https://github.com/milesial/Pytorch-UNet` (Commit: `b170e95`).
* **Source Files Ported**:
  - `unet/unet_parts.py`: `DoubleConv`, `Down`, `Up`, `OutConv`.
  - `unet/unet_model.py`: `UNet`.

## 2. Architecture & Binary Flood Adaptation
* **Backbone**: Standard 4-stage convolutional encoder-decoder with skip connections:
  - Channels: $64 \to 128 \to 256 \to 512 \to 512$ (bottleneck) $\to 256 \to 128 \to 64$.
* **Binary Adaptation**: OutConv set to `n_classes=2` (Channel 0 = Background, Channel 1 = Flood).
* **Supervised Loss**: Combination of Dice Loss and Cross-Entropy Loss:
  $$\mathcal{L}_{sup} = \mathcal{L}_{dice}(p_1, y) + \mathcal{L}_{ce}(p, y)$$

## 3. Hyperparameter Specification

| Hyperparameter | Reference Value (milesial / standard) | Benchmark Value (This Repo) | Status / Justification |
| :--- | :--- | :--- | :--- |
| **Input Channels** | 3 (RGB) | 3 (RGB) | Verified |
| **Output Classes** | 2 (Carvana binary) | 2 (Flood vs Background) | Verified |
| **Input Resolution**| $512 \times 512$ | $256 \times 256$ | Unified aerial resolution |
| **Optimizer** | Adam ($\text{lr}=1\text{e-}4$, $\beta_1=0.9, \beta_2=0.999$) | Adam ($\text{lr}=1\text{e-}4$, $\text{wd}=1\text{e-}4$) | Verified |
| **Scheduler** | ReduceLROnPlateau / Cosine | Cosine Annealing | Unified benchmark schedule |
| **Batch Size** | 4 | 4 | Aligned with BSPC benchmark |
