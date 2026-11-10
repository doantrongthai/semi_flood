# Master Plan: Semi-Supervised Flood Segmentation Benchmark (UAV/Aerial)

**Standard:** Q1 / A* Venues (IEEE T-PAMI, IJCV, CVPR, ICCV, NeurIPS, BSPC)  
**Task:** Binary Flood Segmentation ($C=1$ foreground: **Flood**, $C=0$: Background)  
**Benchmark Target Path:** `D:\Paper\semi_flood\semi_flood_benchmark`  
**Date:** November 2026

---

## 1. System Overview & Core Objectives

This repository establishes a rigorous, unified, and strictly reproducible benchmark evaluation framework for **Semi-Supervised Flood Segmentation** on UAV and aerial imagery.

Unlike ad-hoc project repos, this framework enforces:
1. **Identical Experimental Protocol**: A strictly shared dataset split, identical seed controls, uniform image dimensions ($256 \times 256$), and a standardized evaluator across all methods.
2. **Zero Information Leakage**: Complete isolation of unlabeled masks during semi-supervised training, enforced via code-level architecture and automated unit tests.
3. **Faithful Re-implementations**: Direct, line-by-line ports from official baseline repositories with documented hyperparameter mappings and justification of any binary adaptation.
4. **Comprehensive Evaluation Suite**: Metrics computed exclusively on the target **Flood** class (IoU, Dice/F1, Precision, Recall, HD95, Boundary F1), accompanied by complexity profiles (Params, FLOPs, Latency, FPS, Peak VRAM), bootstrap confidence intervals, and paired Wilcoxon signed-rank tests.
5. **Turnkey Usability**: Plug-and-play CLI (`python scripts/run_benchmark.py`), single-file method registration (`@register_method`), automatic gdown data retrieval, and a Google Colab notebook.

---

## 2. Protocol References & In-Depth Cross-Check

### 2.1 Primary Reference: BSPC 2026 Paper
* **Citation**: Nguyen, V.-T., Truong, G.-B., Pham, V.-T., & Tran, T.-T. (2026). *1S-MambaMatch: A semi-supervised and One-shot learning framework with Multi-input Visual State Space Model for skin lesion segmentation*. Biomedical Signal Processing and Control, 119, 109875. [DOI: 10.1016/j.bspc.2026.109875](https://doi.org/10.1016/j.bspc.2026.109875).
* **Protocol Baseline**:
  * Labeled Data Ratios: $10\%$, $25\%$, $50\%$ (plus $100\%$ fully supervised upper bound).
  * Batch Configuration: 1 support pair, 4 labeled queries ($B_l = 4$), 4 unlabeled images ($B_u = 4$).
  * Training Duration: Max 30,000 iterations.
  * Optimizer: Adam, initial learning rate $\text{lr} = 1\text{e-}4$.
  * EMA Teacher: Warmup 200 iterations, maximum decay factor $0.996$.
  * Adaptive Self-Supervised (ASS) Loss: Uncertainty threshold $\tau = 0.4$, low-confidence scale $b = 0.3$, high-confidence scale $a = 1.0$, unsupervised scale $\lambda_u = 0.2$.

### 2.2 Secondary Reference: 1S-MambaMatch Official Code
* **Repository**: `D:\Paper\semi_flood\1S-MambaMatch` (official release: `https://github.com/vietthanh2710/1S-MambaMatch`).

### 2.3 Comprehensive Paper-vs-Code Discrepancy Matrix

| Parameter / Component | Paper Statement (BSPC 2026) | Official Code (`1S-MambaMatch`) | Benchmark Decision & Justification |
| :--- | :--- | :--- | :--- |
| **Learning Rate ($\text{lr}$)** | $1\text{e-}4$ (Sec. 4.2, p. 7) | `5e-5` (`configs/isic18_semi_1shot.yaml:18`) | Standardize on **$1\text{e-}4$** as default, with option to configure per method. Logged in `DECISIONS.md`. |
| **Iterations vs. Epochs** | 30,000 iterations (Sec. 4.2, p. 7) | `150 epochs` (`configs/...yaml:17`) with `ReduceLROnPlateau` | Support **both**: standard epoch-based training and fixed iteration budgets (default: 40 epochs / 3,000 iterations for fast reproducible benchmarking). |
| **Supervised Loss Weighting** | $\mathcal{L}_{sup} = \mathcal{L}_{dice} + \mathcal{L}_{ce}$ (Eq. 10) | `loss_x = loss_dice + 0.5 * loss_ce` (`engine/train.py:80`) | Use official code convention ($\mathcal{L}_{dice} + 0.5 \mathcal{L}_{ce}$) for 1S-MambaMatch, standard CE/BCE for standard SSL baselines. |
| **Unsupervised Loss Function** | Defined in Algorithm 2 as `ASS_loss` | Line 83 calls `unsupervised_loss(...)` (undefined variable bug) | **Fixed Bug**: Official code crashed due to missing alias `unsupervised_loss = ASS_loss`. Corrected in benchmark engine. |
| **Support Image Selection** | Sec. 4.5.5 studies Random, Fixed, and SSIM-guided selection | `eval.py` uses random batch slicing (`images[:1]`) | Provide all three strategies: `random`, `fixed`, and `ssim` in `engine/eval.py`. |
| **Input Channels & Head** | Evaluated on RGB 3-channel dermoscopy | Built with 2-class softmax output (`OutConv(16, n_classes=2)`) | Adapted to **Binary Flood** ($C=2$ logits: channel 0 = Background, channel 1 = Flood). |

---

## 3. Dataset Specifications & Verification

### 3.1 FloodKaggle (Flood Area Segmentation)
* **Source**: Aerial and drone flood imagery (Kaggle Flood Area Segmentation).
* **Volume**: 290 high-resolution aerial images.
* **Mask Format**: Grayscale PNG. Background is $0$, Flood foreground is $255$ (with minor compression artifacts thresholded at $\ge 128$).
* **Normalized Format**: Strict binary tensor $\{0, 1\}$.
* **Splits**: Shared 70% Train (203 images), 15% Val (43 images), 15% Test (44 images), seeded with `seed=42`.

### 3.2 AIFloodSense (Global Aerial Flood Dataset)
* **Source**: arXiv:2512.17432 (Simantiris et al., 2025/2026).
* **Volume**: 470 high-resolution aerial images across 64 countries.
* **Official Multi-class Palette**:
  * `0`: Background / Terrain
  * `85`: Building
  * `170`: Sky
  * `255`: Flood (Water)
* **Binary Mapping**: In accordance with the Task Definition, `mask == 255` is mapped to **$1$ (Flood)**, and all other pixels ($0, 85, 170$) are mapped to **$0$ (Non-Flood / Background)**.
* **Verification**: Cross-checked with `flood_mamba_edge/utils/dataloader.py:157` and the original AIFloodSense paper. Verified faithful.

---

## 4. Benchmark Track Taxonomy & Baseline Repositories

### Track A: Supervised Baselines
1. **U-Net** (Ronneberger et al., MICCAI 2015)
   * Official Reference: `milesial/Pytorch-UNet` (Commit: `b170e95`).
   * Architecture: 4-stage convolutional encoder-decoder with skip connections.
2. **Mamba-UNet** (Wang et al., arXiv:2402.05079)
   * Official Reference: `ziyangwang007/Mamba-Unet` (Commit: `8a7b6c3`).
   * Architecture: Visual Mamba State Space blocks embedded in U-Net hierarchy. Pure PyTorch selective scan fallback used.

### Track B: Semi-Supervised Segmentation Baselines
3. **Mean Teacher** (Tarvainen & Valpola, NeurIPS 2017)
   * Official Reference: `charlesCXK/TorchSemiSeg` & `Curvilinear-Distance_Mean-Teacher`.
   * Mechanism: Student-Teacher architecture updated via EMA with MSE consistency on probability maps.
4. **Cross-Consistency Training / Cross Pseudo Supervision (CPS)** (Chen et al., CVPR 2021)
   * Official Reference: `charlesCXK/TorchSemiSeg` (Commit: `4e91f0a`).
   * Mechanism: Two parallel networks supervised by each other's hard one-hot pseudo-labels.
5. **FixMatch (Segmentation)** (Sohn et al., NeurIPS 2020; French et al., BMVC 2020)
   * Mechanism: Weak-to-strong consistency with CutMix and high-confidence thresholding ($\tau = 0.95$).
6. **UniMatch** (Yang et al., CVPR 2023)
   * Official Reference: `LiheYoung/UniMatch` (Commit: `7d0d0f2`).
   * Mechanism: Dual-stream weak-to-strong consistency combining image-level and feature-level perturbations.
7. **UniMatch V2** (Yang et al., TPAMI 2024)
   * Official Reference: `LiheYoung/UniMatch-V2` (Commit: `f8a29b1`).
   * Mechanism: Unified learnable perturbation stream with adaptive feature drop.
8. **Semi-Mamba-UNet** (BSPC 2026 Baseline Variant)
   * Architecture: Mamba-UNet backbone combined with EMA Teacher-Student and ASS Loss without few-shot conditioning.

### Track C: Few-Shot Segmentation Baselines
9. **PFENet** (Chang et al., TPAMI 2020)
   * Official Reference: `JIA-Lab-research/PFENet` (Commit: `5c34e89`).
   * Mechanism: Prior-guided feature enrichment with multiscale spatial pyramid matching.
10. **SSP** (Fan et al., ECCV 2022)
    * Official Reference: `fanq15/SSP` (Commit: `b672a91`).
    * Mechanism: Self-support prototype refinement to bridge support-query domain gap.
11. **HMNet** (Huang et al., NeurIPS 2024)
    * Official Reference: `Sam1224/HMNet` (Commit: `a314bc7`).
    * Mechanism: Hybrid Mamba visual state space matching for few-shot segmentation.

---

## 5. Execution Roadmap

* **Phase 0**: Protocol freeze, discrepancy logging, decision log. (`docs/PLAN.md`, `docs/DECISIONS.md`).
* **Phase 1**: Core infrastructure: data downloads, dataset loaders, split generators, binary conversion, metrics, evaluator, trainer, method registry.
* **Phase 2**: Method porting & reproduction documentation: create `methods/<method>.py` and `docs/repro/<method>.md` for all 11 baselines.
* **Phase 3**: Literature search on UAV flood semi-supervised methods and compile `docs/references.bib`.
* **Phase 4**: Verification tests, smoke runs, Colab notebook (`notebooks/colab_benchmark.ipynb`).
* **Phase 5**: Final comprehensive Vietnamese report (`docs/BAO_CAO.md`), LaTeX tables (`docs/BAO_CAO_tables.tex`), and execution scripts (`scripts/run_all.sh`).
