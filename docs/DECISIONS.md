# Architectural & Protocol Decisions Log

This document records all architectural, data, protocol, and algorithmic decisions made in the `semi_flood_benchmark` framework. Every decision cites concrete evidence and lists associated risks.

---

## Decision 1: Model Prediction Output Convention (2-Class Logits)

* **Context:** The task is Binary Flood Segmentation (Flood vs Background). Models can output either 1 channel (logit + Sigmoid) or 2 channels (logits + Softmax).
* **Options Considered:**
  1. *1 Channel + Sigmoid:* Traditional binary segmentation convention. Simpler for supervised loss (BCEWithLogits).
  2. *2 Channels + Softmax:* Standard convention in semi-supervised semantic segmentation frameworks (UniMatch, FixMatch, CPS, Mean Teacher, 1S-MambaMatch).
* **Evidence:**
  - Official `UniMatch` (`LiheYoung/UniMatch:model/semseg/deeplabv3plus.py`) and `FixMatch` rely on `torch.argmax(dim=1)` for pseudo-label generation and `F.cross_entropy(logits, pseudo_labels)`.
  - `1S-MambaMatch` (`models/oneshotmambaunet.py:87, 126`) explicitly defines `n_classes: int = 2` with `OutConv(16, n_classes)`.
  - Entropy computation for ASS Loss ($-\sum_c p_c \log p_c$) mathematically generalizes over discrete probability distributions ($K=2$).
* **Choice:** **Option 2 (2-Class Logits `(B, 2, H, W)`)**. Channel 0 is Background, Channel 1 is Flood.
* **Risk:** Minimal. All metrics evaluate exclusively on Channel 1 (Flood) via `(logits.argmax(dim=1) == 1)`.

---

## Decision 2: AIFloodSense Binary Mask Mapping

* **Context:** AIFloodSense is a multi-class dataset containing annotations for Background, Building, Sky, and Flood. The benchmark task is strictly binary flood extraction.
* **Options Considered:**
  1. *Map all non-flood classes to background (255 -> 1, {0, 85, 170} -> 0).*
  2. *Treat Building as a separate foreground flood class.* (Rejected: violates task definition).
  3. *Mask out Sky as ignore_index.* (Considered, but sky is simply non-flood area).
* **Evidence:**
  - AIFloodSense paper (arXiv:2512.17432, Simantiris et al.): Class palette sets `255` as Flood water.
  - `flood_mamba_edge/utils/dataloader.py:157`: `mask = (mask_img == 255).astype(np.uint8)`.
* **Choice:** **Option 1**: Pixel value `255` is mapped to `1` (Flood); all pixels $< 255$ are mapped to `0` (Background).
* **Status:** **VERIFIED**.

---

## Decision 3: FloodKaggle Mask Value Sanitization

* **Context:** Aerial masks in the Kaggle Flood Area Segmentation dataset are stored as 8-bit images. JPEG compression or anti-aliased resizing can produce intermediate values between 0 and 255.
* **Options Considered:**
  1. *Strict equality `mask == 255`.* (Risk: may miss boundary pixels with values 250-254).
  2. *Thresholding at mid-point: `(mask >= 128).astype(np.uint8)`.*
* **Evidence:**
  - Standard computer vision practice for digitized binary masks.
  - `flood_mamba_edge/utils/dataloader.py:164`: `mask = (mask_img > 127).astype(np.uint8)`.
* **Choice:** **Option 2**: `(mask >= 128).astype(np.uint8)`.
* **Status:** **VERIFIED**.

---

## Decision 4: Shared Train/Val/Test Split Protocol

* **Context:** FloodKaggle does not provide an official competition split in its flat directory. AIFloodSense provides random image folders.
* **Options Considered:**
  1. *Random split dynamically at each run.* (Fatal flaw: breaks comparability across methods).
  2. *Fixed, pre-generated JSON split file saved to `splits/<dataset>_trainvaltest.json`.*
* **Evidence:**
  - Best practices in benchmark literature (Pascal VOC, Cityscapes, Medical Segmentation Decathlon).
  - Standard distribution: 70% Train, 15% Validation, 15% Test, seeded with `seed=42`.
* **Choice:** **Option 2**. All models evaluate on the identical test set and tune checkpoints strictly on the validation set.
* **Status:** **VERIFIED**.

---

## Decision 5: Semi-Supervised Labeled-Subset Nested Sampling

* **Context:** Sampling $10\%$, $25\%$, $50\%$ labeled subsets from the Train set.
* **Options Considered:**
  1. *Independent random sampling at each ratio.* (Risk: $10\%$ set might contain images not present in the $25\%$ set, introducing statistical noise).
  2. *Nested sampling:* $D_{10\%} \subset D_{25\%} \subset D_{50\%} \subset D_{\text{train}}$.
* **Evidence:**
  - Standard semi-supervised protocol (Sohn et al., FixMatch; Yang et al., UniMatch).
  - Guarantees monotonic increase of supervision information.
* **Choice:** **Option 2 (Nested Sampling)** with fixed seeds (Seeds: 42, 123, 456).
* **Status:** **VERIFIED**.

---

## Decision 6: Pure PyTorch Selective Scan Fallback

* **Context:** Official Mamba models (`mamba_ssm`, `causal_conv1d`) require compiled CUDA C++ extensions which frequently fail on Windows and restricted Colab environments.
* **Options Considered:**
  1. *Mandate compiled `mamba_ssm`.* (Risk: 80% of users cannot install or run the benchmark).
  2. *Integrate official Pure PyTorch chunked selective scan (`selective_scan_multi_input` / `selective_scan_pure_pytorch`).*
* **Evidence:**
  - `1S-MambaMatch/models/mivss.py:14-121`: Pure PyTorch chunked scan reproduces exact mathematical state transitions.
  - `flood_mamba_edge/models/mambafloodlite.py:59-125`: Pure PyTorch fallback achieves numerical parity within $1\text{e-}5$ tolerance.
* **Choice:** **Option 2**. Default to Pure PyTorch with auto-switch to CUDA kernel if available.
* **Status:** **VERIFIED**.

---

## Decision 7: Bug Correction in Official 1S-MambaMatch Engine

* **Context:** Line 83 in `1S-MambaMatch/engine/train.py` calls `unsupervised_loss(pred_u_s1, pred_u_w, tau=unsup_tau)` which raises `NameError: name 'unsupervised_loss' is not defined`.
* **Options Considered:**
  1. *Leave raw code as is.* (Causes fatal runtime crash).
  2. *Alias `unsupervised_loss = ASS_loss` in engine, as implemented in `engine/losses.py:6`.*
* **Evidence:**
  - `engine/losses.py` defines `ASS_loss(s_logits, w_logits, tau=0.4)`. The arguments and mathematical intent match line 83 exactly.
* **Choice:** **Option 2**. Fixed in the benchmark engine.
* **Status:** **VERIFIED**.

---

## Decision 8: Default Training Budget for Benchmarking

* **Context:** BSPC paper ran up to 30,000 iterations on large medical datasets. For aerial flood datasets (Train size $\approx 200$ images), 30,000 iterations with batch size 4 corresponds to 600 epochs, risking severe overfitting on tiny labeled subsets.
* **Options Considered:**
  1. *Enforce 30,000 iterations strictly.* (Overkill for fast benchmarking on 200 images, takes 8+ hours per run on CPU/T4).
  2. *Standardize on 40-50 epochs ($\approx 2,000 - 3,000$ iterations) with early stopping on validation flood-IoU, while providing `--epochs` and `--iterations` CLI flags to run full 30,000 iterations.*
* **Choice:** **Option 2**. Default to 40 epochs for benchmarking, with full 30k iteration profile available.
* **Status:** **UNVERIFIED** (needs empirical confirmation on whether 40 epochs reaches plateau vs 100 epochs). Documented in UNVERIFIED list.
