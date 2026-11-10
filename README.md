# Semi-Supervised Flood Segmentation Benchmark for UAV & Aerial Imagery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/doantrongthai/semi_flood/blob/main/notebooks/colab_benchmark.ipynb)

A rigorous, reproducible semi-supervised segmentation benchmark framework designed for UAV and aerial flood disaster response. Built to Q1 / A* venue standards (*IEEE T-PAMI, IJCV, CVPR, ICCV, NeurIPS, BSPC*).

---

## 🌟 Key Features

- **Strict Binary Flood Segmentation**: Standardized binary framing with Channel 0 = Background, Channel 1 = Flood ($C=1$). Zero multi-class leakage.
- **Strict Spatial Normalization**: All images and ground-truth masks are strictly standardized to $256 \times 256$.
- **Two Real-World Aerial Flood Datasets**:
  - **FloodKaggle** (290 high-resolution aerial drone/helicopter flood images).
  - **AIFloodSense** (470 multi-sensor aerial images across diverse flood catastrophes; verified mask mapping `mask == 255` $\to$ Flood).
- **11 Baselines across 3 Benchmark Tracks**:
  - **Track A (Supervised Bounds)**: Standard U-Net, Mamba-UNet (evaluated at 10%, 25%, 50%, 100% labels).
  - **Track B (Semi-Supervised)**: Mean Teacher, Cross Teaching (CPS), FixMatch, UniMatch, UniMatch V2, Semi-Mamba-UNet.
  - **Track C (Few-Shot)**: PFENet, SSP, HMNet (1-shot support-query conditioning).
- **Pure PyTorch Portability**: Selective scan / State-Space Models (SSM) include vectorized pure-PyTorch fallbacks, guaranteeing seamless execution on Google Colab (Tesla T4) and Windows without requiring CUDA C++ compilation.
- **Zero Information Leakage**: Evaluated on strictly disjoint, deterministic 70/15/15 splits. Unlabeled loaders strictly reject ground-truth masks.
- **Extensible Architecture**: Drop-in registration for new methods via `@register_method` and `methods/template_new_method.py`.

---

## 📁 Repository Structure

```text
semi_flood_benchmark/
├── configs/                  # Benchmark YAML hyperparameters
├── data/                     # Dataset download, binary sanitization, splits, loaders
│   ├── download.py           # Automated gdown Google Drive downloader
│   ├── binary_mask.py        # Mask verification & binary normalization
│   ├── splits.py             # Deterministic 70/15/15 and SSL split generator
│   ├── augmentations.py      # Weak & strong geometric/photometric augmentations
│   └── datasets.py           # Leak-free PyTorch Dataset implementations
├── engine/                   # Unified training, EMA, and validation engine
│   ├── base_method.py        # Abstract BaseMethod class
│   ├── trainer.py            # Master semi-supervised training engine
│   ├── evaluator.py          # Validation and test evaluation pipeline
│   └── ema.py                # Polyak-Ruppert moving average scheduler
├── methods/                  # 11 Ported baseline implementations + template
│   ├── registry.py           # Method registration decorator factory
│   ├── unet.py               # Classical U-Net baseline
│   ├── mamba_unet.py         # Pure PyTorch Mamba-UNet
│   ├── mean_teacher.py       # Mean Teacher (EMA consistency)
│   ├── cross_teaching.py     # Cross Teaching / CPS (dual-branch pseudo-supervision)
│   ├── fixmatch.py           # FixMatch (confidence threshold pseudo-labeling)
│   ├── unimatch.py           # UniMatch (unified strong dual-stream perturbations)
│   ├── unimatch_v2.py        # UniMatch V2 (feature dropout & multi-head consistency)
│   ├── semi_mamba_unet.py    # Semi-Mamba-UNet (Mamba backbone + SSL)
│   ├── pfenet.py             # PFENet (Prior-guided feature enrichment)
│   ├── ssp.py                # SSP (Self-support prototype adaptation)
│   ├── hmnet.py              # HMNet (Hierarchical matching network)
│   └── template_new_method.py# Template to register custom novel methods
├── notebooks/
│   └── colab_benchmark.ipynb # Turnkey Google Colab notebook
├── scripts/
│   ├── run_benchmark.py      # CLI runner for training & evaluation
│   ├── make_tables.py        # Markdown, CSV, & LaTeX table generator
│   ├── make_figures.py       # Label-efficiency & convergence plotter
│   └── run_all.sh            # Master batch execution script
├── utils/                    # Metrics, boundary Hausdorff HD95, FLOPs profiler
├── docs/                     # Exported LaTeX tables & figures
│   ├── benchmark_tables.tex  # LaTeX booktabs table for paper submissions
│   └── figures/              # Generated publication figures
├── requirements.txt          # Python dependencies
└── setup.py                  # Package installer
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/doantrongthai/semi_flood.git
cd semi_flood
pip install -r requirements.txt
pip install -e .
```

### 2. Auto-Download Datasets

Downloads and extracts FloodKaggle and AIFloodSense directly from verified Google Drive archives:

```bash
python data/download.py --dataset all --root_dir data_raw
```

### 3. Run a Smoke-Test (2 Epochs)

Verify the complete training, checkpointing, and test evaluation pipeline:

```bash
python scripts/run_benchmark.py --dataset floodkaggle --ratio 0.10 --methods unet,fixmatch --seed 42 --smoke_test
```

### 4. Run Full Benchmark Experiments

Train baselines on FloodKaggle at 10% labeled ratio for 40 epochs:

```bash
python scripts/run_benchmark.py \
    --dataset floodkaggle \
    --ratio 0.10 \
    --methods unet,mamba_unet,mean_teacher,cross_teaching,fixmatch,unimatch,unimatchv2,semi_mamba_unet \
    --seed 42 \
    --epochs 40 \
    --batch_size 4 \
    --size 256
```

### 5. Generate LaTeX Tables and Plots

```bash
python scripts/make_tables.py
python scripts/make_figures.py
```

Generated LaTeX tables will be saved to `docs/benchmark_tables.tex` and `docs/table_<dataset>.tex`.

---

## 💡 Adding a Custom Method

Adding your own novel semi-supervised method requires only **one file** and **zero modifications** to the core benchmark engine:

1. Copy `methods/template_new_method.py` to `methods/my_new_method.py`.
2. Implement your model architecture and `train_step(labeled_batch, unlabeled_batch, current_epoch, total_epochs)`.
3. Register your class using the `@register_method("my_new_method")` decorator:

```python
from engine.base_method import BaseMethod
from methods.registry import register_method

@register_method("my_new_method")
class MyNewMethod(BaseMethod):
    def build_model(self):
        # Build your custom architecture
        pass

    def train_step(self, labeled_batch, unlabeled_batch, current_epoch, total_epochs):
        # Compute supervised + unsupervised loss
        return total_loss, {"sup_loss": sup_loss.item(), "unsup_loss": unsup_loss.item()}
```

4. Immediately run the benchmark:
```bash
python scripts/run_benchmark.py --dataset floodkaggle --ratio 0.10 --methods my_new_method
```

---

## 📊 Experimental Results

All benchmark metrics are populated dynamically from `results/` without human fabrication. Cells for unexecuted experiments display as `TBD (run pending)`.

### FloodKaggle Benchmark Summary (Flood Class)

| Track | Method | 10% Labeled | 25% Labeled | 50% Labeled | 100% Upper Bound |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Track A | **U-Net** | 0.5450 / 0.6938 | TBD | TBD | TBD |
| Track A | **Mamba-UNet** | TBD | TBD | TBD | TBD |
| Track B | **Mean Teacher** | TBD | TBD | TBD | TBD |
| Track B | **FixMatch** | TBD | TBD | TBD | TBD |
| Track B | **UniMatch V2** | TBD | TBD | TBD | TBD |
| Track B | **Semi-Mamba-UNet** | TBD | TBD | TBD | TBD |

---

## 📜 Citation

If you use this benchmark or codebase in your research, please cite:

```bibtex
@article{bspc2026mambamatch,
  title={Semi-Supervised Flood Segmentation in UAV Imagery via State-Space Consistency},
  journal={Biomedical Signal Processing and Control (BSPC)},
  year={2026}
}

@article{aifloodsense2025,
  title={AIFloodSense: A Global Benchmark Dataset for Aerial Flood Disaster Mapping},
  journal={arXiv preprint arXiv:2512.17432},
  year={2025}
}
```
