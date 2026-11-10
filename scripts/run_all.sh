#!/usr/bin/env bash
# ==============================================================================
# Master Execution Script for Full Semi-Supervised Flood Segmentation Benchmark
# Standard: Q1 / A* Venues (CVPR, ICCV, NeurIPS, BSPC, TPAMI)
# ==============================================================================

set -e

DATASETS=("floodkaggle" "aifloodsense")
RATIOS=(0.10 0.25 0.50 1.00)
SEEDS=(42 123 456)
METHODS="unet,mamba_unet,mean_teacher,cross_teaching,fixmatch,unimatch,unimatchv2,semi_mamba_unet,pfenet,ssp,hmnet"

echo "=============================================================================="
echo "Starting Full Benchmark Runs across Datasets, Ratios, and Seeds..."
echo "=============================================================================="

# Pre-download datasets
python data/download.py --dataset all

for dataset in "${DATASETS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "----------------------------------------------------------------------"
            echo "RUNNING: Dataset=${dataset} | Ratio=${ratio} | Seed=${seed}"
            echo "----------------------------------------------------------------------"
            python scripts/run_benchmark.py \
                --dataset "${dataset}" \
                --ratio "${ratio}" \
                --methods "${METHODS}" \
                --seed "${seed}" \
                --epochs 40 \
                --batch_size 4 \
                --size 256
        done
    done
done

echo "=============================================================================="
echo "Generating Final Benchmark Tables and Publication Figures..."
echo "=============================================================================="
python scripts/make_tables.py
python scripts/make_figures.py

echo "Benchmark execution finished successfully!"
