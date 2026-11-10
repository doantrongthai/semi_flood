import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from utils.metrics import compute_flood_metrics
from utils.boundary_metrics import compute_hd95, compute_boundary_f1


def test_perfect_prediction():
    # 10x10 mask with flood in center 4x4
    gt = np.zeros((1, 10, 10), dtype=np.int64)
    gt[0, 3:7, 3:7] = 1

    pred = np.zeros((1, 2, 10, 10), dtype=np.float32)
    pred[0, 0] = 1.0  # Background logit
    pred[0, 1, 3:7, 3:7] = 10.0  # Flood logit high

    metrics = compute_flood_metrics(pred, gt)
    assert np.isclose(metrics["flood_iou"], 1.0)
    assert np.isclose(metrics["flood_dice"], 1.0)
    assert np.isclose(metrics["pixel_accuracy"], 1.0)

    hd95 = compute_hd95(pred, gt)
    bf1 = compute_boundary_f1(pred, gt)
    assert np.isclose(hd95, 0.0)
    assert np.isclose(bf1, 1.0)


def test_completely_empty_both():
    # Both ground truth and prediction are completely non-flooded (common in aerial tests)
    gt = np.zeros((1, 10, 10), dtype=np.int64)
    pred = np.zeros((1, 2, 10, 10), dtype=np.float32)
    pred[0, 0] = 5.0  # Predicts background everywhere

    metrics = compute_flood_metrics(pred, gt)
    assert np.isclose(metrics["flood_iou"], 1.0), "Empty-both agreement must yield IoU 1.0"
    assert np.isclose(metrics["flood_dice"], 1.0)

    hd95 = compute_hd95(pred, gt)
    assert np.isclose(hd95, 0.0)


def test_complete_miss():
    # GT has flood, pred has none
    gt = np.zeros((1, 10, 10), dtype=np.int64)
    gt[0, 2:8, 2:8] = 1

    pred = np.zeros((1, 2, 10, 10), dtype=np.float32)
    pred[0, 0] = 5.0

    metrics = compute_flood_metrics(pred, gt)
    assert np.isclose(metrics["flood_iou"], 0.0)
    assert np.isclose(metrics["flood_dice"], 0.0)


if __name__ == "__main__":
    test_perfect_prediction()
    test_completely_empty_both()
    test_complete_miss()
    print("[PASS] Metrics correctness and edge-case handling tests passed successfully!")
