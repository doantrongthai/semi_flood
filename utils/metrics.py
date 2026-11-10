from typing import Dict, Union
import numpy as np
import torch


def compute_binary_confusion_matrix(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, int]:
    """
    Computes TP, FP, TN, FN for the target Flood class (foreground = 1).
    Accepts:
      - Logits (B, 2, H, W) -> argmax == 1
      - Probabilities / Logits (B, 1, H, W) or (B, H, W) -> > threshold
      - Integer binary masks (B, H, W) with values {0, 1}
    """
    if isinstance(pred, torch.Tensor):
        if pred.ndim == 4 and pred.shape[1] == 2:
            pred_flood = (pred.argmax(dim=1) == 1).detach().cpu().numpy().astype(bool)
        elif pred.ndim == 4 and pred.shape[1] == 1:
            pred_flood = (torch.sigmoid(pred[:, 0]) > threshold).detach().cpu().numpy().astype(bool)
        else:
            pred_flood = (pred > threshold).detach().cpu().numpy().astype(bool)
    else:
        if pred.ndim == 4 and pred.shape[1] == 2:
            pred_flood = (np.argmax(pred, axis=1) == 1).astype(bool)
        else:
            pred_flood = (pred > threshold).astype(bool)

    if isinstance(target, torch.Tensor):
        target_flood = (target > 0).detach().cpu().numpy().astype(bool)
    else:
        target_flood = (target > 0).astype(bool)

    if target_flood.ndim == 4 and target_flood.shape[1] == 1:
        target_flood = target_flood[:, 0]
    if pred_flood.ndim == 4 and pred_flood.shape[1] == 1:
        pred_flood = pred_flood[:, 0]

    tp = int(np.sum(pred_flood & target_flood))
    fp = int(np.sum(pred_flood & (~target_flood)))
    fn = int(np.sum((~pred_flood) & target_flood))
    tn = int(np.sum((~pred_flood) & (~target_flood)))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_flood_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Comprehensive Flood-class segmentation metrics:
    - flood_iou: IoU on Flood class (foreground = 1)
    - flood_dice: Dice / F1 score on Flood class
    - flood_precision: Precision on Flood class
    - flood_recall: Recall (Sensitivity) on Flood class
    - pixel_accuracy: Overall pixel accuracy
    - bg_iou: Background class IoU
    - mean_iou_2class: Average of {background_iou, flood_iou} for BSPC table compatibility
    """
    cm = compute_binary_confusion_matrix(pred, target)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

    total_pixels = tp + fp + fn + tn
    pixel_acc = float((tp + tn) / max(1, total_pixels))

    # Boundary handling for completely non-flooded images
    if (tp + fp + fn) == 0:
        # Both ground truth and prediction are empty: perfect agreement on negative
        flood_iou = 1.0
        flood_dice = 1.0
        flood_prec = 1.0
        flood_rec = 1.0
    else:
        flood_iou = float(tp / (tp + fp + fn + eps))
        flood_dice = float((2 * tp) / (2 * tp + fp + fn + eps))
        flood_prec = float(tp / (tp + fp + eps))
        flood_rec = float(tp / (tp + fn + eps))

    # Background IoU
    if (tn + fp + fn) == 0:
        bg_iou = 1.0
    else:
        bg_iou = float(tn / (tn + fp + fn + eps))

    mean_iou_2class = float((flood_iou + bg_iou) / 2.0)

    return {
        "flood_iou": flood_iou,
        "flood_dice": flood_dice,
        "flood_precision": flood_prec,
        "flood_recall": flood_rec,
        "pixel_accuracy": pixel_acc,
        "bg_iou": bg_iou,
        "mean_iou_2class": mean_iou_2class,
    }


class SegmentationMetricsTracker:
    """
    Accumulates per-image metrics and computes dataset-level macro averages and standard errors.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.records = {
            "flood_iou": [],
            "flood_dice": [],
            "flood_precision": [],
            "flood_recall": [],
            "pixel_accuracy": [],
            "mean_iou_2class": [],
        }

    def update(self, metrics: Dict[str, float]):
        for k in self.records.keys():
            if k in metrics:
                self.records[k].append(float(metrics[k]))

    def get_summary(self) -> Dict[str, float]:
        summary = {}
        for k, v in self.records.items():
            arr = np.array(v, dtype=np.float64)
            summary[k] = float(np.mean(arr)) if len(arr) > 0 else 0.0
            summary[f"{k}_std"] = float(np.std(arr)) if len(arr) > 0 else 0.0
        return summary
