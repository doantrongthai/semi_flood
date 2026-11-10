import os
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image


def create_error_map(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Renders a color-coded error map:
      - Green  [0, 255, 0]: True Positive (Correct Flood)
      - Red    [255, 0, 0]: False Positive (False Alarm)
      - Blue   [0, 100, 255]: False Negative (Missed Flood)
      - Dark   [30, 30, 30]: True Negative (Background)
    """
    h, w = gt_mask.shape[:2]
    gt_bool = (gt_mask > 0)
    pred_bool = (pred_mask > 0)

    error_map = np.full((h, w, 3), 30, dtype=np.uint8)  # TN background

    tp = gt_bool & pred_bool
    fp = (~gt_bool) & pred_bool
    fn = gt_bool & (~pred_bool)

    error_map[tp] = [0, 220, 0]    # Green: TP
    error_map[fp] = [220, 0, 0]    # Red: FP
    error_map[fn] = [0, 120, 255]  # Orange/Blue: FN

    return error_map


def save_qualitative_comparison(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    method_name: str = "Method",
    iou_score: Optional[float] = None,
):
    """
    Generates a 4-panel comparison grid:
    [Raw UAV Image] | [Ground Truth] | [Predicted Mask] | [Error Map]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    h, w = image.shape[:2]

    # Convert grayscale GT/Pred to 3-channel RGB for display
    gt_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    gt_rgb[gt_mask > 0] = [0, 220, 220]  # Cyan

    pred_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    pred_rgb[pred_mask > 0] = [0, 220, 220]

    err_rgb = create_error_map(gt_mask, pred_mask)

    # Annotate panels with text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1

    def add_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 25), font, scale, (255, 255, 255), thick + 1, cv2.LINE_AA)
        cv2.putText(out, text, (10, 25), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
        return out

    p1 = add_label(image, "Input UAV Image")
    p2 = add_label(gt_rgb, "Ground Truth")
    score_txt = f"{method_name} (IoU: {iou_score:.3f})" if iou_score is not None else method_name
    p3 = add_label(pred_rgb, score_txt)
    p4 = add_label(err_rgb, "Error (G:TP, R:FP, B:FN)")

    grid = np.hstack([p1, p2, p3, p4])
    Image.fromarray(grid).save(save_path)
