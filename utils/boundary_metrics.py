from typing import Union
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch


def extract_binary_boundaries(mask: np.ndarray) -> np.ndarray:
    """Extracts boundary contour pixels via morphological gradient."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = (dilated - eroded) > 0
    return boundary


def _to_2d_bool_mask(m: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Helper to convert any tensor or ndarray prediction/target into a 2D boolean mask (H, W)."""
    if isinstance(m, torch.Tensor):
        if m.ndim == 4 and m.shape[1] == 2:
            arr = (m.argmax(dim=1) == 1).detach().cpu().numpy()[0]
        elif m.ndim == 4 and m.shape[1] == 1:
            arr = (m[:, 0] > 0.5).detach().cpu().numpy()[0]
        elif m.ndim == 3:
            arr = (m > 0.5).detach().cpu().numpy()[0]
        else:
            arr = (m > 0.5).detach().cpu().numpy()
    else:
        if m.ndim == 4 and m.shape[1] == 2:
            arr = (np.argmax(m, axis=1) == 1)[0]
        elif m.ndim == 4 and m.shape[1] == 1:
            arr = (m[:, 0] > 0.5)[0]
        elif m.ndim == 3:
            arr = (m > 0.5)[0]
        else:
            arr = (m > 0.5)

    if arr.ndim > 2:
        arr = arr.squeeze()
    return arr.astype(bool)


def compute_hd95(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    max_dist: float = 100.0,
) -> float:
    """
    Computes 95th percentile Hausdorff Distance (HD95) for binary segmentation.
    Returns 0.0 if both masks are empty; returns max_dist if one is empty.
    """
    p = _to_2d_bool_mask(pred)
    t = _to_2d_bool_mask(target)

    p_sum = np.sum(p)
    t_sum = np.sum(t)

    if p_sum == 0 and t_sum == 0:
        return 0.0
    if p_sum == 0 or t_sum == 0:
        return float(max_dist)

    b_pred = extract_binary_boundaries(p)
    b_target = extract_binary_boundaries(t)

    if not np.any(b_pred) and not np.any(b_target):
        return 0.0
    if not np.any(b_pred) or not np.any(b_target):
        return float(max_dist)

    dist_target = distance_transform_edt(~b_target)
    dist_pred = distance_transform_edt(~b_pred)

    d_p_to_t = dist_target[b_pred]
    d_t_to_p = dist_pred[b_target]

    if len(d_p_to_t) == 0 or len(d_t_to_p) == 0:
        return float(max_dist)

    hd95 = float(max(np.percentile(d_p_to_t, 95), np.percentile(d_t_to_p, 95)))
    return min(hd95, max_dist)


def compute_boundary_f1(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    theta: int = 2,
    eps: float = 1e-7,
) -> float:
    """
    Computes Boundary F1 (BF-score) within tolerance distance theta pixels.
    """
    p = _to_2d_bool_mask(pred)
    t = _to_2d_bool_mask(target)

    if not np.any(p) and not np.any(t):
        return 1.0
    if not np.any(p) or not np.any(t):
        return 0.0

    b_pred = extract_binary_boundaries(p)
    b_target = extract_binary_boundaries(t)

    if not np.any(b_pred) and not np.any(b_target):
        return 1.0
    if not np.any(b_pred) or not np.any(b_target):
        return 0.0

    dist_target = distance_transform_edt(~b_target)
    dist_pred = distance_transform_edt(~b_pred)

    precision = np.sum(dist_target[b_pred] <= theta) / (np.sum(b_pred) + eps)
    recall = np.sum(dist_pred[b_target] <= theta) / (np.sum(b_target) + eps)

    if precision + recall == 0:
        return 0.0
    f1 = float(2 * precision * recall / (precision + recall + eps))
    return f1
