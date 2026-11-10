from typing import Any, Dict, List, Optional
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import compute_flood_metrics, SegmentationMetricsTracker
from utils.boundary_metrics import compute_hd95, compute_boundary_f1
from utils.visualization import save_qualitative_comparison


@torch.no_grad()
def evaluate_dataset(
    method: Any,
    dataloader: DataLoader,
    device: torch.device,
    compute_boundaries: bool = True,
    save_qualitative_dir: Optional[str] = None,
    method_name: str = "Method",
    max_qualitative_samples: int = 10,
) -> Dict[str, float]:
    """
    Evaluates a method across the target dataset split.
    All metrics are computed strictly on the target Flood class (foreground = 1).
    """
    eval_model = method.get_eval_model()
    eval_model.eval()

    tracker = SegmentationMetricsTracker()
    hd95_scores: List[float] = []
    bf1_scores: List[float] = []

    qualitative_saved = 0
    if save_qualitative_dir:
        os.makedirs(save_qualitative_dir, exist_ok=True)

    for batch in dataloader:
        # Standard format: (image, mask, img_name)
        images, masks, img_names = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = method.predict(images)

        # Batch-wise metrics
        for i in range(images.size(0)):
            pred_single = logits[i : i + 1]
            mask_single = masks[i : i + 1]
            img_name = img_names[i]

            m = compute_flood_metrics(pred_single, mask_single)
            tracker.update(m)

            if compute_boundaries:
                h95 = compute_hd95(pred_single, mask_single)
                b1 = compute_boundary_f1(pred_single, mask_single)
                hd95_scores.append(h95)
                bf1_scores.append(b1)

            # Save qualitative visual comparison if requested
            if save_qualitative_dir and qualitative_saved < max_qualitative_samples:
                # Convert normalized tensor (3, H, W) to uint8 RGB (H, W, 3)
                raw_np = (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
                gt_np = masks[i, 0].cpu().numpy().astype("uint8")
                pred_np = (pred_single[0].argmax(dim=0).cpu().numpy()).astype("uint8")

                out_path = os.path.join(save_qualitative_dir, f"{img_name}.png")
                save_qualitative_comparison(
                    raw_np, gt_np, pred_np, out_path, method_name=method_name, iou_score=m["flood_iou"]
                )
                qualitative_saved += 1

    summary = tracker.get_summary()
    if compute_boundaries and len(hd95_scores) > 0:
        import numpy as np
        summary["hd95"] = float(np.mean(hd95_scores))
        summary["hd95_std"] = float(np.std(hd95_scores))
        summary["boundary_f1"] = float(np.mean(bf1_scores))
        summary["boundary_f1_std"] = float(np.std(bf1_scores))

    return summary
