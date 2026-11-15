from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from methods.registry import register_method
from methods.unet import UNetModel, dice_loss


def generate_classmix_mask(pred_mask: torch.Tensor) -> torch.Tensor:
    """
    ClassMix mask generation for binary segmentation (WACV 2021).
    Selects semantically coherent object boundaries (Flood vs Background)
    instead of arbitrary rectangular CutMix patches.
    """
    b, h, w = pred_mask.shape
    masks = []
    for i in range(b):
        p = pred_mask[i]
        unique_classes = torch.unique(p).tolist()
        if len(unique_classes) > 1:
            # Randomly select one class (Foreground Flood or Background) to transplant
            chosen = unique_classes[torch.randint(0, len(unique_classes), (1,)).item()]
            m = (p == chosen).float()
        elif len(unique_classes) == 1:
            # If homogenous prediction, fallback to spatial half-split
            m = torch.zeros((h, w), device=pred_mask.device)
            m[:h // 2, :] = 1.0
        else:
            m = torch.zeros((h, w), device=pred_mask.device)
        masks.append(m)
    return torch.stack(masks, dim=0).unsqueeze(1)  # (B, 1, H, W)


@register_method("classmix")
class ClassMixMethod(BaseMethod):
    """
    ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning (WACV 2021).
    Mixes unlabelled samples using predicted segmentation masks to respect natural object boundaries.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.build_model()

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-2))
        self.loss_u_weight = float(self.config.get("loss_u_weight", 1.0))

        self.model = UNetModel(n_channels=3, n_classes=2, bilinear=True).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def train_step(
        self,
        batch_labeled: Tuple[torch.Tensor, torch.Tensor, Any],
        batch_unlabeled: Optional[Tuple[torch.Tensor, ...]],
        step: int,
    ) -> Dict[str, float]:
        self.model.train()
        images_l, masks_l, _ = batch_labeled
        images_l = images_l.to(self.device, non_blocking=True)
        masks_l = masks_l.to(self.device, non_blocking=True)

        # 1. Supervised step on labeled data
        logits_l = self.model(images_l)
        target_ce = masks_l.squeeze(1).long()
        loss_ce = F.cross_entropy(logits_l, target_ce)
        loss_d = dice_loss(logits_l, masks_l)
        loss_sup = loss_ce + loss_d

        # 2. ClassMix consistency on unlabeled data
        loss_u = torch.tensor(0.0, device=self.device)
        if batch_unlabeled is not None:
            images_u = batch_unlabeled[0].to(self.device, non_blocking=True)
            b = images_u.shape[0]

            with torch.no_grad():
                logits_u = self.model(images_u)
                pred_u = torch.argmax(logits_u, dim=1)  # (B, H, W)

                # Permute batch for pairing
                perm = torch.roll(torch.arange(b, device=self.device), shifts=1)
                images_u_perm = images_u[perm]
                pred_u_perm = pred_u[perm]

                # Semantic ClassMix mask
                class_mask = generate_classmix_mask(pred_u)  # (B, 1, H, W)
                images_mix = class_mask * images_u + (1.0 - class_mask) * images_u_perm
                target_mix = class_mask.squeeze(1) * pred_u.float() + (1.0 - class_mask.squeeze(1)) * pred_u_perm.float()
                target_mix = target_mix.long()

            logits_mix = self.model(images_mix)
            loss_u_ce = F.cross_entropy(logits_mix, target_mix)
            loss_u_dice = dice_loss(logits_mix, target_mix.unsqueeze(1))
            loss_u = loss_u_ce + loss_u_dice

        loss = loss_sup + self.loss_u_weight * loss_u

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "loss_sup": float(loss_sup.item()),
            "loss_u": float(loss_u.item()),
        }

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(images)
