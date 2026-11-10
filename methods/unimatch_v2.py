from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.augmentations import augment_feature_map
from engine.base_method import BaseMethod
from methods.registry import register_method
from methods.unet import UNetModel, dice_loss


@register_method("unimatchv2")
class UniMatchV2Method(BaseMethod):
    """
    UniMatch V2 semi-supervised baseline (Yang et al., TPAMI 2024).
    Unified learnable stream with dynamic confidence thresholding.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.build_model()

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-4))
        self.loss_u_weight = float(self.config.get("loss_u_weight", 0.5))
        self.total_steps = int(self.config.get("total_steps", 2000))

        self.model = UNetModel(n_channels=3, n_classes=2, bilinear=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

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

        # 1. Supervised loss
        logits_l = self.model(images_l)
        target_ce = masks_l.squeeze(1).long()
        loss_sup = F.cross_entropy(logits_l, target_ce) + dice_loss(logits_l, masks_l)

        # 2. Dynamic threshold calculation
        progress = min(1.0, float(step) / float(max(1, self.total_steps)))
        tau_dynamic = 0.85 + 0.10 * progress

        loss_u = torch.tensor(0.0, device=self.device)
        mask_ratio = 0.0

        if batch_unlabeled is not None:
            img_u_w, img_u_s1, _, cutmix_box1, _, _ = batch_unlabeled
            img_u_w = img_u_w.to(self.device, non_blocking=True)
            img_u_s1 = img_u_s1.to(self.device, non_blocking=True)
            cutmix_box1 = cutmix_box1.to(self.device, non_blocking=True)

            # Generate pseudo-labels from weak view
            with torch.no_grad():
                logits_u_w = self.model(img_u_w)
                prob_u_w = F.softmax(logits_u_w, dim=1)
                max_prob, pseudo_label = torch.max(prob_u_w, dim=1)
                conf_mask = (max_prob >= tau_dynamic).float()
                mask_ratio = float(conf_mask.mean().item())

            # Unified strong stream: CutMix + Feature Perturbation
            if cutmix_box1.ndim == 2:
                cutmix_box1 = cutmix_box1.unsqueeze(0)
            if cutmix_box1.ndim == 3:
                cutmix_box1 = cutmix_box1.unsqueeze(1)
            box = (cutmix_box1.expand_as(img_u_s1) == 1.0)
            img_u_s_mixed = img_u_s1.clone()
            img_u_s_mixed[box] = img_u_s1.flip(0)[box]

            # Forward with feature dropout
            x1 = self.model.inc(img_u_s_mixed)
            x2 = self.model.down1(x1)
            x3 = self.model.down2(x2)
            x4 = self.model.down3(x3)
            x5 = self.model.down4(x4)
            x5_pert = augment_feature_map(x5, p=0.3, noise_std=0.03)
            x = self.model.up1(x5_pert, x4)
            x = self.model.up2(x, x3)
            x = self.model.up3(x, x2)
            x = self.model.up4(x, x1)
            logits_u_s = self.model.outc(x)

            loss_ce_pixel = F.cross_entropy(logits_u_s, pseudo_label, reduction="none")

            if conf_mask.sum() > 0:
                loss_u = (loss_ce_pixel * conf_mask).sum() / (conf_mask.sum() + 1e-6)
                loss_u = loss_u * self.loss_u_weight

        total_loss = loss_sup + loss_u

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "loss_sup": float(loss_sup.item()),
            "loss_u": float(loss_u.item()),
            "mask_ratio": float(mask_ratio),
            "tau_dynamic": float(tau_dynamic),
        }

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(images)
