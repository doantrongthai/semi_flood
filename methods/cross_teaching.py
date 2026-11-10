from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from methods.registry import register_method
from methods.unet import UNetModel, dice_loss


@register_method("cross_teaching")
class CrossTeachingMethod(BaseMethod):
    """
    Cross Teaching / Cross Pseudo Supervision (CPS) baseline (Chen et al., CVPR 2021).
    Maintains two parallel branches (Net 1 and Net 2) mutually cross-supervising each other.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.build_model()

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-4))
        self.cps_weight = float(self.config.get("cps_weight", 0.5))

        # Branch 1
        self.net1 = UNetModel(n_channels=3, n_classes=2, bilinear=True).to(self.device)
        # Branch 2 (different initialization)
        self.net2 = UNetModel(n_channels=3, n_classes=2, bilinear=True).to(self.device)

        params = list(self.net1.parameters()) + list(self.net2.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def train_step(
        self,
        batch_labeled: Tuple[torch.Tensor, torch.Tensor, Any],
        batch_unlabeled: Optional[Tuple[torch.Tensor, ...]],
        step: int,
    ) -> Dict[str, float]:
        self.net1.train()
        self.net2.train()

        images_l, masks_l, _ = batch_labeled
        images_l = images_l.to(self.device, non_blocking=True)
        masks_l = masks_l.to(self.device, non_blocking=True)
        target_ce = masks_l.squeeze(1).long()

        # 1. Supervised forward pass for both branches
        logits1_l = self.net1(images_l)
        logits2_l = self.net2(images_l)

        loss_sup1 = F.cross_entropy(logits1_l, target_ce) + dice_loss(logits1_l, masks_l)
        loss_sup2 = F.cross_entropy(logits2_l, target_ce) + dice_loss(logits2_l, masks_l)
        loss_sup = loss_sup1 + loss_sup2

        # 2. Cross Pseudo Supervision on Unlabeled Data
        loss_cps = torch.tensor(0.0, device=self.device)
        if batch_unlabeled is not None:
            img_u_w, _, _, _, _, _ = batch_unlabeled
            img_u_w = img_u_w.to(self.device, non_blocking=True)

            logits1_u = self.net1(img_u_w)
            logits2_u = self.net2(img_u_w)

            pseudo_label1 = logits1_u.detach().argmax(dim=1)
            pseudo_label2 = logits2_u.detach().argmax(dim=1)

            # Net1 is supervised by Net2's pseudo labels, and vice-versa
            loss_cps = (
                F.cross_entropy(logits1_u, pseudo_label2) + F.cross_entropy(logits2_u, pseudo_label1)
            ) * self.cps_weight

        total_loss = loss_sup + loss_cps

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "loss_sup": float(loss_sup.item()),
            "loss_cps": float(loss_cps.item()),
        }

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Ensemble average of Net1 and Net2 logits at inference."""
        self.net1.eval()
        self.net2.eval()
        with torch.no_grad():
            out1 = self.net1(images)
            out2 = self.net2(images)
            return (out1 + out2) * 0.5
