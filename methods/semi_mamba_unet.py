from typing import Any, Dict, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from engine.ema import compute_ema_decay, ema_update
from methods.registry import register_method
from methods.mamba_unet import MambaUNetModel
from methods.unet import dice_loss


def adaptive_self_supervised_loss(
    s_logits: torch.Tensor,
    w_logits: torch.Tensor,
    tau: float = 0.4,
    b_weight: float = 0.3,
) -> torch.Tensor:
    """
    Adaptive Self-Supervised (ASS) Loss from BSPC 2026 paper (Algorithm 2).
    Evaluates teacher entropy; applies full cross-entropy to confident pixels
    and downweighted CE + student entropy regularization to uncertain boundary pixels.
    """
    w_prob = F.softmax(w_logits, dim=1)
    entropy_w = -torch.sum(w_prob * torch.log(w_prob + 1e-6), dim=1)  # (B, H, W)

    uncertain = (entropy_w > tau).float()
    adjust = torch.ones_like(uncertain)
    adjust[uncertain == 1.0] = b_weight

    pseudo = torch.argmax(w_logits, dim=1)
    ce = F.cross_entropy(s_logits, pseudo, reduction="none")

    s_prob = F.softmax(s_logits, dim=1)
    entropy_s = -torch.sum(s_prob * torch.log(s_prob + 1e-6), dim=1)

    loss = (ce * adjust + entropy_s * uncertain).mean()
    return loss


@register_method("semi_mamba_unet")
class SemiMambaUNetMethod(BaseMethod):
    """
    Semi-Supervised Mamba-UNet baseline equipped with EMA Teacher-Student and ASS Loss.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.build_model()

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-2))
        self.tau = float(self.config.get("tau", 0.4))
        self.b_weight = float(self.config.get("b_weight", 0.3))
        self.loss_u_scale = float(self.config.get("loss_u_scale", 0.2))
        self.warmup_iters = int(self.config.get("ema_warmup_iters", 200))
        self.max_decay = float(self.config.get("ema_max_decay", 0.996))

        self.student = MambaUNetModel(n_channels=3, n_classes=2).to(self.device)
        self.teacher = copy.deepcopy(self.student).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=wd)

    def train_step(
        self,
        batch_labeled: Tuple[torch.Tensor, torch.Tensor, Any],
        batch_unlabeled: Optional[Tuple[torch.Tensor, ...]],
        step: int,
    ) -> Dict[str, float]:
        self.student.train()
        self.teacher.eval()

        images_l, masks_l, _ = batch_labeled
        images_l = images_l.to(self.device, non_blocking=True)
        masks_l = masks_l.to(self.device, non_blocking=True)

        logits_l = self.student(images_l)
        target_ce = masks_l.squeeze(1).long()
        loss_sup = F.cross_entropy(logits_l, target_ce) + dice_loss(logits_l, masks_l)

        loss_u = torch.tensor(0.0, device=self.device)
        if batch_unlabeled is not None:
            img_u_w, img_u_s1, _, cutmix_box1, _, _ = batch_unlabeled
            img_u_w = img_u_w.to(self.device, non_blocking=True)
            img_u_s1 = img_u_s1.to(self.device, non_blocking=True)
            cutmix_box1 = cutmix_box1.to(self.device, non_blocking=True)

            with torch.no_grad():
                logits_u_w = self.teacher(img_u_w)

            if cutmix_box1.ndim == 2:
                cutmix_box1 = cutmix_box1.unsqueeze(0)
            if cutmix_box1.ndim == 3:
                cutmix_box1 = cutmix_box1.unsqueeze(1)
            box = (cutmix_box1.expand_as(img_u_s1) == 1.0)
            img_u_s1_mixed = img_u_s1.clone()
            img_u_s1_mixed[box] = img_u_s1.flip(0)[box]

            logits_u_s1 = self.student(img_u_s1_mixed)
            loss_u = adaptive_self_supervised_loss(
                logits_u_s1, logits_u_w, tau=self.tau, b_weight=self.b_weight
            ) * self.loss_u_scale

        total_loss = loss_sup + loss_u

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        decay = compute_ema_decay(step, warmup_iters=self.warmup_iters, max_decay=self.max_decay)
        ema_update(self.student, self.teacher, decay)

        return {
            "loss": float(total_loss.item()),
            "loss_sup": float(loss_sup.item()),
            "loss_u": float(loss_u.item()),
        }

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        with torch.no_grad():
            return self.teacher(images)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
