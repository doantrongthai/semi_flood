from typing import Any, Dict, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from engine.ema import compute_ema_decay, ema_update
from methods.registry import register_method
from methods.unet import UNetModel, dice_loss


@register_method("mean_teacher")
class MeanTeacherMethod(BaseMethod):
    """
    Mean Teacher semi-supervised segmentation baseline (Tarvainen & Valpola, NeurIPS 2017).
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.build_model()

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-4))
        self.loss_u_weight = float(self.config.get("loss_u_weight", 0.5))
        self.warmup_iters = int(self.config.get("ema_warmup_iters", 200))
        self.max_decay = float(self.config.get("ema_max_decay", 0.996))

        # Student model
        self.student = UNetModel(n_channels=3, n_classes=2, bilinear=True).to(self.device)
        # Teacher model (evaluated on test)
        self.teacher = copy.deepcopy(self.student).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr, weight_decay=wd)

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

        # 1. Supervised step on student
        logits_l = self.student(images_l)
        target_ce = masks_l.squeeze(1).long()
        loss_sup = F.cross_entropy(logits_l, target_ce) + dice_loss(logits_l, masks_l)

        # 2. Unlabeled consistency step
        loss_u = torch.tensor(0.0, device=self.device)
        if batch_unlabeled is not None:
            img_u_w, img_u_s1, _, _, _, _ = batch_unlabeled
            img_u_w = img_u_w.to(self.device, non_blocking=True)
            img_u_s1 = img_u_s1.to(self.device, non_blocking=True)

            # Teacher prediction on weak view
            with torch.no_grad():
                logits_u_teacher = self.teacher(img_u_w)
                prob_u_teacher = F.softmax(logits_u_teacher, dim=1)

            # Student prediction on perturbed view
            logits_u_student = self.student(img_u_s1)
            prob_u_student = F.softmax(logits_u_student, dim=1)

            # MSE consistency loss
            loss_u = F.mse_loss(prob_u_student, prob_u_teacher) * self.loss_u_weight

        total_loss = loss_sup + loss_u

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        # 3. Update teacher weights via EMA
        decay = compute_ema_decay(step, warmup_iters=self.warmup_iters, max_decay=self.max_decay)
        ema_update(self.student, self.teacher, decay)

        return {
            "loss": float(total_loss.item()),
            "loss_sup": float(loss_sup.item()),
            "loss_u": float(loss_u.item()),
            "ema_decay": float(decay),
        }

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Inference evaluates the EMA Teacher model as specified by the Mean Teacher protocol."""
        self.teacher.eval()
        with torch.no_grad():
            return self.teacher(images)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
