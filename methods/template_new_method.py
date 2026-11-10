"""
Template for Adding a New Method to the Semi-Supervised Flood Segmentation Benchmark.

Instructions:
1. Copy this template to `methods/my_new_method.py`.
2. Implement `build_model`, `train_step`, and `predict`.
3. Register using `@register_method("my_new_method")`.
4. Create a corresponding config at `configs/my_new_method.yaml`.
5. Run benchmark:
   python scripts/run_benchmark.py --dataset floodkaggle --ratio 0.1 --methods my_new_method --seed 42
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from methods.registry import register_method
from methods.unet import UNetModel, dice_loss


@register_method("template_new_method")
class TemplateNewMethod(BaseMethod):
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.build_model()

    def build_model(self) -> None:
        """
        Initialize network architecture, loss criteria, optimizers, and schedulers.
        Read hyperparameters from self.config.
        """
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-4))

        # Example: Using UNetModel as backbone (replace with your novel architecture)
        self.model = UNetModel(n_channels=3, n_classes=2, bilinear=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

    def train_step(
        self,
        batch_labeled: Tuple[torch.Tensor, torch.Tensor, Any],
        batch_unlabeled: Optional[Tuple[torch.Tensor, ...]],
        step: int,
    ) -> Dict[str, float]:
        """
        Executes one training iteration.
        Args:
          batch_labeled: (images_l, masks_l, img_names_l)
          batch_unlabeled: (img_u_w, img_u_s1, img_u_s2, box1, box2, img_names_u) or None
          step: Current global iteration index
        Returns:
          Dictionary of scalar float loss values. Must include key 'loss'.
        """
        self.model.train()
        images_l, masks_l, _ = batch_labeled
        images_l = images_l.to(self.device, non_blocking=True)
        masks_l = masks_l.to(self.device, non_blocking=True)

        # 1. Supervised stream
        logits_l = self.model(images_l)
        target_ce = masks_l.squeeze(1).long()
        loss_sup = F.cross_entropy(logits_l, target_ce) + dice_loss(logits_l, masks_l)

        # 2. Novel unsupervised stream (add your custom consistency, prototype, or gating logic here)
        loss_u = torch.tensor(0.0, device=self.device)
        if batch_unlabeled is not None:
            img_u_w, img_u_s1, img_u_s2, box1, box2, _ = batch_unlabeled
            img_u_w = img_u_w.to(self.device, non_blocking=True)
            # Add custom unsupervised loss computation here
            pass

        total_loss = loss_sup + loss_u

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "loss_sup": float(loss_sup.item()),
            "loss_u": float(loss_u.item()),
        }

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Inference prediction forward pass.
        Args:
          images: FloatTensor (B, 3, H, W) in [0, 1]
        Returns:
          logits: FloatTensor (B, 2, H, W)
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(images)
