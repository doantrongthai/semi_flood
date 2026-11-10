from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from methods.registry import register_method
from methods.unet import DoubleConv, Down, Up, OutConv, dice_loss


class SSPModel(nn.Module):
    """
    Self-Support Prototype (SSP) Few-Shot network (Fan et al., ECCV 2022).
    Refines support prototype using query self-support representations.
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 2, alpha: float = 0.6):
        super().__init__()
        self.alpha = alpha
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # Prototype refinement convolution
        self.refine = nn.Sequential(
            nn.Conv2d(256 + 2, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up1 = Up(256 + 128, 128, bilinear=True)
        self.up2 = Up(128 + 64, 64, bilinear=True)
        self.up3 = Up(64 + 32, 32, bilinear=True)
        self.outc = OutConv(32, n_classes)

    def extract_features(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4

    def forward(
        self,
        q_img: torch.Tensor,
        s_img: torch.Tensor,
        s_mask: torch.Tensor,
    ) -> torch.Tensor:
        b = q_img.size(0)
        q1, q2, q3, q4 = self.extract_features(q_img)
        _, _, _, s4 = self.extract_features(s_img)

        s_mask_down = F.interpolate(s_mask.float(), size=s4.shape[-2:], mode="nearest")
        s_fg = (s_mask_down > 0).float()
        proto_s = (s4 * s_fg).sum(dim=(-2, -1), keepdim=True) / (s_fg.sum(dim=(-2, -1), keepdim=True) + 1e-6)

        # Initial query cosine similarity map
        cos_sim1 = F.cosine_similarity(q4, proto_s, dim=1).unsqueeze(1)  # (B, 1, H/8, W/8)

        # Self-support extraction: masked average pooling on initial query confident region
        q_fg = (cos_sim1 > 0.5).float()
        proto_q = (q4 * q_fg).sum(dim=(-2, -1), keepdim=True) / (q_fg.sum(dim=(-2, -1), keepdim=True) + 1e-6)

        # Refined prototype blending
        proto_refined = self.alpha * proto_s + (1.0 - self.alpha) * proto_q
        cos_sim2 = F.cosine_similarity(q4, proto_refined, dim=1).unsqueeze(1)

        sim_maps = torch.cat([cos_sim1, cos_sim2], dim=1)  # (B, 2, H/8, W/8)
        enriched = self.refine(torch.cat([q4, sim_maps], dim=1))

        x = self.up1(enriched, q3)
        x = self.up2(x, q2)
        x = self.up3(x, q1)
        logits = self.outc(x)
        return logits


@register_method("ssp")
class SSPMethod(BaseMethod):
    """SSP few-shot baseline."""
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.build_model()
        self.fixed_support_img: Optional[torch.Tensor] = None
        self.fixed_support_mask: Optional[torch.Tensor] = None

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-4))
        self.model = SSPModel(n_channels=3, n_classes=2, alpha=0.6).to(self.device)
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

        s_img, s_mask = images_l[:1], masks_l[:1]
        q_img, q_mask = images_l[1:], masks_l[1:]

        if self.fixed_support_img is None:
            self.fixed_support_img = s_img.detach().clone()
            self.fixed_support_mask = s_mask.detach().clone()

        if q_img.size(0) == 0:
            q_img = s_img
            q_mask = s_mask

        logits = self.model(q_img, s_img, s_mask)
        target_ce = q_mask.squeeze(1).long()
        loss = F.cross_entropy(logits, target_ce) + dice_loss(logits, q_mask)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.item())}

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            if self.fixed_support_img is None:
                s_img = images[:1]
                s_mask = torch.ones((1, 1, images.size(2), images.size(3)), device=self.device)
            else:
                s_img = self.fixed_support_img
                s_mask = self.fixed_support_mask
            return self.model(images, s_img, s_mask)
