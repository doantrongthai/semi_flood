from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from methods.registry import register_method
from methods.mamba_unet import selective_scan_pure_pytorch
from methods.unet import DoubleConv, Down, Up, OutConv, dice_loss


class HybridMambaMatchingBlock(nn.Module):
    """
    Hybrid Mamba block for support-query sequence interaction (HMNet, NeurIPS 2024).
    """
    def __init__(self, d_model: int = 256, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, q_feat: torch.Tensor, s_feat: torch.Tensor, s_mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = q_feat.shape
        l = h * w

        # Masked support tokens
        s_mask_down = F.interpolate(s_mask.float(), size=(h, w), mode="nearest")
        s_masked = s_feat * s_mask_down

        q_flat = q_feat.permute(0, 2, 3, 1).reshape(b, l, c)  # (B, L, C)
        s_flat = s_masked.permute(0, 2, 3, 1).reshape(-1, l, c)  # (1, L, C)
        if s_flat.size(0) != b:
            s_flat = s_flat.expand(b, l, c)  # Expand 1 support across all B queries

        # Interleave support and query tokens along the sequence dimension: [S, Q]
        compound = torch.cat([s_flat, q_flat], dim=1)  # (B, 2L, C)
        compound = self.norm(compound)
        u = self.in_proj(compound).permute(0, 2, 1)  # (B, C, 2L)

        delta = torch.ones_like(u) * 0.05
        B = torch.ones((b, self.d_state, 2 * l), device=q_feat.device) * 0.1
        C = torch.ones((b, self.d_state, 2 * l), device=q_feat.device) * 0.1
        A = -torch.exp(self.A_log)

        y = selective_scan_pure_pytorch(u, delta, A, B, C, self.D)  # (B, C, 2L)
        y = y.permute(0, 2, 1)  # (B, 2L, C)
        y_q = y[:, l:, :]  # Extract query tokens
        out_q = self.out_proj(y_q).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return q_feat + out_q


class HMNetModel(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 2):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.mamba_matcher = HybridMambaMatchingBlock(d_model=256, d_state=16)

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

    def forward(self, q_img: torch.Tensor, s_img: torch.Tensor, s_mask: torch.Tensor) -> torch.Tensor:
        q1, q2, q3, q4 = self.extract_features(q_img)
        _, _, _, s4 = self.extract_features(s_img)

        enriched_q4 = self.mamba_matcher(q4, s4, s_mask)

        x = self.up1(enriched_q4, q3)
        x = self.up2(x, q2)
        x = self.up3(x, q1)
        logits = self.outc(x)
        return logits


@register_method("hmnet")
class HMNetMethod(BaseMethod):
    """HMNet few-shot baseline."""
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.build_model()
        self.fixed_support_img: Optional[torch.Tensor] = None
        self.fixed_support_mask: Optional[torch.Tensor] = None

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-2))
        self.model = HMNetModel(n_channels=3, n_classes=2).to(self.device)
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
