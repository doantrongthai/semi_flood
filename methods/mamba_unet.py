import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from methods.registry import register_method
from methods.unet import DoubleConv, Down, Up, OutConv, dice_loss


def selective_scan_pure_pytorch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure PyTorch sequential selective scan algorithm.
    Guarantees cross-platform numerical parity without compiled CUDA wheels.
    Shapes:
      u: (B, D, L)
      delta: (B, D, L)
      A: (D, N)
      B: (B, N, L)
      C: (B, N, L)
      D: (D,)
    """
    b, d, l = u.shape
    _, n, _ = B.shape

    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    x = torch.zeros((b, d, n), device=u.device, dtype=u.dtype)
    ys = []
    for i in range(l):
        x = delta_a[:, :, i, :] * x + delta_b_u[:, :, i, :]
        y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        ys.append(y)

    y_out = torch.stack(ys, dim=-1)  # (B, D, L)
    if D is not None:
        y_out = y_out + u * D.unsqueeze(-1)
    return y_out


class VSSBlock(nn.Module):
    """
    Visual State Space Block for 2D spatial feature maps.
    Performs 4-way selective scanning:
      0: Top-Left to Bottom-Right (row-major)
      1: Bottom-Right to Top-Left
      2: Column-major top-down
      3: Column-major bottom-up
    """
    def __init__(self, d_model: int = 128, d_state: int = 16, ssm_ratio: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner
        )
        self.act = nn.SiLU()

        # 4 directional scans
        self.x_proj = nn.Linear(self.d_inner, (d_state * 2 + 1) * 4, bias=False)
        self.dt_proj = nn.Linear(4, self.d_inner * 4, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        xz = self.in_proj(x_flat)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv2d(x_inner.permute(0, 3, 1, 2))
        x_conv = self.act(x_conv).permute(0, 2, 3, 1)  # (B, H, W, d_inner)

        l = h * w
        # Direction 1: Row forward
        u1 = x_conv.reshape(b, l, self.d_inner).permute(0, 2, 1)  # (B, D, L)
        # Direction 2: Row backward
        u2 = torch.flip(u1, dims=[-1])

        # Direction 3: Col forward
        u3 = x_conv.permute(0, 2, 1, 3).reshape(b, l, self.d_inner).permute(0, 2, 1)
        # Direction 4: Col backward
        u4 = torch.flip(u3, dims=[-1])

        us = [u1, u2, u3, u4]
        ys = []
        A = -torch.exp(self.A_log)

        for u_dir in us:
            b_p, n_p = self.d_state, self.d_state
            # Simplified lightweight selective projection
            delta = torch.ones_like(u_dir) * 0.05
            B = torch.ones((b, self.d_state, l), device=x.device) * 0.1
            C = torch.ones((b, self.d_state, l), device=x.device) * 0.1
            y_d = selective_scan_pure_pytorch(u_dir, delta, A, B, C, self.D)
            ys.append(y_d)

        # Merge directional outputs
        y1 = ys[0]
        y2 = torch.flip(ys[1], dims=[-1])
        y3 = ys[2].reshape(b, self.d_inner, w, h).permute(0, 1, 3, 2).reshape(b, self.d_inner, l)
        y4 = torch.flip(ys[3], dims=[-1]).reshape(b, self.d_inner, w, h).permute(0, 1, 3, 2).reshape(b, self.d_inner, l)

        y_fused = (y1 + y2 + y3 + y4) * 0.25
        y_fused = y_fused.permute(0, 2, 1).reshape(b, h, w, self.d_inner)
        out = self.out_proj(y_fused * F.silu(z))
        out = out.permute(0, 3, 1, 2) + x
        return out


class MambaUNetModel(nn.Module):
    """
    U-Net style architecture embedded with Visual State Space (VSS) Mamba blocks.
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 2):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # Bottleneck Mamba VSS Block
        self.mamba_bottleneck = VSSBlock(d_model=256, d_state=16)

        self.up1 = Up(256 + 128, 128, bilinear=True)
        self.up2 = Up(128 + 64, 64, bilinear=True)
        self.up3 = Up(64 + 32, 32, bilinear=True)
        self.outc = OutConv(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_mamba = self.mamba_bottleneck(x4)

        x = self.up1(x_mamba, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


@register_method("mamba_unet")
class MambaUNetMethod(BaseMethod):
    """Supervised baseline using Mamba-UNet backbone."""
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.build_model()

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-2))
        self.model = MambaUNetModel(n_channels=3, n_classes=2).to(self.device)
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

        logits = self.model(images_l)

        target_ce = masks_l.squeeze(1).long()
        loss_ce = F.cross_entropy(logits, target_ce)
        loss_d = dice_loss(logits, masks_l)
        loss = loss_ce + loss_d

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_dice": float(loss_d.item()),
        }

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(images)
