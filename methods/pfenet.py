from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from methods.registry import register_method
from methods.unet import DoubleConv, Down, dice_loss


def compute_prior_map(query_feat: torch.Tensor, support_feat: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
    """
    Training-Free Prior Mask calculation from PFENet (IEEE T-PAMI 2020).
    Calculates pixel-wise maximum cosine similarity between Query features
    and foreground Support features.
    query_feat: (B_q, C, H, W)
    support_feat: (B_s, C, H, W)
    support_mask: (B_s, 1, H, W)
    Returns: Prior map of shape (B_q, 1, H, W) normalized to [0, 1].
    """
    b_q, c, h, w = query_feat.shape
    b_s = support_feat.shape[0]

    if support_mask.shape[-2:] != (h, w):
        support_mask = F.interpolate(support_mask.float(), size=(h, w), mode="nearest")

    q_flat = F.normalize(query_feat.view(b_q, c, -1), p=2, dim=1)  # (B_q, C, HW)
    s_mask_fg = (support_mask > 0).view(b_s, 1, -1)  # (B_s, 1, HW)
    s_flat = F.normalize(support_feat.view(b_s, c, -1), p=2, dim=1)  # (B_s, C, HW)

    # Mask foreground support features
    s_flat = s_flat * s_mask_fg  # (B_s, C, HW)

    # Cosine correlation: (B_q, HW_q, HW_s)
    # Average across support batch if B_s > 1
    corr = torch.bmm(q_flat.permute(0, 2, 1), s_flat[0:1].expand(b_q, -1, -1))  # (B_q, HW_q, HW_s)
    prior, _ = torch.max(corr, dim=-1)  # (B_q, HW_q)
    prior = prior.view(b_q, 1, h, w)

    # Normalize to [0, 1]
    prior_min = prior.amin(dim=(-2, -1), keepdim=True)
    prior_max = prior.amax(dim=(-2, -1), keepdim=True)
    prior = (prior - prior_min) / (prior_max - prior_min + 1e-6)
    return prior


class FeatureEnrichmentModule(nn.Module):
    """
    Multi-Scale Feature Enrichment Module (FEM) from PFENet (IEEE T-PAMI 2020).
    Aggregates multi-scale context conditioned on the prior similarity map.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Input: feature + 1 prior map channel
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels + 1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Multi-scale dilated branches (dilations 1, 2, 4)
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d((out_channels // 2) * 3, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, q_feat: torch.Tensor, prior_map: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q_feat, prior_map], dim=1)
        x = self.conv_in(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return fused + x


class PFENetModel(nn.Module):
    """
    Prior Guided Feature Enrichment Network (PFENet) for Few-Shot Segmentation (IEEE T-PAMI 2020).
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 128, n_classes: int = 2):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, feature_dim)

        self.fem = FeatureEnrichmentModule(in_channels=feature_dim, out_channels=feature_dim)
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x

    def forward(
        self,
        support_img: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> torch.Tensor:
        out_size = (query_img.shape[2], query_img.shape[3])

        s_feat = self.extract_features(support_img)
        q_feat = self.extract_features(query_img)

        # 1. Training-free prior mask
        prior = compute_prior_map(q_feat, s_feat, support_mask)

        # 2. Multi-scale feature enrichment
        enriched = self.fem(q_feat, prior)

        # 3. Dense classification
        logits_low = self.classifier(enriched)
        logits = F.interpolate(logits_low, size=out_size, mode="bilinear", align_corners=True)
        return logits


@register_method("pfenet")
class PFENetMethod(BaseMethod):
    """
    PFENet: Prior Guided Feature Enrichment Network (IEEE T-PAMI 2020).
    Evaluated under the 1-Shot setting using labeled reference support samples.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.support_img: Optional[torch.Tensor] = None
        self.support_mask: Optional[torch.Tensor] = None
        self.build_model()

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-4))

        self.model = PFENetModel(in_channels=3, feature_dim=128, n_classes=2).to(self.device)
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

        # Cache reference support image from labeled training set for evaluation
        if self.support_img is None or (step % 50 == 0 and images_l.shape[0] > 0):
            self.support_img = images_l[0:1].detach().clone()
            self.support_mask = masks_l[0:1].detach().clone()

        s_img = images_l[0:1]
        s_mask = masks_l[0:1]
        q_img = images_l
        q_mask = masks_l

        logits_q = self.model(s_img, s_mask, q_img)

        target_ce = q_mask.squeeze(1).long()
        loss_ce = F.cross_entropy(logits_q, target_ce)
        loss_d = dice_loss(logits_q, q_mask)
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
            images = images.to(self.device, non_blocking=True)
            if self.support_img is None:
                s_img = images[0:1]
                s_mask = torch.zeros((1, 1, images.shape[2], images.shape[3]), device=self.device)
            else:
                s_img = self.support_img
                s_mask = self.support_mask

            return self.model(s_img, s_mask, images)
