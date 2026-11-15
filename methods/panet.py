from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from methods.registry import register_method
from methods.unet import DoubleConv, Down, dice_loss


def masked_average_pooling(feature: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Masked Average Pooling (MAP) from PANet (ICCV 2019).
    feature: (B, C, H, W)
    mask: (B, 1, H, W)
    Returns prototype vector of shape (C,)
    """
    if mask.shape[-2:] != feature.shape[-2:]:
        mask = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")

    mask_fg = (mask > 0).float()
    masked_feat = feature * mask_fg
    pixel_count = mask_fg.sum(dim=(0, 2, 3), keepdim=True).clamp(min=1.0)
    prototype = masked_feat.sum(dim=(0, 2, 3), keepdim=True) / pixel_count
    return prototype.squeeze()  # (C,)


class PANetEncoder(nn.Module):
    """Convolutional feature extractor for PANet."""
    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x


class PANetModel(nn.Module):
    """
    Prototype Alignment Network (PANet) for Few-Shot Segmentation (ICCV 2019).
    Computes class prototypes from support samples via MAP and segments query images
    via metric distance learning.
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 128, metric_scale: float = 20.0):
        super().__init__()
        self.encoder = PANetEncoder(in_channels=in_channels, feature_dim=feature_dim)
        self.metric_scale = metric_scale

    def forward_metric(
        self,
        q_feat: torch.Tensor,
        p_bg: torch.Tensor,
        p_fg: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Computes cosine similarity distance logits between query features and prototypes.
        """
        q_norm = F.normalize(q_feat, p=2, dim=1)
        p_bg_norm = F.normalize(p_bg, p=2, dim=0).view(1, -1, 1, 1)
        p_fg_norm = F.normalize(p_fg, p=2, dim=0).view(1, -1, 1, 1)

        sim_bg = torch.sum(q_norm * p_bg_norm, dim=1, keepdim=True) * self.metric_scale
        sim_fg = torch.sum(q_norm * p_fg_norm, dim=1, keepdim=True) * self.metric_scale

        logits_low = torch.cat([sim_bg, sim_fg], dim=1)  # (B, 2, H', W')
        logits = F.interpolate(logits_low, size=out_size, mode="bilinear", align_corners=True)
        return logits

    def forward_support_query(
        self,
        support_img: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes query prediction and Prototype Alignment Regularization (PAR).
        """
        out_size = (query_img.shape[2], query_img.shape[3])

        s_feat = self.encoder(support_img)
        q_feat = self.encoder(query_img)

        # 1. Support Prototypes via Masked Average Pooling
        p_fg = masked_average_pooling(s_feat, (support_mask > 0).float())
        p_bg = masked_average_pooling(s_feat, (support_mask == 0).float())

        # 2. Query prediction
        logits_query = self.forward_metric(q_feat, p_bg, p_fg, out_size)

        return logits_query, s_feat, q_feat, (p_bg, p_fg)


@register_method("panet")
class PANetMethod(BaseMethod):
    """
    PANet: Prototype Alignment Network for Few-Shot Segmentation (ICCV 2019).
    Evaluated under the 1-Shot setting using labeled reference samples.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.support_img: Optional[torch.Tensor] = None
        self.support_mask: Optional[torch.Tensor] = None
        self.build_model()

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-4))
        self.par_weight = float(self.config.get("par_weight", 0.5))

        self.model = PANetModel(in_channels=3, feature_dim=128).to(self.device)
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

        # In 1-shot episodic training, support is the first sample, query is the entire labeled batch
        s_img = images_l[0:1]
        s_mask = masks_l[0:1]
        q_img = images_l
        q_mask = masks_l

        logits_q, s_feat, q_feat, (p_bg, p_fg) = self.model.forward_support_query(s_img, s_mask, q_img)

        # Query Loss
        target_ce = q_mask.squeeze(1).long()
        loss_ce = F.cross_entropy(logits_q, target_ce)
        loss_d = dice_loss(logits_q, q_mask)
        loss_q = loss_ce + loss_d

        # Prototype Alignment Regularization (PAR): Query aligns to predict Support
        out_size_s = (s_img.shape[2], s_img.shape[3])
        p_fg_q = masked_average_pooling(q_feat, (q_mask > 0).float())
        p_bg_q = masked_average_pooling(q_feat, (q_mask == 0).float())
        logits_s = self.model.forward_metric(s_feat, p_bg_q, p_fg_q, out_size_s)

        target_ce_s = s_mask.squeeze(1).long()
        loss_par = F.cross_entropy(logits_s, target_ce_s)

        loss = loss_q + self.par_weight * loss_par

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "loss_query": float(loss_q.item()),
            "loss_par": float(loss_par.item()),
        }

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device, non_blocking=True)
            if self.support_img is None:
                # Fallback to self-support if predict is called before train_step
                s_img = images[0:1]
                s_mask = torch.zeros((1, 1, images.shape[2], images.shape[3]), device=self.device)
            else:
                s_img = self.support_img
                s_mask = self.support_mask

            logits_q, _, _, _ = self.model.forward_support_query(s_img, s_mask, images)
            return logits_q
