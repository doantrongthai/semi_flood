from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.base_method import BaseMethod
from methods.registry import register_method
from methods.unet import DoubleConv, Down, Up, OutConv, dice_loss


def feature_noise(x: torch.Tensor, uniform_range: float = 0.3) -> torch.Tensor:
    """F-Noise perturbation from CCT (CVPR 2020)."""
    noise = (torch.rand_like(x) - 0.5) * 2.0 * uniform_range
    return x + noise * x


def feature_drop(x: torch.Tensor, drop_rate: float = 0.5) -> torch.Tensor:
    """F-Drop perturbation from CCT (CVPR 2020)."""
    mask = (torch.rand_like(x) > drop_rate).float()
    return (x * mask) / (1.0 - drop_rate + 1e-8)


def spatial_drop(x: torch.Tensor, drop_rate: float = 0.3) -> torch.Tensor:
    """Spatial channel dropout perturbation from CCT (CVPR 2020)."""
    b, c, _, _ = x.shape
    mask = (torch.rand(b, c, 1, 1, device=x.device) > drop_rate).float()
    return (x * mask) / (1.0 - drop_rate + 1e-8)


class CCTDecoder(nn.Module):
    """Auxiliary / Main decoder module for CCT framework."""
    def __init__(self, in_channels: int = 512, n_classes: int = 2, bilinear: bool = True):
        super().__init__()
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(
        self,
        x5: torch.Tensor,
        x4: torch.Tensor,
        x3: torch.Tensor,
        x2: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class CCTModel(nn.Module):
    """
    Cross-Consistency Training (CCT) Model Architecture (CVPR 2020).
    Shared encoder + 1 Main Decoder + 3 Perturbed Auxiliary Decoders.
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 2, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        factor = 2 if bilinear else 1

        # Shared Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Main Decoder (trained with supervised labels & self-consistency anchor)
        self.main_decoder = CCTDecoder(in_channels=512, n_classes=n_classes, bilinear=bilinear)

        # Auxiliary Decoders (supervised via cross-consistency on perturbed representations)
        self.aux_decoder_noise = CCTDecoder(in_channels=512, n_classes=n_classes, bilinear=bilinear)
        self.aux_decoder_drop = CCTDecoder(in_channels=512, n_classes=n_classes, bilinear=bilinear)
        self.aux_decoder_spatial = CCTDecoder(in_channels=512, n_classes=n_classes, bilinear=bilinear)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference & evaluation pass using main decoder."""
        x1, x2, x3, x4, x5 = self.encode(x)
        return self.main_decoder(x5, x4, x3, x2, x1)

    def forward_train(
        self,
        x_l: torch.Tensor,
        x_u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Training pass:
        - Labeled input x_l decoded via main decoder.
        - Unlabeled input x_u decoded via main decoder and perturbed auxiliary decoders.
        """
        # 1. Labeled pass
        l1, l2, l3, l4, l5 = self.encode(x_l)
        logits_l = self.main_decoder(l5, l4, l3, l2, l1)

        # 2. Unlabeled pass through shared encoder
        u1, u2, u3, u4, u5 = self.encode(x_u)

        # Clean main prediction (anchor target)
        logits_u_main = self.main_decoder(u5, u4, u3, u2, u1)

        # Perturbed auxiliary predictions
        u5_noise = feature_noise(u5, uniform_range=0.3)
        logits_aux1 = self.aux_decoder_noise(u5_noise, u4, u3, u2, u1)

        u5_drop = feature_drop(u5, drop_rate=0.5)
        logits_aux2 = self.aux_decoder_drop(u5_drop, u4, u3, u2, u1)

        u5_spatial = spatial_drop(u5, drop_rate=0.3)
        logits_aux3 = self.aux_decoder_spatial(u5_spatial, u4, u3, u2, u1)

        return logits_l, logits_u_main, [logits_aux1, logits_aux2, logits_aux3]


@register_method("cct")
class CCTMethod(BaseMethod):
    """
    Cross-Consistency Training (CCT) Method (CVPR 2020).
    Enforces prediction consistency between a main decoder and multiple auxiliary decoders
    operating on perturbed feature representations.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.build_model()

    def build_model(self) -> None:
        lr = float(self.config.get("lr", 1e-4))
        wd = float(self.config.get("weight_decay", 1e-2))
        self.consistency_weight = float(self.config.get("consistency_weight", 1.0))
        self.warmup_epochs = int(self.config.get("warmup_epochs", 10))

        self.model = CCTModel(n_channels=3, n_classes=2, bilinear=True).to(self.device)
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

        if batch_unlabeled is not None:
            images_u = batch_unlabeled[0].to(self.device, non_blocking=True)
            logits_l, logits_u_main, aux_logits_list = self.model.forward_train(images_l, images_u)
        else:
            logits_l = self.model(images_l)
            logits_u_main = None
            aux_logits_list = []

        # Supervised Loss on Main Decoder
        target_ce = masks_l.squeeze(1).long()
        loss_ce = F.cross_entropy(logits_l, target_ce)
        loss_d = dice_loss(logits_l, masks_l)
        loss_sup = loss_ce + loss_d

        # Unsupervised Consistency Loss across Auxiliary Decoders
        loss_u = torch.tensor(0.0, device=self.device)
        if logits_u_main is not None and len(aux_logits_list) > 0:
            target_prob = F.softmax(logits_u_main.detach(), dim=1)
            cons_losses = []
            for aux_logits in aux_logits_list:
                aux_prob = F.softmax(aux_logits, dim=1)
                # Official CCT MSE consistency on softmax distributions
                cons_losses.append(F.mse_loss(aux_prob, target_prob))
            loss_u = torch.mean(torch.stack(cons_losses))

        loss = loss_sup + self.consistency_weight * loss_u

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "loss_sup": float(loss_sup.item()),
            "loss_u": float(loss_u.item()),
        }

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(images)
