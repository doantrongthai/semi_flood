import random
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms


def obtain_cutmix_box(img_size: Tuple[int, int], p: float = 0.5) -> torch.Tensor:
    """
    Generates a binary CutMix box mask (H, W) where 1 indicates the region to replace.
    Returns zeros if probability check fails.
    """
    h, w = img_size
    box = torch.zeros((h, w), dtype=torch.float32)
    if random.random() > p:
        return box

    lam = np.random.beta(1.0, 1.0)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    box[bby1:bby2, bbx1:bbx2] = 1.0
    return box


def blur_image(image: Image.Image, p: float = 0.5) -> Image.Image:
    """Applies random Gaussian blur with radius between 0.1 and 2.0."""
    if random.random() < p:
        sigma = float(np.random.uniform(0.1, 2.0))
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    return image


class WeakAugmentation:
    """
    Deterministic/Seeded weak geometric transformations applied jointly to image & mask:
    - Random horizontal flip (p=0.5)
    - Random vertical flip (p=0.5)
    - Random slight rotation (-10 to 10 deg, p=0.5)
    - Resize to target (size, size)
    """
    def __init__(self, size: int = 256):
        self.size = size

    def __call__(self, img: Image.Image, mask: Image.Image = None):
        # Resize first to target canvas
        img = img.resize((self.size, self.size), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((self.size, self.size), Image.NEAREST)

        # Horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Vertical flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Slight rotation
        if random.random() > 0.5:
            deg = float(np.random.uniform(-10.0, 10.0))
            img = img.rotate(deg, Image.BILINEAR)
            if mask is not None:
                mask = mask.rotate(deg, Image.NEAREST)

        if mask is not None:
            return img, mask
        return img


class StrongAugmentation:
    """
    Strong photometric perturbations applied exclusively to images for student streams:
    - Strong Color Jitter (brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.8)
    - Random Grayscale (p=0.2)
    - Gaussian Blur (p=0.5)
    """
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25
        )
        self.random_gray = transforms.RandomGrayscale(p=0.2)

    def __call__(self, image: Image.Image) -> Image.Image:
        img_out = image.copy()
        if random.random() < 0.8:
            img_out = self.color_jitter(img_out)
        img_out = self.random_gray(img_out)
        img_out = blur_image(img_out, p=0.5)
        return img_out


def augment_feature_map(
    x: torch.Tensor, p: float = 0.1, noise_std: float = 0.05, shuffle_ratio: float = 0.25
) -> torch.Tensor:
    """
    Latent-level feature map perturbation (as used in 1S-MambaMatch / UniMatch).
    Adds Gaussian noise and randomly permutes a fraction of feature channels.
    """
    if random.random() > p:
        return x
    b, c, h, w = x.shape
    x = x + torch.randn_like(x) * noise_std
    num_shuffle = int(c * shuffle_ratio)
    if num_shuffle > 1:
        perm = torch.randperm(num_shuffle, device=x.device)
        shuffled = x[:, :num_shuffle][:, perm]
        x = torch.cat([shuffled, x[:, num_shuffle:]], dim=1)
    return x
