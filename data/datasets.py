import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.binary_mask import sanitize_binary_mask
from data.augmentations import WeakAugmentation, StrongAugmentation, obtain_cutmix_box


def resolve_image_mask_dirs(root_dir: str) -> Tuple[str, str]:
    """
    Finds the image and mask directory inside root_dir across common conventions.
    """
    candidates_img = ["images", "image", "Images", "train_images"]
    candidates_mask = ["masks", "mask", "labels", "label", "Masks", "Labels", "annotations"]

    img_dir = None
    for c in candidates_img:
        p = os.path.join(root_dir, c)
        if os.path.isdir(p):
            img_dir = p
            break
    if img_dir is None:
        img_dir = root_dir

    mask_dir = None
    for c in candidates_mask:
        p = os.path.join(root_dir, c)
        if os.path.isdir(p):
            mask_dir = p
            break
    if mask_dir is None:
        mask_dir = root_dir

    return img_dir, mask_dir


def load_raw_image(img_path: str) -> Image.Image:
    """Loads RGB image safely via PIL."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at '{img_path}'.")
    img = Image.open(img_path).convert("RGB")
    return img


def load_raw_mask(mask_path: str) -> np.ndarray:
    """Loads grayscale mask via OpenCV."""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found at '{mask_path}'.")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask image from '{mask_path}'.")
    return mask


class FloodLabeledDataset(Dataset):
    """
    Dataset for supervised and semi-supervised labeled images (D_l).
    Yields:
      img: FloatTensor (3, H, W) normalized to [0, 1]
      mask: LongTensor (1, H, W) with binary values {0, 1}
      img_name: str
    """
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        file_list: List[str],
        dataset_name: str,
        size: int = 256,
        augment: bool = True,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = sorted(list(file_list))
        self.dataset_name = dataset_name
        self.size = size
        self.augment = augment
        self.weak_aug = WeakAugmentation(size=size)
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.file_list)

    def _find_mask_path(self, img_name: str) -> str:
        base = os.path.splitext(img_name)[0]
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".bmp", ".PNG", ".JPG"]:
            cand = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(cand):
                return cand
        # Fallback to direct name
        cand = os.path.join(self.mask_dir, img_name)
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"Could not locate mask for image '{img_name}' in '{self.mask_dir}'.")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = self._find_mask_path(img_name)

        pil_img = load_raw_image(img_path)
        np_mask = load_raw_mask(mask_path)
        bin_mask = sanitize_binary_mask(np_mask, self.dataset_name)
        pil_mask = Image.fromarray(bin_mask)

        if self.augment:
            pil_img, pil_mask = self.weak_aug(pil_img, pil_mask)
        else:
            pil_img = pil_img.resize((self.size, self.size), Image.BILINEAR)
            pil_mask = pil_mask.resize((self.size, self.size), Image.NEAREST)

        img_tensor = self.to_tensor(pil_img)  # (3, H, W) in [0, 1]
        mask_np = np.array(pil_mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)

        return img_tensor, mask_tensor, img_name


class FloodUnlabeledDataset(Dataset):
    """
    Dataset for semi-supervised unlabeled images (D_u).
    STRICT ZERO-INFORMATION-LEAKAGE ENFORCEMENT:
      Does NOT accept mask directory or load any ground-truth mask.
    Yields:
      img_w: Weakly-augmented view FloatTensor (3, H, W)
      img_s1: Strongly-augmented view 1 FloatTensor (3, H, W)
      img_s2: Strongly-augmented view 2 FloatTensor (3, H, W)
      cutmix_box1: FloatTensor (H, W) binary box
      cutmix_box2: FloatTensor (H, W) binary box
      img_name: str
    """
    def __init__(
        self,
        img_dir: str,
        file_list: List[str],
        size: int = 256,
    ):
        self.img_dir = img_dir
        self.file_list = sorted(list(file_list))
        self.size = size
        self.weak_aug = WeakAugmentation(size=size)
        self.strong_aug = StrongAugmentation()
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        raw_img = load_raw_image(img_path)

        # 1. Weak geometric augmentation (resizing, flips, mild rotation)
        img_w = self.weak_aug(raw_img, mask=None)

        # 2. Two distinct strong photometric augmentations for student streams
        img_s1 = self.strong_aug(img_w)
        img_s2 = self.strong_aug(img_w)

        # 3. CutMix box coordinates
        cutmix_box1 = obtain_cutmix_box((self.size, self.size), p=0.5)
        cutmix_box2 = obtain_cutmix_box((self.size, self.size), p=0.5)

        t_w = self.to_tensor(img_w)
        t_s1 = self.to_tensor(img_s1)
        t_s2 = self.to_tensor(img_s2)

        return t_w, t_s1, t_s2, cutmix_box1, cutmix_box2, img_name


class FloodValTestDataset(Dataset):
    """
    Evaluation dataset for validation and test splits.
    Deterministic resizing without data augmentation.
    """
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        file_list: List[str],
        dataset_name: str,
        size: int = 256,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_list = sorted(list(file_list))
        self.dataset_name = dataset_name
        self.size = size
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.file_list)

    def _find_mask_path(self, img_name: str) -> str:
        base = os.path.splitext(img_name)[0]
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".bmp", ".PNG", ".JPG"]:
            cand = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(cand):
                return cand
        cand = os.path.join(self.mask_dir, img_name)
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"Could not locate mask for image '{img_name}' in '{self.mask_dir}'.")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = self._find_mask_path(img_name)

        pil_img = load_raw_image(img_path).resize((self.size, self.size), Image.BILINEAR)
        np_mask = load_raw_mask(mask_path)
        bin_mask = sanitize_binary_mask(np_mask, self.dataset_name)
        pil_mask = Image.fromarray(bin_mask).resize((self.size, self.size), Image.NEAREST)

        img_tensor = self.to_tensor(pil_img)
        mask_tensor = torch.from_numpy(np.array(pil_mask, dtype=np.int64)).unsqueeze(0)

        return img_tensor, mask_tensor, img_name
