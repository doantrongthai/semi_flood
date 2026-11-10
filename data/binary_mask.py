import numpy as np


def convert_floodkaggle_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Converts raw FloodKaggle mask into strict binary flood mask {0, 1}.
    - 0: Background
    - 1: Flood
    If mask is already binary {0, 1}, preserves it.
    If mask is 8-bit [0, 255], thresholds at 128 to remove compression artifacts or anti-aliasing edges.
    """
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[:, :, 0]

    max_val = np.max(raw_mask) if raw_mask.size > 0 else 0
    if max_val <= 1:
        binary_mask = (raw_mask > 0).astype(np.uint8)
    else:
        binary_mask = (raw_mask >= 128).astype(np.uint8)
    return binary_mask


def convert_aifloodsense_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Converts raw AIFloodSense multi-class mask into strict binary flood mask {0, 1}.
    Official AIFloodSense classes:
      0: Background
      85: Building
      170: Sky
      255: Flood (Water)
    
    Mapping rule:
      Only pixel value 255 is classified as foreground Flood (1).
      All other pixels (0, 85, 170, and any intermediate values) are mapped to 0.
      If mask is already binary {0, 1}, preserves it.
    """
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[:, :, 0]

    max_val = np.max(raw_mask) if raw_mask.size > 0 else 0
    if max_val <= 1:
        binary_mask = (raw_mask > 0).astype(np.uint8)
    else:
        # Exactly 255 is Flood in the 4-class grayscale mapping
        binary_mask = (raw_mask == 255).astype(np.uint8)
    return binary_mask


def sanitize_binary_mask(mask: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Dispatches to the appropriate dataset-specific binary conversion routine
    and validates that the output mask contains strictly {0, 1} values.
    """
    ds = dataset_name.lower()
    if ds in ["floodkaggle", "flood_kaggle", "kaggle"]:
        bin_mask = convert_floodkaggle_mask(mask)
    elif ds in ["aifloodsense", "ai_floodsense", "floodscene"]:
        bin_mask = convert_aifloodsense_mask(mask)
    else:
        # Default fallback: threshold at 128
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        bin_mask = (mask >= 128).astype(np.uint8)

    uniques = set(np.unique(bin_mask))
    if not uniques.issubset({0, 1}):
        raise ValueError(f"Binary mask conversion failed: found unexpected values {uniques}")

    return bin_mask


def compute_flood_ratio(binary_mask: np.ndarray) -> float:
    """
    Computes the fraction of pixels classified as Flood in a binary mask {0, 1}.
    """
    if binary_mask.size == 0:
        return 0.0
    return float(np.sum(binary_mask == 1) / binary_mask.size)
