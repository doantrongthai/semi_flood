import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from data.binary_mask import convert_floodkaggle_mask, convert_aifloodsense_mask, sanitize_binary_mask


def test_convert_floodkaggle_mask():
    # Test typical values: 0 (background), 255 (flood), 130 (near-edge), 120 (near-bg)
    raw = np.array([[0, 255], [120, 130]], dtype=np.uint8)
    bin_m = convert_floodkaggle_mask(raw)

    expected = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    assert np.array_equal(bin_m, expected), f"Expected {expected}, got {bin_m}"
    assert set(np.unique(bin_m)).issubset({0, 1})


def test_convert_aifloodsense_mask():
    # AIFloodSense: 0 (Bg), 85 (Building), 170 (Sky), 255 (Flood)
    raw = np.array([[0, 85], [170, 255]], dtype=np.uint8)
    bin_m = convert_aifloodsense_mask(raw)

    # Only 255 should be 1, everything else 0
    expected = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    assert np.array_equal(bin_m, expected), f"Expected {expected}, got {bin_m}"
    assert set(np.unique(bin_m)).issubset({0, 1})


def test_sanitize_binary_mask_validation():
    # Valid mask
    valid = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    out = sanitize_binary_mask(valid, "floodkaggle")
    assert np.array_equal(out, valid)


if __name__ == "__main__":
    test_convert_floodkaggle_mask()
    test_convert_aifloodsense_mask()
    test_sanitize_binary_mask_validation()
    print("[PASS] All binary mask conversion tests passed successfully!")
