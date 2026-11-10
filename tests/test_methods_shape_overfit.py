import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from methods.registry import get_method_class, list_available_methods
import methods  # Triggers registrations


def test_methods_shape_and_forward():
    device = torch.device("cpu")
    dummy_img = torch.randn(2, 3, 64, 64)
    dummy_mask = torch.randint(0, 2, (2, 1, 64, 64), dtype=torch.int64)
    batch_l = (dummy_img, dummy_mask, ["img1", "img2"])

    batch_u = (
        dummy_img,
        dummy_img + 0.1,
        dummy_img - 0.1,
        torch.zeros((2, 64, 64)),
        torch.zeros((2, 64, 64)),
        ["u1", "u2"],
    )

    available = list_available_methods()
    print(f"Testing {len(available)} registered methods: {available}")

    for m_name in available:
        cls = get_method_class(m_name)
        cfg = {"lr": 1e-3, "weight_decay": 1e-4, "total_steps": 100}
        method = cls(cfg, device)

        # 1. Forward shape test
        out = method.predict(dummy_img)
        assert out.shape == (2, 2, 64, 64), f"Method {m_name} output shape {out.shape} != (2, 2, 64, 64)"

        # 2. Train step test
        losses = method.train_step(batch_l, batch_u, step=1)
        assert "loss" in losses, f"Method {m_name} must return 'loss' in train_step output."
        assert not np.isnan(losses["loss"]), f"Method {m_name} produced NaN loss!"

        print(f"  [OK] Method '{m_name}' passed forward shape and train_step tests.")


def test_overfit_small_subset():
    device = torch.device("cpu")
    cls = get_method_class("unet")
    cfg = {"lr": 1e-3, "weight_decay": 0.0}
    method = cls(cfg, device)

    # Fixed single image overfit
    img = torch.randn(2, 3, 64, 64)
    mask = torch.zeros((2, 1, 64, 64), dtype=torch.int64)
    mask[:, :, 20:44, 20:44] = 1
    batch_l = (img, mask, ["ov1", "ov2"])

    initial_loss = method.train_step(batch_l, None, step=1)["loss"]
    for s in range(2, 20):
        losses = method.train_step(batch_l, None, step=s)

    final_loss = losses["loss"]
    assert final_loss < initial_loss, f"Overfit test failed: initial {initial_loss:.4f} -> final {final_loss:.4f}"
    print(f"[PASS] Overfit test passed: {initial_loss:.4f} -> {final_loss:.4f}")


if __name__ == "__main__":
    test_methods_shape_and_forward()
    test_overfit_small_subset()
    print("\n[ALL PASS] All methods successfully verified for forward shapes, loss validity, and overfitting ability!")
