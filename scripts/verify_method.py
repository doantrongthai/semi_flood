import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from methods.registry import get_method_class, list_available_methods
from utils.complexity import count_parameters, compute_flops, measure_latency_fps
import methods  # Register methods


def verify_method(method_name: str, input_size: int = 256):
    print(f"\n{'='*70}")
    print(f"VERIFYING METHOD: {method_name.upper()}")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = get_method_class(method_name)
    config = {"lr": 1e-4, "weight_decay": 1e-4}

    method = cls(config, device)
    eval_model = method.get_eval_model()

    # 1. Parameter counts
    params = count_parameters(eval_model)
    print(f"[Complexity] Total Parameters: {params['total_params']:,} ({params['total_params_m']} M)")
    print(f"[Complexity] Trainable Parameters: {params['trainable_params']:,}")

    # 2. FLOPs computation
    flops = compute_flops(eval_model, input_size=(1, 3, input_size, input_size))
    print(f"[Complexity] Computational Cost: {flops} GFLOPs (Input: 1x3x{input_size}x{input_size})")

    # 3. Latency & FPS
    perf = measure_latency_fps(
        eval_model, input_size=(1, 3, input_size, input_size), device=str(device), warmup=5, repetitions=15
    )
    print(f"[Inference] Latency: {perf['latency_ms']} ms/frame")
    print(f"[Inference] Throughput: {perf['fps']} FPS (Device: {device})")
    if device.type == "cuda":
        print(f"[Inference] Peak VRAM: {perf['peak_vram_mb']} MB")

    # 4. Functional Forward & Backward Sanity Test
    method.train()
    img_l = torch.randn(2, 3, input_size, input_size, device=device)
    mask_l = torch.randint(0, 2, (2, 1, input_size, input_size), dtype=torch.int64, device=device)
    batch_l = (img_l, mask_l, ["test1", "test2"])

    img_u = torch.randn(2, 3, input_size, input_size, device=device)
    batch_u = (
        img_u,
        img_u + 0.05,
        img_u - 0.05,
        torch.zeros((2, input_size, input_size), device=device),
        torch.zeros((2, input_size, input_size), device=device),
        ["u1", "u2"],
    )

    losses = method.train_step(batch_l, batch_u, step=1)
    print(f"[Sanity] Forward/Backward Step Successful. Loss Dictionary: {losses}")
    print(f"[Status] Method '{method_name}' is VERIFIED and fully benchmark-ready.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="all", help="Method name or 'all'")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    targets = list_available_methods() if args.method == "all" else [args.method]
    for t in targets:
        verify_method(t, input_size=args.size)
