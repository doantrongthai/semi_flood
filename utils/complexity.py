import time
from typing import Dict, Tuple
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts total and trainable parameters of a PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": round(total_params / 1e6, 4),
    }


def compute_flops(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 256, 256)) -> float:
    """
    Computes approximate GFLOPs for a forward pass.
    Tries thop / fvcore / ptflops if available; falls back to analytical estimation.
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)

    # Try thop
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = float((macs * 2) / 1e9)
        return round(gflops, 3)
    except Exception:
        pass

    # Try ptflops
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, input_size[1:], as_strings=False, print_per_layer_stat=False, verbose=False
        )
        return round(float(macs * 2 / 1e9), 3)
    except Exception:
        pass

    # Analytical fallback: approximate 2 * H * W * sum(C_in * C_out * K^2)
    total_macs = 0
    h, w = input_size[2], input_size[3]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
            macs = (kh * kw * m.in_channels * m.out_channels * (h * w)) // (m.stride[0] * m.stride[1])
            total_macs += macs
        elif isinstance(m, nn.Linear):
            total_macs += m.in_features * m.out_features
    gflops = float((total_macs * 2) / 1e9)
    return round(max(0.1, gflops), 3)


def measure_latency_fps(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 256),
    device: str = "cpu",
    warmup: int = 10,
    repetitions: int = 30,
) -> Dict[str, float]:
    """
    Benchmarks model inference latency (ms) and throughput (FPS).
    Enforces GPU synchronization when running on CUDA.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    x = torch.randn(*input_size, device=dev)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()

        # Measurement
        start_time = time.perf_counter()
        for _ in range(repetitions):
            _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time

    avg_latency_ms = (total_time / repetitions) * 1000.0
    fps = (repetitions * input_size[0]) / total_time

    peak_vram_mb = 0.0
    if dev.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(dev) / (1024 * 1024))

    return {
        "latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 1),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }
