import math
import torch
import torch.nn as nn


def compute_ema_decay(step: int, warmup_iters: int = 200, max_decay: float = 0.996) -> float:
    """
    Computes scheduled Exponential Moving Average decay factor.
    Ramps smoothly from 0.0 to max_decay over warmup_iters.
    """
    if step < warmup_iters:
        decay = (1.0 - math.exp(-float(step) / float(max(1, warmup_iters)))) * max_decay
    else:
        decay = max_decay
    return min(max_decay, max(0.0, decay))


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """
    Updates teacher model parameters via Polyak-Ruppert moving average:
    theta_teacher = decay * theta_teacher + (1 - decay) * theta_student
    """
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(decay).add_(param_s.data, alpha=1.0 - decay)

    # Copy running buffers (e.g. BatchNorm stats)
    for buf_s, buf_t in zip(student.buffers(), teacher.buffers()):
        buf_t.data.copy_(buf_s.data)
