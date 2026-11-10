import os
import random
import numpy as np
import torch
import warnings


def set_seed(seed: int = 42, strict: bool = False):
    """
    Sets random seeds across python, numpy, and PyTorch for reproducible benchmark runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if strict:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore", category=UserWarning)
    print(f"[Seed] Global seed set to {seed} (strict={strict})")


def seed_worker(worker_id: int):
    """
    Worker init function for PyTorch DataLoader to ensure worker-thread reproducibility.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
