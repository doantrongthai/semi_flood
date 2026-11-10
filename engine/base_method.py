from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class BaseMethod(ABC, nn.Module):
    """
    Universal Base Class for all methods in the benchmark.
    Guarantees consistent execution across Supervised, Semi-Supervised, and Few-Shot tracks.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.num_classes = 2  # Channel 0: Background, Channel 1: Flood

    @abstractmethod
    def build_model(self) -> None:
        """Constructs model architectures, optimizers, and schedulers."""
        pass

    @abstractmethod
    def train_step(
        self,
        batch_labeled: Tuple[torch.Tensor, torch.Tensor, Any],
        batch_unlabeled: Optional[Tuple[torch.Tensor, ...]],
        step: int,
    ) -> Dict[str, float]:
        """
        Executes one forward + backward + optimizer step.
        Returns a dictionary of loss values for logging (e.g. {'loss': ..., 'loss_sup': ...}).
        """
        pass

    @abstractmethod
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Inference forward pass.
        Returns logits tensor of shape (B, 2, H, W).
        """
        pass

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Standard PyTorch forward mapping to predict()."""
        return self.predict(*args, **kwargs)

    def get_eval_model(self) -> nn.Module:
        """
        Returns the appropriate evaluation network (e.g., student or EMA teacher).
        Defaults to self.
        """
        return self

    def save_checkpoint(self, path: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
        """Saves model state and metadata."""
        ckpt = {
            "state_dict": self.state_dict(),
            "config": self.config,
            "extra_info": extra_info or {},
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str) -> None:
        """Restores model state from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
