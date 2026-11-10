import os
import time
import json
import subprocess
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.evaluator import evaluate_dataset


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash if running in a git repo."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "non_git_or_uncommitted"


def train_semi_supervised(
    method: Any,
    train_loader_l: DataLoader,
    train_loader_u: Optional[DataLoader],
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    epochs: int = 40,
    eval_every: int = 1,
    patience: int = 15,
) -> Dict[str, Any]:
    """
    Standardized training routine for semi-supervised and supervised baselines.
    Enforces validation-based model checkpoint selection on flood_iou.
    """
    os.makedirs(output_dir, exist_ok=True)
    best_val_iou = -1.0
    best_epoch = 0
    patience_counter = 0

    log_history = []
    global_step = 0

    # Save initial run environment dump
    env_info = {
        "python_version": subprocess.check_output(["python", "--version"]).decode().strip(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": get_git_commit_hash(),
        "device": str(device),
        "epochs": epochs,
    }
    with open(os.path.join(output_dir, "environment_dump.json"), "w") as f:
        json.dump(env_info, f, indent=2)

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        method.train()
        epoch_losses: Dict[str, float] = {}
        n_batches = 0

        # Construct paired iterator
        if train_loader_u is not None:
            # Pair labeled and unlabeled streams
            loader_u_iter = iter(train_loader_u)
            loader_l_iter = iter(train_loader_l)
            # Cycle through the longer stream
            max_len = max(len(train_loader_l), len(train_loader_u))
        else:
            loader_l_iter = iter(train_loader_l)
            loader_u_iter = None
            max_len = len(train_loader_l)

        pbar = tqdm(range(max_len), desc=f"Epoch {epoch}/{epochs}", leave=False)
        for _ in pbar:
            global_step += 1
            # Retrieve labeled batch
            try:
                batch_l = next(loader_l_iter)
            except StopIteration:
                loader_l_iter = iter(train_loader_l)
                batch_l = next(loader_l_iter)

            # Retrieve unlabeled batch if semi-supervised
            batch_u = None
            if loader_u_iter is not None:
                try:
                    batch_u = next(loader_u_iter)
                except StopIteration:
                    loader_u_iter = iter(train_loader_u)
                    batch_u = next(loader_u_iter)

            losses = method.train_step(batch_l, batch_u, global_step)

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + float(v)
            n_batches += 1

            pbar.set_postfix({"loss": f"{losses.get('loss', 0.0):.4f}"})

        # Epoch average loss
        avg_losses = {k: round(v / max(1, n_batches), 4) for k, v in epoch_losses.items()}

        # Validation evaluation
        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics = evaluate_dataset(
                method, val_loader, device, compute_boundaries=False, method_name=method.__class__.__name__
            )
            val_iou = val_metrics["flood_iou"]
            val_dice = val_metrics["flood_dice"]

            record = {
                "epoch": epoch,
                "step": global_step,
                **avg_losses,
                "val_flood_iou": val_iou,
                "val_flood_dice": val_dice,
            }
            log_history.append(record)

            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] Loss: {avg_losses.get('loss', 0.0):.4f} | "
                f"Val Flood-IoU: {val_iou:.4f} | Val Flood-Dice: {val_dice:.4f}"
            )

            # Checkpoint selection on VAL flood_iou
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_epoch = epoch
                patience_counter = 0
                best_ckpt_path = os.path.join(output_dir, "best_val_model.pth")
                method.save_checkpoint(best_ckpt_path, extra_info={"val_metrics": val_metrics, "epoch": epoch})
                print(f"  --> Saved new best checkpoint (Val Flood-IoU: {val_iou:.4f}) to '{best_ckpt_path}'")
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= 20:
                    print(f"[Early Stopping] No improvement in validation flood-IoU for {patience} epochs. Terminating.")
                    break

    total_train_sec = round(time.time() - start_training_time, 2)

    # Save last model checkpoint
    last_ckpt_path = os.path.join(output_dir, "last_model.pth")
    method.save_checkpoint(last_ckpt_path, extra_info={"epoch": epoch, "last_val_metrics": val_metrics})

    # Save training logs
    with open(os.path.join(output_dir, "training_logs.json"), "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_iou": best_val_iou,
                "total_train_sec": total_train_sec,
                "history": log_history,
            },
            f,
            indent=2,
        )

    return {
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "total_train_sec": total_train_sec,
        "best_ckpt_path": os.path.join(output_dir, "best_val_model.pth"),
    }
