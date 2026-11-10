import argparse
import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from data.binary_mask import sanitize_binary_mask
from data.datasets import (
    FloodLabeledDataset,
    FloodUnlabeledDataset,
    FloodValTestDataset,
    resolve_image_mask_dirs,
)
from data.download import download_and_extract
from data.splits import get_or_create_semi_split, get_or_create_trainvaltest_split
from engine.evaluator import evaluate_dataset
from engine.trainer import train_semi_supervised
import methods  # Register methods
from methods.registry import get_method_class, list_available_methods
from utils.complexity import compute_flops, count_parameters
from utils.seed import seed_worker, set_seed


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_single_experiment(
    dataset_name: str,
    method_name: str,
    ratio: float,
    seed: int,
    data_root: Optional[str] = None,
    output_root: Optional[str] = None,
    epochs: int = 40,
    batch_size: int = 4,
    size: int = 256,
    lr: float = 1e-4,
    smoke_test: bool = False,
    auto_download: bool = True,
):
    if data_root is None:
        data_root = os.path.join(_REPO_ROOT, "data_raw")
    if output_root is None:
        output_root = os.path.join(_REPO_ROOT, "results")
    print(f"\n{'='*75}")
    print(f"BENCHMARK RUN: Dataset={dataset_name.upper()} | Method={method_name.upper()} | Ratio={ratio*100:.0f}% | Seed={seed}")
    print(f"{'='*75}")

    set_seed(seed, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Check or download dataset
    ds_dir = os.path.join(data_root, dataset_name)
    if not os.path.exists(ds_dir) or len(os.listdir(ds_dir)) == 0:
        if auto_download:
            download_and_extract(dataset_name, root_dir=data_root)
        else:
            raise FileNotFoundError(f"Dataset directory '{ds_dir}' not found.")

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp", ".PNG", ".JPG"}
    has_subsplits = (
        os.path.isdir(os.path.join(ds_dir, "train"))
        and os.path.isdir(os.path.join(ds_dir, "val"))
        and os.path.isdir(os.path.join(ds_dir, "test"))
    )

    if has_subsplits:
        train_img_dir, train_mask_dir = resolve_image_mask_dirs(os.path.join(ds_dir, "train"))
        val_img_dir, val_mask_dir = resolve_image_mask_dirs(os.path.join(ds_dir, "val"))
        test_img_dir, test_mask_dir = resolve_image_mask_dirs(os.path.join(ds_dir, "test"))

        train_imgs = sorted([f for f in os.listdir(train_img_dir) if os.path.splitext(f)[1] in valid_exts])
        val_imgs = sorted([f for f in os.listdir(val_img_dir) if os.path.splitext(f)[1] in valid_exts])
        test_imgs = sorted([f for f in os.listdir(test_img_dir) if os.path.splitext(f)[1] in valid_exts])

        splits_dir = os.path.join(_REPO_ROOT, "splits")
        predefined = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
        split_info = get_or_create_trainvaltest_split(
            [], dataset_name, splits_dir=splits_dir, seed=42, predefined_split=predefined
        )
    else:
        splits_dir = os.path.join(_REPO_ROOT, "splits")
        img_dir, mask_dir = resolve_image_mask_dirs(ds_dir)
        train_img_dir, train_mask_dir = img_dir, mask_dir
        val_img_dir, val_mask_dir = img_dir, mask_dir
        test_img_dir, test_mask_dir = img_dir, mask_dir

        all_images = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1] in valid_exts])
        if len(all_images) == 0:
            raise RuntimeError(f"No image files found in '{img_dir}'.")
        split_info = get_or_create_trainvaltest_split(all_images, dataset_name, splits_dir=splits_dir, seed=42)

    train_images = split_info["train"]
    val_images = split_info["val"]
    test_images = split_info["test"]

    # 3. Semi-supervised Labeled/Unlabeled Split
    semi_info = get_or_create_semi_split(train_images, dataset_name, ratio=ratio, seed=seed, splits_dir=splits_dir)
    labeled_images = semi_info["labeled"]
    unlabeled_images = semi_info["unlabeled"]

    if smoke_test:
        labeled_images = labeled_images[:4]
        unlabeled_images = unlabeled_images[:4]
        val_images = val_images[:4]
        test_images = test_images[:4]
        epochs = 2

    print(f"[Data Partition] Total Train: {len(train_images)} | Labeled: {len(labeled_images)} | Unlabeled: {len(unlabeled_images)} | Val: {len(val_images)} | Test: {len(test_images)}")

    # 4. Construct Datasets & Loaders
    dataset_l = FloodLabeledDataset(train_img_dir, train_mask_dir, labeled_images, dataset_name, size=size, augment=True)
    dataset_val = FloodValTestDataset(val_img_dir, val_mask_dir, val_images, dataset_name, size=size)
    dataset_test = FloodValTestDataset(test_img_dir, test_mask_dir, test_images, dataset_name, size=size)

    dataset_u = None
    if len(unlabeled_images) > 0 and ratio < 1.0:
        dataset_u = FloodUnlabeledDataset(train_img_dir, unlabeled_images, size=size)

    loader_l = DataLoader(dataset_l, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, drop_last=False)
    loader_u = None
    if dataset_u is not None:
        loader_u = DataLoader(dataset_u, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, drop_last=False)

    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # 5. Initialize Method
    cls = get_method_class(method_name)
    config = {
        "lr": lr,
        "weight_decay": 1e-4,
        "epochs": epochs,
        "batch_size": batch_size,
        "total_steps": epochs * len(loader_l),
    }
    method = cls(config, device)

    # Complexity estimation
    eval_m = method.get_eval_model()
    comp_params = count_parameters(eval_m)
    comp_flops = compute_flops(eval_m, input_size=(1, 3, size, size))

    # 6. Train
    exp_name = f"{method_name}_r{int(ratio*100)}p_seed{seed}"
    exp_dir = os.path.join(output_root, dataset_name, exp_name)
    train_res = train_semi_supervised(
        method, loader_l, loader_u, loader_val, device, exp_dir, epochs=epochs, eval_every=1
    )

    # 7. Evaluate Best Checkpoint on Test Set
    best_ckpt = train_res["best_ckpt_path"]
    if os.path.exists(best_ckpt):
        method.load_checkpoint(best_ckpt)

    qual_dir = os.path.join(exp_dir, "qualitative_test_samples")
    test_metrics = evaluate_dataset(
        method,
        loader_test,
        device,
        compute_boundaries=True,
        save_qualitative_dir=qual_dir,
        method_name=method_name,
        max_qualitative_samples=5,
    )

    # 8. Record Final Result JSON
    result_record = {
        "dataset": dataset_name,
        "method": method_name,
        "ratio": ratio,
        "seed": seed,
        "epochs": epochs,
        "smoke_test": smoke_test,
        "best_epoch": train_res["best_epoch"],
        "best_val_iou": train_res["best_val_iou"],
        "training_time_sec": train_res["total_train_sec"],
        "complexity": {
            "total_params": comp_params["total_params"],
            "total_params_m": comp_params["total_params_m"],
            "gflops": comp_flops,
        },
        "test_metrics": test_metrics,
    }

    result_file = os.path.join(exp_dir, "final_result.json")
    with open(result_file, "w") as f:
        json.dump(result_record, f, indent=2)

    print(f"\n[Test Result] Flood-IoU: {test_metrics['flood_iou']:.4f} | Flood-Dice: {test_metrics['flood_dice']:.4f} | HD95: {test_metrics.get('hd95', 0.0):.2f} px | BF1: {test_metrics.get('boundary_f1', 0.0):.4f}")
    print(f"[Saved] Full results and qualitative error grids saved to '{exp_dir}'.\n")
    return result_record


def main():
    parser = argparse.ArgumentParser(description="Semi-Supervised Flood Segmentation Benchmark CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file (e.g. configs/floodkaggle.yaml)")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "floodkaggle", "aifloodsense"], help="Target dataset ('all' runs both floodkaggle and aifloodsense)")
    parser.add_argument("--ratio", type=float, default=None, help="Single labeled ratio, e.g. 0.10 (overrides --ratios)")
    parser.add_argument("--ratios", type=str, default="0.10,0.25,0.50,1.00", help="Comma-separated labeled ratios (default: '0.10,0.25,0.50,1.00')")
    parser.add_argument("--methods", type=str, default="unet", help="Comma-separated method names or 'all' (e.g. 'unet', 'fixmatch,unimatchv2')")
    parser.add_argument("--seed", type=int, default=None, help="Single seed, e.g. 42 (overrides --seeds)")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated random seeds (e.g. '42' or '1,2,3,4,5')")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--smoke_test", action="store_true", help="Runs 2-epoch smoke test on small subset for quick pipeline verification")
    parser.add_argument("--download", action="store_true", default=True, help="Automatically download missing datasets")
    args = parser.parse_args()

    # Determine target datasets
    if args.dataset == "all":
        target_datasets = ["floodkaggle", "aifloodsense"]
    else:
        target_datasets = [args.dataset]

    # Determine ratios list
    if args.ratio is not None:
        target_ratios = [float(args.ratio)]
    else:
        target_ratios = [float(r.strip()) for r in args.ratios.split(",")]

    # Determine seeds list
    if args.seed is not None:
        target_seeds = [int(args.seed)]
    else:
        target_seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # If --config is supplied, load and merge YAML config
    if args.config is not None:
        cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO_ROOT, args.config)
        if os.path.exists(cfg_path):
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            print(f"[Config] Loaded configuration from '{cfg_path}'.")
            if "dataset" in cfg and args.dataset == "all":
                target_datasets = [cfg["dataset"]]
            if "seed" in cfg and args.seed is None and args.seeds == "42":
                target_seeds = [int(cfg["seed"])]
            if "data" in cfg and isinstance(cfg["data"], dict) and "size" in cfg["data"]:
                args.size = cfg["data"]["size"]
            if "train" in cfg and isinstance(cfg["train"], dict):
                t_cfg = cfg["train"]
                if "epochs" in t_cfg and args.epochs == 40:
                    args.epochs = t_cfg["epochs"]
                if "batch_size" in t_cfg and args.batch_size == 4:
                    args.batch_size = t_cfg["batch_size"]
                if "lr" in t_cfg and args.lr == 1e-4:
                    args.lr = float(t_cfg["lr"])
        else:
            print(f"[Warning] Config file '{cfg_path}' not found. Using CLI defaults.")

    targets = list_available_methods() if args.methods.lower() == "all" else [m.strip() for m in args.methods.split(",")]
    
    print("\n" + "#" * 75)
    print(f"BENCHMARK PLAN: Datasets={target_datasets} | Methods={targets} | Ratios={target_ratios} | Seeds={target_seeds}")
    print("#" * 75 + "\n")

    for ds in target_datasets:
        for s in target_seeds:
            for r in target_ratios:
                for m in targets:
                    run_single_experiment(
                        dataset_name=ds,
                        method_name=m,
                        ratio=r,
                        seed=s,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        size=args.size,
                        lr=args.lr,
                        smoke_test=args.smoke_test,
                        auto_download=args.download,
                    )

    # Automatically generate tables and figures after all runs complete
    print("\n" + "=" * 75)
    print("[Post-Processing] Updating LaTeX Benchmark Tables and Figures...")
    print("=" * 75)
    try:
        from scripts.make_tables import main as generate_all_tables
        from scripts.make_figures import plot_label_efficiency_curve
    except ImportError:
        from make_tables import main as generate_all_tables
        from make_figures import plot_label_efficiency_curve

    try:
        generate_all_tables()
        plot_label_efficiency_curve()
        print("[Success] All tables and figures updated successfully!")
    except Exception as e:
        print(f"[Warning] Post-processing encountered an issue: {e}")

    try:
        import glob
        from IPython.display import Image, display
        qual_samples = glob.glob(os.path.join(_REPO_ROOT, "results", "*", "*", "qualitative_test_samples", "*.png"))
        if qual_samples:
            print("\n" + "=" * 75)
            print(f"[Visualization] Displaying test error map preview: {os.path.basename(qual_samples[0])}")
            print("=" * 75)
            display(Image(qual_samples[0]))
    except Exception:
        pass


if __name__ == "__main__":
    main()
