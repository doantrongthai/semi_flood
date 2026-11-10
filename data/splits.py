import os
import json
import random
from typing import Dict, List, Optional, Tuple


def get_or_create_trainvaltest_split(
    image_names: List[str],
    dataset_name: str,
    splits_dir: str = "splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    predefined_split: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Retrieves or generates a deterministic Train/Val/Test split dictionary.
    If predefined_split is provided (from existing train/val/test folders), uses it directly.
    Otherwise generates a split using seed and specified ratios.
    """
    os.makedirs(splits_dir, exist_ok=True)
    split_file = os.path.join(splits_dir, f"{dataset_name.lower()}_trainvaltest.json")

    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            split_data = json.load(f)
        return split_data

    if predefined_split is not None:
        split_data = {
            "dataset": dataset_name,
            "seed": seed,
            "total_images": len(predefined_split["train"]) + len(predefined_split["val"]) + len(predefined_split["test"]),
            "train": sorted(predefined_split["train"]),
            "val": sorted(predefined_split["val"]),
            "test": sorted(predefined_split["test"]),
        }
        with open(split_file, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"[Splits] Recorded official dataset split to '{split_file}' (Train: {len(split_data['train'])}, Val: {len(split_data['val'])}, Test: {len(split_data['test'])}).")
        return split_data

    # Ensure deterministic starting order
    sorted_images = sorted(list(set(image_names)))
    rng = random.Random(seed)
    indices = list(range(len(sorted_images)))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    split_data = {
        "dataset": dataset_name,
        "seed": seed,
        "total_images": n_total,
        "train": [sorted_images[i] for i in sorted(train_idx)],
        "val": [sorted_images[i] for i in sorted(val_idx)],
        "test": [sorted_images[i] for i in sorted(test_idx)],
    }

    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"[Splits] Saved fixed train/val/test split to '{split_file}' (Train: {len(split_data['train'])}, Val: {len(split_data['val'])}, Test: {len(split_data['test'])}).")
    return split_data


def get_or_create_semi_split(
    train_images: List[str],
    dataset_name: str,
    ratio: float,
    seed: int,
    splits_dir: str = "splits",
) -> Dict[str, List[str]]:
    """
    Creates or loads nested semi-supervised labeled and unlabeled subsets from the train set.
    For ratios in [0.10, 0.25, 0.50, 1.00]:
      - Labeled subset is chosen deterministically using the specified seed.
      - Unlabeled subset = remaining train images.
      - Nested property: If parent 0.25 split exists for this seed, 0.10 is sampled as a strict subset of 0.25.
    """
    os.makedirs(splits_dir, exist_ok=True)
    ratio_str = f"{int(ratio * 100)}p" if ratio < 1.0 else "100p"
    semi_file = os.path.join(splits_dir, f"{dataset_name.lower()}_{ratio_str}_seed{seed}.json")

    if os.path.exists(semi_file):
        with open(semi_file, "r") as f:
            semi_data = json.load(f)
        return semi_data

    sorted_train = sorted(list(set(train_images)))
    n_train = len(sorted_train)
    n_labeled = max(1, int(round(n_train * ratio)))

    rng = random.Random(seed)
    shuffled_train = list(sorted_train)
    rng.shuffle(shuffled_train)

    labeled_set = sorted(shuffled_train[:n_labeled])
    unlabeled_set = sorted([img for img in sorted_train if img not in set(labeled_set)])

    semi_data = {
        "dataset": dataset_name,
        "ratio": ratio,
        "seed": seed,
        "n_train_total": n_train,
        "n_labeled": len(labeled_set),
        "n_unlabeled": len(unlabeled_set),
        "labeled": labeled_set,
        "unlabeled": unlabeled_set,
    }

    with open(semi_file, "w") as f:
        json.dump(semi_data, f, indent=2)

    print(f"[Splits] Saved semi split to '{semi_file}' ({len(labeled_set)} labeled, {len(unlabeled_set)} unlabeled).")
    return semi_data
