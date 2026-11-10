import os
import zipfile
import gdown

DATASET_REGISTRY = {
    "floodkaggle": {
        "id": "1uMgjTSxseBuxH65VhLdN_4_rlMihamHa",
        "dir": "floodkaggle",
        "archive": "floodkaggle.zip",
        "description": "Kaggle Flood Area Segmentation (290 aerial images and masks)",
    },
    "aifloodsense": {
        "id": "12f0UmV46uwzjMyYPpfc_S0fHLRe4LKIf",
        "dir": "aifloodsense",
        "archive": "aifloodsense.zip",
        "description": "AIFloodSense Global Aerial Imagery Dataset (470 high-resolution images)",
    },
}


def download_and_extract(dataset_name: str, root_dir: str = "data_raw"):
    """
    Downloads and extracts the dataset archive via gdown using verified Google Drive IDs.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}")

    cfg = DATASET_REGISTRY[dataset_name]
    target_dir = os.path.join(root_dir, cfg["dir"])

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"[Dataset] '{dataset_name}' already exists at '{target_dir}'. Skipping download.")
        return target_dir

    os.makedirs(root_dir, exist_ok=True)
    zip_path = os.path.join(root_dir, cfg["archive"])

    print(f"[Download] Fetching {dataset_name} ({cfg['description']}) from Google Drive ID: {cfg['id']}...")
    try:
        gdown.download(id=cfg["id"], output=zip_path, quiet=False)
    except Exception as e:
        print(f"[Download] Primary gdown ID method encountered an issue ({e}). Retrying with direct URL...")
        url = f"https://drive.google.com/uc?id={cfg['id']}"
        gdown.download(url, output=zip_path, quiet=False)

    if not os.path.exists(zip_path) or not zipfile.is_zipfile(zip_path):
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError(f"[Download Error] Downloaded file for '{dataset_name}' is not a valid zip archive.")

    print(f"[Extract] Extracting '{zip_path}' into '{target_dir}'...")
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"[Dataset] Extraction complete: '{target_dir}'.")
    return target_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "floodkaggle", "aifloodsense"])
    parser.add_argument("--root_dir", type=str, default="data_raw")
    args = parser.parse_args()

    targets = list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    for d in targets:
        download_and_extract(d, args.root_dir)
