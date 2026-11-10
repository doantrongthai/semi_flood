import os
import json
import glob
from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_label_efficiency_curve(results_dir: str = None, out_path: str = None):
    """
    Plots Flood-IoU performance versus labeled ratio (10%, 25%, 50%, 100%) for all evaluated methods.
    """
    if results_dir is None:
        results_dir = os.path.join(_REPO_ROOT, "results")
    if out_path is None:
        out_path = os.path.join(_REPO_ROOT, "docs", "figures", "label_efficiency.png")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    records = defaultdict(lambda: defaultdict(list))
    pattern = os.path.join(results_dir, "*", "*", "final_result.json")

    for path in glob.glob(pattern):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            ds = data.get("dataset", "")
            m = data.get("method", "")
            r = round(float(data.get("ratio", 0.0)), 2)
            score = float(data.get("test_metrics", {}).get("flood_iou", 0.0))
            label_key = f"{m.upper()} ({ds.capitalize()})"
            records[label_key][r].append(score)
        except Exception:
            pass

    if len(records) == 0:
        print("[Plot] No completed experiment runs found in results/ yet. Skipping figure generation.")
        return

    plt.figure(figsize=(9, 6), dpi=300)
    plt.grid(True, linestyle="--", alpha=0.5)

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "X"]
    ratios = [0.10, 0.25, 0.50, 1.00]
    ratio_labels = ["10%", "25%", "50%", "100%"]

    for idx, (method_label, ratio_dict) in enumerate(sorted(records.items())):
        means = []
        valid_ratios = []
        for r in ratios:
            if r in ratio_dict and len(ratio_dict[r]) > 0:
                means.append(np.mean(ratio_dict[r]))
                valid_ratios.append(r)
        if len(means) >= 1:
            marker = markers[idx % len(markers)]
            linestyle = "-" if len(means) > 1 else ""
            plt.plot(valid_ratios, means, marker=marker, linestyle=linestyle, markersize=8, linewidth=2, label=method_label)

    plt.xlabel("Labeled Data Ratio", fontsize=12, fontweight="bold")
    plt.ylabel("Test Flood-IoU", fontsize=12, fontweight="bold")
    plt.title("Label Efficiency: Semi-Supervised Flood Segmentation Benchmark", fontsize=14, fontweight="bold")
    plt.xticks(ratios, ratio_labels)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved label efficiency curve to '{out_path}'.")


if __name__ == "__main__":
    plot_label_efficiency_curve()
