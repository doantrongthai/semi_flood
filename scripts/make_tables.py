import os
import json
import glob
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np


ALL_METHODS = [
    # Track A: Supervised
    ("Track A: Supervised", "unet", "U-Net"),
    ("Track A: Supervised", "mamba_unet", "Mamba-UNet"),
    # Track B: Semi-Supervised
    ("Track B: Semi-Supervised", "mean_teacher", "Mean Teacher"),
    ("Track B: Semi-Supervised", "cross_teaching", "Cross Teaching (CPS)"),
    ("Track B: Semi-Supervised", "fixmatch", "FixMatch"),
    ("Track B: Semi-Supervised", "unimatch", "UniMatch"),
    ("Track B: Semi-Supervised", "unimatchv2", "UniMatch V2"),
    ("Track B: Semi-Supervised", "semi_mamba_unet", "Semi-Mamba-UNet"),
    # Track C: Few-Shot
    ("Track C: Few-Shot", "pfenet", "PFENet"),
    ("Track C: Few-Shot", "ssp", "SSP"),
    ("Track C: Few-Shot", "hmnet", "HMNet"),
]

RATIOS = [0.10, 0.25, 0.50, 1.00]


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_all_results(results_dir: str = None) -> Dict[str, Dict[str, Dict[float, List[Dict[str, Any]]]]]:
    """
    Loads all result JSON files from results/<dataset>/<exp_name>/final_result.json.
    Structure: results[dataset][method][ratio] = [record_seed1, record_seed2, ...]
    """
    if results_dir is None:
        results_dir = os.path.join(_REPO_ROOT, "results")
    records = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pattern = os.path.join(results_dir, "*", "*", "final_result.json")

    for path in glob.glob(pattern):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            ds = data.get("dataset", "").lower()
            method = data.get("method", "").lower()
            ratio = round(float(data.get("ratio", 0.0)), 2)
            records[ds][method][ratio].append(data)
        except Exception as e:
            print(f"[Warning] Failed to read result at '{path}': {e}")

    return records


def format_score(records_list: List[Dict[str, Any]], metric_key: str) -> str:
    """Formats score as mean±std or 'TBD (run pending)'."""
    if not records_list:
        return "TBD (run pending)"

    vals = []
    for r in records_list:
        m = r.get("test_metrics", {})
        if metric_key in m:
            vals.append(float(m[metric_key]))

    if len(vals) == 0:
        return "TBD (run pending)"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"


def generate_markdown_tables(records: dict, dataset_name: str) -> str:
    ds_records = records.get(dataset_name.lower(), {})
    lines = []
    lines.append(f"\n### Main Semi-Supervised Benchmark Table: {dataset_name.upper()} (Flood Class)")
    lines.append("| Track | Method | 10% Labeled (IoU / Dice) | 25% Labeled (IoU / Dice) | 50% Labeled (IoU / Dice) | 100% Upper Bound (IoU / Dice) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")

    current_track = ""
    for track, m_key, m_name in ALL_METHODS:
        if track != current_track:
            current_track = track
        
        row_cols = []
        for r in RATIOS:
            r = round(r, 2)
            r_list = ds_records.get(m_key, {}).get(r, [])
            iou_str = format_score(r_list, "flood_iou")
            dice_str = format_score(r_list, "flood_dice")
            if "TBD" in iou_str:
                cell = "TBD (run pending)"
            else:
                cell = f"{iou_str} / {dice_str}"
            row_cols.append(cell)

        lines.append(f"| {track} | **{m_name}** | {row_cols[0]} | {row_cols[1]} | {row_cols[2]} | {row_cols[3]} |")

    return "\n".join(lines)


def generate_latex_tables(records: dict, dataset_name: str, out_tex_path: str):
    if not os.path.isabs(out_tex_path):
        out_tex_path = os.path.join(_REPO_ROOT, out_tex_path)
    ds_records = records.get(dataset_name.lower(), {})
    tex = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(rf"\caption{{Semi-Supervised Flood Segmentation Benchmark on \textbf{{{dataset_name.upper()}}}. All metrics evaluated strictly on the foreground \textbf{{Flood}} class (IoU / Dice).}}")
    tex.append(r"\label{tab:flood_semi_benchmark}")
    tex.append(r"\resizebox{\textwidth}{!}{%")
    tex.append(r"\begin{tabular}{l l cc cc cc cc}")
    tex.append(r"\toprule")
    tex.append(r"\multirow{2}{*}{\textbf{Track}} & \multirow{2}{*}{\textbf{Method}} & \multicolumn{2}{c}{\textbf{10\% Labeled}} & \multicolumn{2}{c}{\textbf{25\% Labeled}} & \multicolumn{2}{c}{\textbf{50\% Labeled}} & \multicolumn{2}{c}{\textbf{100\% Upper Bound}} \\")
    tex.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}")
    tex.append(r" & & IoU$\uparrow$ & Dice$\uparrow$ & IoU$\uparrow$ & Dice$\uparrow$ & IoU$\uparrow$ & Dice$\uparrow$ & IoU$\uparrow$ & Dice$\uparrow$ \\")
    tex.append(r"\midrule")

    current_track = ""
    for track, m_key, m_name in ALL_METHODS:
        if track != current_track:
            if current_track != "":
                tex.append(r"\midrule")
            current_track = track

        cells = []
        for r in RATIOS:
            r = round(r, 2)
            r_list = ds_records.get(m_key, {}).get(r, [])
            iou_str = format_score(r_list, "flood_iou")
            dice_str = format_score(r_list, "flood_dice")
            cells.extend([iou_str, dice_str])

        tex.append(f"{track.split(':')[0]} & {m_name} & " + " & ".join(cells) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}%")
    tex.append(r"}")
    tex.append(r"\end{table*}")

    content = "\n".join(tex)
    os.makedirs(os.path.dirname(out_tex_path), exist_ok=True)
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[LaTeX] Generated booktabs table saved to '{out_tex_path}'.")


def main():
    records = load_all_results()
    for ds in ["floodkaggle", "aifloodsense"]:
        md = generate_markdown_tables(records, ds)
        print(md)
        tex_path = os.path.join(_REPO_ROOT, "docs", f"table_{ds}.tex")
        generate_latex_tables(records, ds, tex_path)

    # Save consolidated LaTeX tables to docs/BAO_CAO_tables.tex
    generate_latex_tables(records, "floodkaggle", os.path.join(_REPO_ROOT, "docs", "BAO_CAO_tables.tex"))


if __name__ == "__main__":
    main()
