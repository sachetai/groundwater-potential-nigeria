#!/usr/bin/env python3
"""
compare_ahp_ml_maps_fixed.py

Compare an AHP map and an ML groundwater potential map on AWS EC2.

IMPORTANT FIXES
---------------
1) AHP TIFF is treated as a rendered RGB/RGBA map, not as a raw class band.
2) AHP is reclassified using advisor's legend:
      Red        = Very High
      Orange     = High
      Yellow     = Moderate
      LightGreen = Low
      Green      = Very Low
3) ML TIFF is treated as a normal single-band 5-class raster.
4) ML is aligned to AHP grid before comparison.
5) Scatter plot is a bubble/count scatter so it is actually informative.
6) Outputs are written to S3 output/ folder.

Advisor legend:
- red region is Very High
- orange is High
- yellow is Moderate
- light green is Low
- green is Very Low

Run:
python compare_ahp_ml_maps_fixed.py \
  --bucket sachet-imerg-nigeria \
  --ahp_key scripts/Final_GWP_Map1.tif \
  --ml_key scripts/CAT_XGB_ENSEMBLE_5class_AOIcrop.tif \
  --out_prefix output \
  --ahp_high_value high \
  --ml_high_value high
"""

import os
import json
import argparse
from typing import Dict, Tuple, List

import numpy as np
import boto3
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import ColorInterp
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# -------------------------
# S3 helpers
# -------------------------
def s3_download(bucket: str, key: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    boto3.client("s3").download_file(bucket, key, out_path)


def s3_upload(bucket: str, local_path: str, key: str) -> None:
    boto3.client("s3").upload_file(local_path, bucket, key)


# -------------------------
# Class labels and colors
# -------------------------
CLASS_NAMES_HIGH_TO_LOW = ["VeryHigh", "High", "Moderate", "Low", "VeryLow"]
CLASS_NAMES_LOW_TO_HIGH = ["VeryLow", "Low", "Moderate", "High", "VeryHigh"]

# advisor legend colors for display
DISPLAY_COLORS_HIGH_TO_LOW = [
    "#ef1010",  # Very High = red
    "#f2a856",  # High = orange
    "#ecec53",  # Moderate = yellow
    "#9ccc65",  # Low = light green
    "#1b9e3e",  # Very Low = green
]

DISPLAY_COLORS_LOW_TO_HIGH = list(reversed(DISPLAY_COLORS_HIGH_TO_LOW))


# -------------------------
# AHP RGB legend mapping
# -------------------------
# We classify RGB pixels by nearest advisor color.
# Order below is class code 1..5 when high=high:
# 1=VeryHigh(red), 2=High(orange), 3=Moderate(yellow), 4=Low(light green), 5=VeryLow(green)
AHP_LEGEND_RGB = {
    1: np.array([239, 16, 16], dtype=np.float32),    # red
    2: np.array([242, 168, 86], dtype=np.float32),   # orange
    3: np.array([236, 236, 83], dtype=np.float32),   # yellow
    4: np.array([156, 204, 101], dtype=np.float32),  # light green
    5: np.array([27, 158, 62], dtype=np.float32),    # green
}


# -------------------------
# Utility functions
# -------------------------
def get_class_names(high_value: str) -> List[str]:
    return CLASS_NAMES_HIGH_TO_LOW if high_value.lower() == "high" else CLASS_NAMES_LOW_TO_HIGH


def get_display_colors(high_value: str) -> List[str]:
    return DISPLAY_COLORS_HIGH_TO_LOW if high_value.lower() == "high" else DISPLAY_COLORS_LOW_TO_HIGH


def maybe_flip_classes(arr: np.ndarray, high_value: str) -> np.ndarray:
    """
    Normalize class direction to:
      1 = Very High, ..., 5 = Very Low  when high_value='high'
      1 = Very Low,  ..., 5 = Very High when high_value='low'
    Internally we compare after converting both rasters to the SAME requested direction.
    """
    high_value = high_value.lower()
    if high_value not in {"high", "low"}:
        raise ValueError("--*_high_value must be 'high' or 'low'")
    # arr already 1..5. If user says high=low, flip.
    if high_value == "low":
        out = arr.copy()
        valid = (out >= 1) & (out <= 5)
        out[valid] = 6 - out[valid]
        return out
    return arr


def write_geotiff(path: str, arr: np.ndarray, profile: dict, nodata=0) -> None:
    prof = profile.copy()
    prof.update(
        {
            "driver": "GTiff",
            "height": arr.shape[0],
            "width": arr.shape[1],
            "count": 1,
            "dtype": arr.dtype,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": nodata,
        }
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(pearsonr(x, y)[0])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(spearmanr(x, y)[0])


# -------------------------
# Reading AHP as RGB/RGBA rendered map
# -------------------------
def read_ahp_rendered_rgb(path: str) -> Tuple[np.ndarray, dict]:
    """
    Read AHP TIFF as rendered image.
    Returns RGB image as HxWx3 uint8 and profile from source.
    """
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        count = src.count

        if count < 3:
            raise RuntimeError(
                "AHP TIFF has fewer than 3 bands. This script expects a rendered RGB/RGBA AHP map."
            )

        arr = src.read()  # (bands, h, w)

        # Use first 3 bands as RGB
        rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.uint8)

        return rgb, profile


def build_ahp_valid_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Exclude background / whitespace / black no-data.
    Keeps colored map pixels.
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    # remove near-white backgrounds
    white_bg = (r > 245) & (g > 245) & (b > 245)
    # remove near-black backgrounds
    black_bg = (r < 10) & (g < 10) & (b < 10)

    # keep all other pixels
    valid = ~(white_bg | black_bg)
    return valid


def classify_ahp_rgb_to_5class(rgb: np.ndarray, high_value: str = "high") -> Tuple[np.ndarray, Dict]:
    """
    Reclassify advisor-style rendered AHP RGB map into 5 classes.
    Base mapping:
      1=VeryHigh(red), 2=High(orange), 3=Moderate(yellow), 4=Low(light green), 5=VeryLow(green)
    If high_value='low', classes are flipped afterward.
    """
    valid_mask = build_ahp_valid_mask(rgb)
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)

    if valid_mask.sum() == 0:
        raise RuntimeError("No valid AHP colored pixels found after background masking.")

    rgb_valid = rgb[valid_mask].astype(np.float32)  # (N,3)

    # distance to advisor legend colors
    legend_codes = np.array(sorted(AHP_LEGEND_RGB.keys()), dtype=np.int32)
    legend_colors = np.stack([AHP_LEGEND_RGB[k] for k in legend_codes], axis=0)  # (5,3)

    # squared euclidean distance
    # result shape: (N, 5)
    d2 = ((rgb_valid[:, None, :] - legend_colors[None, :, :]) ** 2).sum(axis=2)
    nearest_idx = np.argmin(d2, axis=1)
    class_vals = legend_codes[nearest_idx].astype(np.uint8)

    out[valid_mask] = class_vals
    out = maybe_flip_classes(out, high_value)

    info = {
        "method": "rgb_rendered_nearest_legend",
        "high_value": high_value.lower(),
        "legend_rgb_base_mapping_1to5": {
            "1": "VeryHigh=red",
            "2": "High=orange",
            "3": "Moderate=yellow",
            "4": "Low=light_green",
            "5": "VeryLow=green",
        },
        "valid_pixels": int(valid_mask.sum()),
        "unique_classes_found": [int(v) for v in np.unique(out[out > 0]).tolist()],
    }
    return out, info


# -------------------------
# Reading ML raster
# -------------------------
def read_ml_singleband(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        arr = src.read(1)
    return arr, profile


def normalize_ml_to_5class(arr: np.ndarray, high_value: str = "high") -> Tuple[np.ndarray, Dict]:
    """
    ML should already be 1..5.
    If it is 1..5, just possibly flip.
    Otherwise try simple quantile fallback.
    """
    out = np.array(arr, copy=True)
    valid = np.isfinite(out) & (out > 0)

    if valid.sum() == 0:
        raise RuntimeError("ML raster has no valid positive pixels.")

    uniq = np.unique(out[valid])

    if uniq.min() >= 1 and uniq.max() <= 5 and uniq.size <= 5:
        out = out.astype(np.uint8)
        out = maybe_flip_classes(out, high_value)
        info = {
            "method": "already_5class",
            "high_value": high_value.lower(),
            "input_unique_values_sample": [int(v) for v in uniq.tolist()],
            "input_unique_count": int(uniq.size),
        }
        return out, info

    # fallback quantiles
    vals = out[valid].astype(np.float32)
    q20, q40, q60, q80 = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])

    cls = np.zeros_like(out, dtype=np.uint8)
    cls[valid & (out <= q20)] = 1
    cls[valid & (out > q20) & (out <= q40)] = 2
    cls[valid & (out > q40) & (out <= q60)] = 3
    cls[valid & (out > q60) & (out <= q80)] = 4
    cls[valid & (out > q80)] = 5

    cls = maybe_flip_classes(cls, high_value)
    info = {
        "method": "continuous_to_5_quantiles",
        "high_value": high_value.lower(),
        "input_unique_values_sample": [float(v) for v in uniq[:20].tolist()],
        "input_unique_count": int(uniq.size),
        "breaks": {"q20": float(q20), "q40": float(q40), "q60": float(q60), "q80": float(q80)},
    }
    return cls, info


# -------------------------
# Alignment
# -------------------------
def align_ml_to_ahp_grid(ml_arr: np.ndarray, ml_profile: dict, ahp_profile: dict) -> np.ndarray:
    """
    Reproject/resample ML 5-class raster onto AHP grid.
    """
    dst = np.zeros((ahp_profile["height"], ahp_profile["width"]), dtype=np.uint8)

    reproject(
        source=ml_arr,
        destination=dst,
        src_transform=ml_profile["transform"],
        src_crs=ml_profile["crs"],
        dst_transform=ahp_profile["transform"],
        dst_crs=ahp_profile["crs"],
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst


# -------------------------
# Plotting
# -------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="viridis")

    ax.set_title("Confusion Matrix: AHP(5-class) vs ML(5-class)", fontsize=18, pad=12)
    ax.set_xlabel("ML class", fontsize=14)
    ax.set_ylabel("AHP class", fontsize=14)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)

    maxv = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > maxv * 0.45 else "black"
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color=color, fontsize=11)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=11)

    fig.subplots_adjust(left=0.18, right=0.92, bottom=0.18, top=0.90)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_difference_map(diff_arr: np.ndarray, out_png: str) -> None:
    diff = diff_arr.astype(np.float32)
    diff[diff_arr == 255] = np.nan  # keep if any reserved, though not used

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(diff, cmap="viridis", vmin=-4, vmax=4)
    ax.set_title("Difference Map (ML5 - AHP5)", fontsize=18, pad=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Class difference", fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.90)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_bubble_scatter(ahp_vals: np.ndarray, ml_vals: np.ndarray, class_names: List[str], out_png: str) -> None:
    """
    Bubble scatter of class-pair counts, much more informative than plotting all raw pixels.
    """
    counts = np.zeros((5, 5), dtype=np.int64)
    for a in range(1, 6):
        for m in range(1, 6):
            counts[a - 1, m - 1] = np.sum((ahp_vals == a) & (ml_vals == m))

    fig, ax = plt.subplots(figsize=(8, 8))
    maxc = counts.max() if counts.max() > 0 else 1

    for a in range(1, 6):
        for m in range(1, 6):
            c = counts[a - 1, m - 1]
            if c > 0:
                size = 40 + 1400 * (c / maxc)
                ax.scatter(a, m, s=size, alpha=0.65)
                ax.text(a, m, f"{c}", ha="center", va="center", fontsize=9)

    ax.set_title("Correlation Scatter: AHP vs ML", fontsize=18, pad=12)
    ax.set_xlabel("AHP class", fontsize=14)
    ax.set_ylabel("ML class", fontsize=14)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)

    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.90)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side(
    ahp_arr: np.ndarray,
    ml_arr: np.ndarray,
    class_names: List[str],
    colors: List[str],
    out_png: str,
) -> None:
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(np.where(ahp_arr > 0, ahp_arr, np.nan), cmap=cmap, norm=norm)
    axes[0].set_title("AHP (5 classes)", fontsize=18)
    axes[0].axis("off")

    im = axes[1].imshow(np.where(ml_arr > 0, ml_arr, np.nan), cmap=cmap, norm=norm)
    axes[1].set_title("ML (5 classes)", fontsize=18)
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(class_names)
    cbar.set_label("Groundwater Potential Class", fontsize=13)

    fig.subplots_adjust(left=0.03, right=0.92, bottom=0.04, top=0.90, wspace=0.10)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--ahp_key", required=True, help="e.g. scripts/Final_GWP_Map1.tif")
    ap.add_argument("--ml_key", required=True, help="e.g. scripts/CAT_XGB_ENSEMBLE_5class_AOIcrop.tif")
    ap.add_argument("--out_prefix", default="output")
    ap.add_argument("--ahp_high_value", default="high", choices=["high", "low"])
    ap.add_argument("--ml_high_value", default="high", choices=["high", "low"])
    ap.add_argument("--local_dir", default="./compare_ahp_ml_work")
    args = ap.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    ahp_local = os.path.join(args.local_dir, os.path.basename(args.ahp_key))
    ml_local = os.path.join(args.local_dir, os.path.basename(args.ml_key))

    print(f"[1/6] Download AHP raster: s3://{args.bucket}/{args.ahp_key}")
    s3_download(args.bucket, args.ahp_key, ahp_local)

    print(f"[2/6] Download ML raster: s3://{args.bucket}/{args.ml_key}")
    s3_download(args.bucket, args.ml_key, ml_local)

    print("[3/6] Read AHP RGB render and align ML to AHP grid")
    ahp_rgb, ahp_profile = read_ahp_rendered_rgb(ahp_local)
    ml_raw, ml_profile = read_ml_singleband(ml_local)

    print("[4/6] Normalize both rasters to 5 classes")
    ahp_5, ahp_info = classify_ahp_rgb_to_5class(ahp_rgb, high_value=args.ahp_high_value)
    ml_5_raw, ml_info = normalize_ml_to_5class(ml_raw, high_value=args.ml_high_value)
    ml_5 = align_ml_to_ahp_grid(ml_5_raw.astype(np.uint8), ml_profile, ahp_profile)

    print("AHP normalization info:")
    print(json.dumps(ahp_info, indent=2))
    print("ML normalization info:")
    print(json.dumps(ml_info, indent=2))

    print("[5/6] Compute confusion matrix, correlation, agreement, difference")
    valid = (ahp_5 >= 1) & (ahp_5 <= 5) & (ml_5 >= 1) & (ml_5 <= 5)

    if valid.sum() == 0:
        raise RuntimeError("No overlapping valid class pixels found between AHP and ML.")

    ahp_vals = ahp_5[valid].astype(np.int16)
    ml_vals = ml_5[valid].astype(np.int16)

    cm = confusion_matrix(ahp_vals, ml_vals, labels=[1, 2, 3, 4, 5])
    kappa = float(cohen_kappa_score(ahp_vals, ml_vals, labels=[1, 2, 3, 4, 5]))
    overall_acc = float((ahp_vals == ml_vals).mean())
    within_one = float((np.abs(ahp_vals - ml_vals) <= 1).mean())
    mean_abs_diff = float(np.mean(np.abs(ahp_vals - ml_vals)))
    pear = safe_corr(ahp_vals.astype(np.float32), ml_vals.astype(np.float32))
    spear = safe_spearman(ahp_vals.astype(np.float32), ml_vals.astype(np.float32))

    diff_map = np.zeros_like(ahp_5, dtype=np.int16)
    diff_map[valid] = ml_5[valid].astype(np.int16) - ahp_5[valid].astype(np.int16)

    metrics = {
        "ahp_info": ahp_info,
        "ml_info": ml_info,
        "n_overlap_valid_pixels": int(valid.sum()),
        "overall_accuracy": overall_acc,
        "cohens_kappa": kappa,
        "pearson_r": pear,
        "spearman_rho": spear,
        "exact_match_percent": round(100.0 * overall_acc, 2),
        "within_one_class_pct": round(100.0 * within_one, 2),
        "mean_abs_class_diff": mean_abs_diff,
        "confusion_matrix_rows_ahp_cols_ml": cm.tolist(),
        "class_order": get_class_names(args.ahp_high_value),
    }

    print("[6/6] Write outputs locally and upload to S3")

    base = f"compare_{os.path.splitext(os.path.basename(args.ahp_key))[0]}_vs_{os.path.splitext(os.path.basename(args.ml_key))[0]}"

    p_ahp_tif = os.path.join(args.local_dir, f"{base}_AHP_reclass_5class.tif")
    p_ml_tif = os.path.join(args.local_dir, f"{base}_ML_aligned_to_AHP_5class.tif")
    p_diff_tif = os.path.join(args.local_dir, f"{base}_difference_map.tif")
    p_json = os.path.join(args.local_dir, f"{base}_metrics.json")
    p_cm_png = os.path.join(args.local_dir, f"{base}_confusion_matrix.png")
    p_diff_png = os.path.join(args.local_dir, f"{base}_difference_map.png")
    p_scatter_png = os.path.join(args.local_dir, f"{base}_correlation_scatter.png")
    p_side_png = os.path.join(args.local_dir, f"{base}_side_by_side.png")

    out_profile = ahp_profile.copy()
    out_profile.update({"count": 1, "dtype": rasterio.uint8})
    write_geotiff(p_ahp_tif, ahp_5.astype(np.uint8), out_profile, nodata=0)
    write_geotiff(p_ml_tif, ml_5.astype(np.uint8), out_profile, nodata=0)

    diff_profile = ahp_profile.copy()
    diff_profile.update({"count": 1, "dtype": rasterio.int16})
    write_geotiff(p_diff_tif, diff_map.astype(np.int16), diff_profile, nodata=0)

    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    class_names = get_class_names(args.ahp_high_value)
    colors = get_display_colors(args.ahp_high_value)

    plot_confusion_matrix(cm, class_names, p_cm_png)
    plot_difference_map(diff_map, p_diff_png)
    plot_bubble_scatter(ahp_vals, ml_vals, class_names, p_scatter_png)
    plot_side_by_side(ahp_5, ml_5, class_names, colors, p_side_png)

    out_prefix = args.out_prefix.strip("/")
    outputs = [
        p_ahp_tif,
        p_ml_tif,
        p_diff_tif,
        p_json,
        p_cm_png,
        p_diff_png,
        p_scatter_png,
        p_side_png,
    ]

    for lp in outputs:
        key = f"{out_prefix}/{os.path.basename(lp)}"
        s3_upload(args.bucket, lp, key)

    print("\n✅ Outputs uploaded:")
    for lp in outputs:
        print(f"s3://{args.bucket}/{out_prefix}/{os.path.basename(lp)}")

    print("\nQuick summary:")
    print(f"overall_accuracy      : {overall_acc:.4f}")
    print(f"cohens_kappa          : {kappa:.4f}")
    print(f"pearson_r             : {pear:.4f}")
    print(f"spearman_rho          : {spear:.4f}")
    print(f"exact_match_percent   : {100.0 * overall_acc:.2f}")
    print(f"within_one_class_pct  : {100.0 * within_one:.2f}")
    print(f"mean_abs_class_diff   : {mean_abs_diff:.4f}")


if __name__ == "__main__":
    main()
