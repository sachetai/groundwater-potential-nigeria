#!/usr/bin/env python3

import os
import zipfile
import json
from pathlib import Path
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle

# =========================
# CONFIG
# =========================
BUCKET = "sachet-imerg-nigeria"

# Input zip shown in screenshot
S3_ZIP_KEY = "scripts/final_models.zip"

# Output folder
S3_OUT_PREFIX = "output/final_models/map_comparison_diagnostics"

WORK = Path("/tmp/map_comparison_diagnostics")
ZIP_LOCAL = WORK / "final_models.zip"
EXTRACT_DIR = WORK / "extracted"
OUT = WORK / "outputs"

WORK.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["rf", "xgb", "catboost", "cat_xgb", "cat_rf"]

CLASS_NAMES = {
    1: "Very High",
    2: "High",
    3: "Moderate",
    4: "Low",
    5: "Very Low",
}

SHORT_NAMES = {
    1: "VH",
    2: "H",
    3: "M",
    4: "L",
    5: "VL",
}

COLORS = ["#e31a1c", "#fdae61", "#ffff66", "#a6d96a", "#1a9850"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], CMAP.N)

DIFF_CMAP = ListedColormap(["#ffffff", "#000000"])
DIFF_NORM = BoundaryNorm([-0.5, 0.5, 1.5], DIFF_CMAP.N)

s3 = boto3.client("s3")


# =========================
# HELPERS
# =========================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def s3_download(key, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading s3://{BUCKET}/{key}")
    s3.download_file(BUCKET, key, str(local_path))


def s3_upload(local_path, key):
    s3.upload_file(str(local_path), BUCKET, key)


def upload_folder(local_dir, s3_prefix):
    local_dir = Path(local_dir)
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            s3_upload(p, f"{s3_prefix}/{rel}")


def read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
    return arr.astype(np.int16), profile


def write_tif(path, arr, profile, dtype="uint8", nodata=0):
    prof = profile.copy()
    prof.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def find_file(root, pattern):
    matches = list(Path(root).rglob(pattern))
    if not matches:
        return None
    return matches[0]


def get_raw_map(model):
    return find_file(EXTRACT_DIR, f"final_models/raw_models/{model}/{model}_groundwater_5class_map.tif")


def get_galileo_map(model):
    return find_file(EXTRACT_DIR, f"final_models/galileo_models/{model}/model2_galileo_{model}_5class_map.tif")


def valid_mask(a, b):
    return np.isin(a, [1, 2, 3, 4, 5]) & np.isin(b, [1, 2, 3, 4, 5])


# =========================
# PLOTTING
# =========================
def add_class_colorbar(fig, ax, im):
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels(["Very High", "High", "Moderate", "Low", "Very Low"])
    cbar.set_label("Groundwater Potential Class")


def save_side_by_side_png(raw, gal, out_png, title):
    raw_m = np.ma.masked_where(raw == 0, raw)
    gal_m = np.ma.masked_where(gal == 0, gal)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    im1 = axes[0].imshow(raw_m, cmap=CMAP, norm=NORM)
    axes[0].set_title("Raw ML")
    axes[0].axis("off")

    im2 = axes[1].imshow(gal_m, cmap=CMAP, norm=NORM)
    axes[1].set_title("Galileo-Infused ML")
    axes[1].axis("off")

    add_class_colorbar(fig, axes[1], im2)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_difference_png(diff, out_png, title):
    masked = np.ma.masked_where(diff < 0, diff)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(masked, cmap=DIFF_CMAP, norm=DIFF_NORM)
    ax.set_title(title, fontsize=16)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Same class", "Changed class"])
    cbar.set_label("Pixel Difference")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_class_shift_png(raw, gal, mask, out_png, title):
    """
    Shift:
    negative = Galileo predicts higher potential than raw
    positive = Galileo predicts lower potential than raw
    because class 1 = Very High, class 5 = Very Low.
    """
    shift = np.full(raw.shape, np.nan, dtype=np.float32)
    shift[mask] = gal[mask] - raw[mask]

    cmap = ListedColormap(["#08306b", "#4292c6", "#ffffff", "#fb6a4a", "#99000d"])
    norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(shift, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=16)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels([
        "Galileo much higher",
        "Galileo higher",
        "Same",
        "Galileo lower",
        "Galileo much lower",
    ])
    cbar.set_label("Class Direction Change")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_circled_difference_png(raw, gal, diff, out_png, title, max_circles=8):
    raw_m = np.ma.masked_where(raw == 0, raw)
    gal_m = np.ma.masked_where(gal == 0, gal)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(raw_m, cmap=CMAP, norm=NORM)
    axes[0].set_title("Raw ML")
    axes[0].axis("off")

    axes[1].imshow(gal_m, cmap=CMAP, norm=NORM)
    axes[1].set_title("Galileo-Infused ML")
    axes[1].axis("off")

    # connected components of changed pixels
    changed = diff == 1
    lab, nlab = label(changed)
    objects = find_objects(lab)

    components = []
    for i, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        ys, xs = slc
        area = int(np.sum(lab[slc] == i))
        if area < 80:
            continue
        cy = (ys.start + ys.stop) / 2
        cx = (xs.start + xs.stop) / 2
        radius = max((ys.stop - ys.start), (xs.stop - xs.start)) / 2
        components.append((area, cx, cy, radius))

    components = sorted(components, reverse=True)[:max_circles]

    for area, cx, cy, radius in components:
        for ax in axes:
            circ = Circle(
                (cx, cy),
                radius * 1.15,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
            )
            ax.add_patch(circ)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_transition_heatmap(matrix_df, out_png, title):
    mat = matrix_df.iloc[:, 1:].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="Blues")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Galileo class")
    ax.set_ylabel("Raw class")

    labels = ["VH", "H", "M", "L", "VL"]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    total = mat.sum()
    for i in range(5):
        for j in range(5):
            pct = (mat[i, j] / total * 100) if total > 0 else 0
            ax.text(j, i, f"{pct:.1f}%", ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# =========================
# ANALYSIS
# =========================
def transition_matrix(raw, gal, mask):
    rows = []
    for r in [1, 2, 3, 4, 5]:
        row = {"raw_class": f"{r}_{CLASS_NAMES[r]}"}
        for g in [1, 2, 3, 4, 5]:
            row[f"galileo_{g}_{CLASS_NAMES[g]}"] = int(np.sum(mask & (raw == r) & (gal == g)))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_pair(model, raw_path, gal_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, profile = read_tif(raw_path)
    gal, _ = read_tif(gal_path)

    mask = valid_mask(raw, gal)

    diff = np.full(raw.shape, -1, dtype=np.int16)
    diff[mask] = (raw[mask] != gal[mask]).astype(np.int16)

    total = int(mask.sum())
    changed = int(np.sum(diff == 1))
    same = int(np.sum(diff == 0))
    change_pct = (changed / total * 100) if total else 0

    # Direction shifts
    shift = gal[mask] - raw[mask]
    gal_higher = int(np.sum(shift < 0))  # class value lower = higher potential
    gal_lower = int(np.sum(shift > 0))
    unchanged = int(np.sum(shift == 0))

    summary = {
        "model": model,
        "total_valid_pixels": total,
        "same_pixels": same,
        "changed_pixels": changed,
        "changed_percent": round(change_pct, 4),
        "unchanged_percent": round(100 - change_pct, 4),
        "galileo_higher_potential_pixels": gal_higher,
        "galileo_lower_potential_pixels": gal_lower,
        "unchanged_pixels": unchanged,
        "raw_tif": str(raw_path),
        "galileo_tif": str(gal_path),
    }

    pd.DataFrame([summary]).to_csv(out_dir / f"{model}_raw_vs_galileo_change_summary.csv", index=False)

    tm = transition_matrix(raw, gal, mask)
    tm.to_csv(out_dir / f"{model}_raw_to_galileo_transition_matrix.csv", index=False)

    write_tif(out_dir / f"{model}_raw_vs_galileo_difference_binary.tif", np.where(diff < 0, 0, diff), profile)
    write_tif(out_dir / f"{model}_raw_vs_galileo_class_shift.tif", np.where(mask, gal - raw + 3, 0), profile)

    save_side_by_side_png(
        raw, gal,
        out_dir / f"{model}_side_by_side_raw_vs_galileo.png",
        f"{model.upper()} Raw vs Galileo Side-by-Side"
    )

    save_difference_png(
        diff,
        out_dir / f"{model}_difference_binary_map.png",
        f"{model.upper()} Difference Map: Raw vs Galileo"
    )

    save_class_shift_png(
        raw, gal, mask,
        out_dir / f"{model}_class_shift_direction_map.png",
        f"{model.upper()} Direction of Change: Galileo vs Raw"
    )

    save_circled_difference_png(
        raw, gal, diff,
        out_dir / f"{model}_circled_difference_regions.png",
        f"{model.upper()} Main Difference Regions Circled"
    )

    save_transition_heatmap(
        tm,
        out_dir / f"{model}_raw_to_galileo_transition_heatmap.png",
        f"{model.upper()} Raw → Galileo Transition Matrix"
    )

    return summary


# =========================
# MAIN
# =========================
def main():
    log("Starting Raw vs Galileo comparison diagnostics")

    # Download and extract zip
    s3_download(S3_ZIP_KEY, ZIP_LOCAL)

    log("Extracting zip")
    with zipfile.ZipFile(ZIP_LOCAL, "r") as z:
        z.extractall(EXTRACT_DIR)

    summaries = []

    for model in MODELS:
        raw_path = get_raw_map(model)
        gal_path = get_galileo_map(model)

        if raw_path is None:
            log(f"WARNING: Raw map missing for {model}")
            continue
        if gal_path is None:
            log(f"WARNING: Galileo map missing for {model}")
            continue

        log(f"Comparing {model}")
        model_out = OUT / model
        summaries.append(compare_pair(model, raw_path, gal_path, model_out))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "ALL_MODELS_raw_vs_galileo_change_summary.csv", index=False)

    # Best category comparison: raw CAT_RF vs Galileo CAT_XGB
    raw_best = get_raw_map("cat_rf")
    gal_best = get_galileo_map("cat_xgb")

    if raw_best is not None and gal_best is not None:
        log("Creating best-model cross-category comparison: raw CAT_RF vs Galileo CAT_XGB")
        best_out = OUT / "best_raw_cat_rf_vs_best_galileo_cat_xgb"
        compare_pair("best_raw_cat_rf_vs_galileo_cat_xgb", raw_best, gal_best, best_out)

    # README
    with open(OUT / "README_HOW_TO_EXPLAIN.txt", "w") as f:
        f.write("Raw vs Galileo Map Comparison Diagnostics\n")
        f.write("========================================\n\n")
        f.write("Purpose:\n")
        f.write("These outputs help explain why raw and Galileo maps may look globally similar but still differ in important transition zones.\n\n")
        f.write("Key files:\n")
        f.write("1. *_difference_binary_map.png\n")
        f.write("   - Black pixels show where Raw and Galileo predicted different classes.\n\n")
        f.write("2. *_class_shift_direction_map.png\n")
        f.write("   - Shows whether Galileo shifted pixels to higher or lower groundwater potential.\n\n")
        f.write("3. *_circled_difference_regions.png\n")
        f.write("   - Automatically circles the largest regions where differences occur.\n\n")
        f.write("4. *_raw_to_galileo_transition_matrix.csv/png\n")
        f.write("   - Shows exact class transitions, e.g., Raw Moderate became Galileo High.\n\n")
        f.write("5. ALL_MODELS_raw_vs_galileo_change_summary.csv\n")
        f.write("   - Gives the percent of pixels that changed between Raw and Galileo.\n\n")
        f.write("Suggested explanation:\n")
        f.write("The maps look similar because both models are trained from the same GMM-derived pseudo labels. ")
        f.write("However, the difference maps show where Galileo modifies the raw prediction, mainly in transition zones. ")
        f.write("This helps demonstrate whether Galileo improves spatial consistency rather than changing the entire regional pattern.\n")

    log("Uploading outputs to S3")
    upload_folder(OUT, S3_OUT_PREFIX)

    log(f"DONE. Outputs uploaded to s3://{BUCKET}/{S3_OUT_PREFIX}/")


if __name__ == "__main__":
    main()
