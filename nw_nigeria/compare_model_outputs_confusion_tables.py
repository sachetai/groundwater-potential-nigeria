#!/usr/bin/env python3
"""
compare_model_outputs_confusion_tables.py

Build pairwise confusion-matrix tables for all groundwater model output rasters.

Expected inputs:
- 5-class GeoTIFFs already produced by your models, for example:
    output/RF_5class_AOIcrop.tif
    output/XGB_5class_AOIcrop.tif
    output/CAT_5class_AOIcrop.tif
    output/CAT_RF_ENSEMBLE_5class_AOIcrop.tif
    output/CAT_XGB_ENSEMBLE_5class_AOIcrop.tif

What it does:
1. Downloads all rasters from S3
2. Reprojects/resamples each raster to a common reference grid
3. Computes pairwise confusion matrices
4. Writes:
   - one raw confusion-matrix CSV per model pair
   - one row-normalized confusion-matrix CSV per model pair
   - one summary CSV with overall agreement stats
5. Uploads all outputs to S3 output/

Notes:
- This compares model output maps to each other, not to borehole truth.
- So "accuracy" here means map-to-map agreement, not real predictive accuracy.
"""

import os
import json
import argparse
import itertools
from typing import Dict, List, Tuple

import boto3
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from sklearn.metrics import confusion_matrix, cohen_kappa_score


CLASS_NAMES = {
    1: "VeryHigh",
    2: "High",
    3: "Moderate",
    4: "Low",
    5: "VeryLow",
}


def s3_download(bucket: str, key: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    boto3.client("s3").download_file(bucket, key, out_path)


def s3_upload(bucket: str, local_path: str, key: str):
    boto3.client("s3").upload_file(local_path, bucket, key)


def read_raster(path: str):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    return arr, profile, transform, crs, nodata


def align_to_reference(src_path: str, ref_profile: dict) -> np.ndarray:
    """
    Reproject/resample src raster to the reference raster grid.
    Since classes are categorical, use nearest-neighbor.
    """
    with rasterio.open(src_path) as src:
        dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.uint8)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return dst


def valid_joint_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Keep only pixels where both maps have valid 5-class values.
    """
    return np.isin(a, [1, 2, 3, 4, 5]) & np.isin(b, [1, 2, 3, 4, 5])


def build_confusion_tables(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    name_a: str,
    name_b: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    labels = [1, 2, 3, 4, 5]
    mask = valid_joint_mask(arr_a, arr_b)

    y_true = arr_a[mask].astype(int)
    y_pred = arr_b[mask].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row_labels = [CLASS_NAMES[x] for x in labels]
    col_labels = [CLASS_NAMES[x] for x in labels]

    cm_df = pd.DataFrame(cm, index=row_labels, columns=col_labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )
    cm_norm_df = pd.DataFrame(cm_norm, index=row_labels, columns=col_labels)

    exact_match = float((y_true == y_pred).mean()) if y_true.size else np.nan
    within_one = float((np.abs(y_true - y_pred) <= 1).mean()) if y_true.size else np.nan
    mean_abs_diff = float(np.abs(y_true - y_pred).mean()) if y_true.size else np.nan
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels)) if y_true.size else np.nan

    summary = {
        "reference_model": name_a,
        "compared_model": name_b,
        "n_valid_pixels": int(y_true.size),
        "exact_match_pct": round(exact_match * 100.0, 4) if y_true.size else np.nan,
        "within_one_class_pct": round(within_one * 100.0, 4) if y_true.size else np.nan,
        "mean_abs_class_difference": round(mean_abs_diff, 6) if y_true.size else np.nan,
        "cohens_kappa": round(kappa, 6) if y_true.size else np.nan,
    }

    return cm_df, cm_norm_df, summary


def sanitize_name(s: str) -> str:
    return s.replace(" ", "_").replace("+", "plus").replace("/", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="S3 bucket name")
    ap.add_argument("--out_prefix", default="output", help="S3 prefix for outputs")
    ap.add_argument(
        "--model_keys",
        nargs="+",
        required=True,
        help=(
            "List of model GeoTIFF keys in S3. Example:\n"
            "output/RF_5class_AOIcrop.tif "
            "output/XGB_5class_AOIcrop.tif "
            "output/CAT_5class_AOIcrop.tif "
            "output/CAT_RF_ENSEMBLE_5class_AOIcrop.tif "
            "output/CAT_XGB_ENSEMBLE_5class_AOIcrop.tif"
        ),
    )
    ap.add_argument(
        "--model_names",
        nargs="+",
        required=True,
        help=(
            "Human-readable names in the same order as --model_keys. Example:\n"
            "RF XGB CAT CATplusRF CATplusXGB"
        ),
    )
    ap.add_argument("--local_dir", default="./work_confusion_tables")
    args = ap.parse_args()

    if len(args.model_keys) != len(args.model_names):
        raise ValueError("--model_keys and --model_names must have the same length.")

    os.makedirs(args.local_dir, exist_ok=True)

    # ----------------------------------------
    # Download all rasters
    # ----------------------------------------
    local_files = {}
    for key, name in zip(args.model_keys, args.model_names):
        local_path = os.path.join(args.local_dir, os.path.basename(key))
        print(f"Downloading {name}: s3://{args.bucket}/{key}")
        s3_download(args.bucket, key, local_path)
        local_files[name] = local_path

    # ----------------------------------------
    # Use first raster as reference grid
    # ----------------------------------------
    ref_name = args.model_names[0]
    ref_path = local_files[ref_name]

    ref_arr, ref_profile, _, _, _ = read_raster(ref_path)
    ref_profile["nodata"] = 0

    aligned = {}
    for name, path in local_files.items():
        if name == ref_name:
            aligned[name] = ref_arr.astype(np.uint8)
        else:
            aligned[name] = align_to_reference(path, ref_profile).astype(np.uint8)

    # ----------------------------------------
    # Pairwise comparisons
    # ----------------------------------------
    summary_rows = []
    output_files = []

    for name_a, name_b in itertools.combinations(args.model_names, 2):
        arr_a = aligned[name_a]
        arr_b = aligned[name_b]

        cm_df, cm_norm_df, summary = build_confusion_tables(arr_a, arr_b, name_a, name_b)
        summary_rows.append(summary)

        base = f"{sanitize_name(name_a)}_vs_{sanitize_name(name_b)}"

        raw_csv = os.path.join(args.local_dir, f"{base}_confusion_raw.csv")
        norm_csv = os.path.join(args.local_dir, f"{base}_confusion_row_normalized.csv")
        meta_json = os.path.join(args.local_dir, f"{base}_summary.json")

        cm_df.to_csv(raw_csv)
        cm_norm_df.to_csv(norm_csv)
        with open(meta_json, "w") as f:
            json.dump(summary, f, indent=2)

        output_files.extend([raw_csv, norm_csv, meta_json])

        print(f"Done: {name_a} vs {name_b}")

    # ----------------------------------------
    # Summary CSV
    # ----------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.local_dir, "all_model_pairwise_confusion_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    output_files.append(summary_csv)

    # ----------------------------------------
    # Upload all outputs
    # ----------------------------------------
    s3_prefix = args.out_prefix.strip("/") + "/model_confusion_tables"
    print("\nUploading outputs...")
    for fp in output_files:
        key = f"{s3_prefix}/{os.path.basename(fp)}"
        s3_upload(args.bucket, fp, key)
        print(f"s3://{args.bucket}/{key}")

    print("\nDone.")


if __name__ == "__main__":
    main()
