#!/usr/bin/env python3
"""
clip_all_gwp_outputs_to_nw_aoi.py

Fixes the previous issue where only the AOI border was kept and the inside
was removed. This version:

1. Reads NEW_AOI.tif
2. Detects the AOI footprint from nonzero / valid pixels
3. Fills enclosed holes/interior so the full AOI body is preserved
4. Reprojects that filled AOI mask to each model raster grid
5. Removes only the outside excess
6. Writes corrected clipped TIFF + PNG outputs to S3 output/

Expected S3 inputs:
- datasets_2024/NEW_AOI.tif
- output/CAT_5class_AOIcrop.tif
- output/RF_5class_AOIcrop.tif
- output/XGB_5class_AOIcrop.tif
- output/CAT_RF_ENSEMBLE_5class_AOIcrop.tif
- output/CAT_XGB_ENSEMBLE_5class_AOIcrop.tif

Outputs to S3 output/:
- CAT_5class_AOIclip.tif / .png
- RF_5class_AOIclip.tif / .png
- XGB_5class_AOIclip.tif / .png
- CAT_RF_ENSEMBLE_5class_AOIclip.tif / .png
- CAT_XGB_ENSEMBLE_5class_AOIclip.tif / .png
"""

import os
import gc
import argparse
from typing import Tuple, List

import boto3
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from scipy.ndimage import binary_fill_holes, binary_closing


CLASS_LABELS = ["Very High", "High", "Moderate", "Low", "Very Low"]
CLASS_COLORS = ["#e41a1c", "#f2a654", "#ecec57", "#9ccc65", "#1b9e3c"]


# -------------------------
# S3 helpers
# -------------------------
def s3_download(bucket: str, key: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    boto3.client("s3").download_file(bucket, key, out_path)


def s3_upload(bucket: str, local_path: str, key: str):
    boto3.client("s3").upload_file(local_path, bucket, key)


# -------------------------
# Raster helpers
# -------------------------
def read_raster(path: str):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        dataset_mask = src.dataset_mask()
    return arr, profile, transform, crs, nodata, dataset_mask


def build_filled_aoi_mask(aoi_path: str):
    """
    Build a FILLED AOI mask from NEW_AOI.tif.
    This fixes the issue where the AOI raster acts like only an outline.
    """
    with rasterio.open(aoi_path) as src:
        arr = src.read(1)
        ds_mask = src.dataset_mask() > 0
        nodata = src.nodata

        candidates = []

        # Candidate 1: dataset mask
        if ds_mask.any():
            candidates.append(("dataset_mask", ds_mask))

        # Candidate 2: nonzero pixels
        nz = np.isfinite(arr) & (arr != 0)
        if nz.any():
            candidates.append(("nonzero", nz))

        # Candidate 3: nodata-aware valid pixels
        if nodata is not None:
            nd = np.isfinite(arr) & (arr != nodata)
            if nd.any():
                candidates.append(("not_nodata", nd))

        # Candidate 4: any finite pixels
        finite = np.isfinite(arr)
        if finite.any():
            candidates.append(("finite", finite))

        if not candidates:
            raise RuntimeError("Could not detect any AOI pixels from NEW_AOI.tif")

        # choose the candidate with the largest support
        name, base_mask = max(candidates, key=lambda x: int(x[1].sum()))

        # Important fix:
        # fill interior if the AOI is boundary-only
        filled = binary_fill_holes(base_mask)

        # tiny cleanup for broken edges
        filled = binary_closing(filled, structure=np.ones((3, 3), dtype=bool))
        filled = binary_fill_holes(filled)

        if filled.sum() == 0:
            raise RuntimeError("Filled AOI mask is empty after processing.")

        return filled.astype(np.uint8), src.transform, src.crs, name


def reproject_aoi_mask_to_model(
    src_mask: np.ndarray,
    src_transform,
    src_crs,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
) -> np.ndarray:
    dst = np.zeros(dst_shape, dtype=np.uint8)

    reproject(
        source=src_mask,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    return dst > 0


def apply_mask(arr: np.ndarray, keep_mask: np.ndarray, nodata_value=0) -> np.ndarray:
    out = arr.copy()
    out[~keep_mask] = nodata_value
    return out


def crop_to_mask_extent(arr: np.ndarray, keep_mask: np.ndarray):
    rows, cols = np.where(keep_mask)
    if rows.size == 0 or cols.size == 0:
        raise RuntimeError("AOI mask produced no valid overlap with model raster.")

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()

    cropped = arr[rmin:rmax + 1, cmin:cmax + 1]
    return cropped, (rmin, rmax, cmin, cmax)


def shift_transform(transform, row_off: int, col_off: int):
    return transform * Affine.translation(col_off, row_off)


def write_geotiff(path: str, arr: np.ndarray, profile: dict, transform):
    p = profile.copy()
    p.update({
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": arr.dtype,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    })
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr, 1)


# -------------------------
# Plotting
# -------------------------
def plot_5class_png(arr: np.ndarray, out_png: str, title: str):
    a = arr.astype(float)
    a[a == 0] = np.nan

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    im = ax.imshow(a, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(CLASS_LABELS)
    cbar.set_label("Groundwater Potential Class")

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    gc.collect()


# -------------------------
# Naming
# -------------------------
def make_output_names(input_key: str):
    base = os.path.basename(input_key)
    if base.lower().endswith(".tif"):
        base = base[:-4]
    base = base.replace("_AOIcrop", "_AOIclip")
    return f"{base}.tif", f"{base}.png"


def title_from_filename(name_no_ext: str):
    return name_no_ext.replace("_AOIclip", "").replace("_", " ")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--aoi_key", required=True)
    ap.add_argument("--model_keys", nargs="+", required=True)
    ap.add_argument("--out_prefix", default="output")
    ap.add_argument("--local_dir", default="./work_clip_outputs")
    args = ap.parse_args()

    bucket = args.bucket
    out_prefix = args.out_prefix.strip("/")
    os.makedirs(args.local_dir, exist_ok=True)

    local_aoi = os.path.join(args.local_dir, os.path.basename(args.aoi_key))
    print(f"[1/4] Download AOI: s3://{bucket}/{args.aoi_key}")
    s3_download(bucket, args.aoi_key, local_aoi)

    print("[2/4] Build FILLED AOI mask from NEW_AOI.tif")
    aoi_mask_src, aoi_transform, aoi_crs, method_name = build_filled_aoi_mask(local_aoi)
    print(f"    AOI source method used: {method_name}")
    print(f"    AOI filled pixels: {int(aoi_mask_src.sum())}")

    uploaded = []

    print("[3/4] Clip all model rasters to filled NW AOI")
    for i, model_key in enumerate(args.model_keys, start=1):
        print(f"\n--- [{i}/{len(args.model_keys)}] Processing {model_key}")

        local_model = os.path.join(args.local_dir, os.path.basename(model_key))
        s3_download(bucket, model_key, local_model)

        arr, profile, transform, crs, nodata, ds_mask = read_raster(local_model)

        keep_mask = reproject_aoi_mask_to_model(
            src_mask=aoi_mask_src,
            src_transform=aoi_transform,
            src_crs=aoi_crs,
            dst_shape=arr.shape,
            dst_transform=transform,
            dst_crs=crs,
        )

        if keep_mask.sum() == 0:
            raise RuntimeError(f"No AOI overlap found for model raster: {model_key}")

        clipped = apply_mask(arr, keep_mask, nodata_value=0)
        clipped_crop, (rmin, rmax, cmin, cmax) = crop_to_mask_extent(clipped, keep_mask)
        new_transform = shift_transform(transform, rmin, cmin)

        tif_name, png_name = make_output_names(model_key)
        local_tif = os.path.join(args.local_dir, tif_name)
        local_png = os.path.join(args.local_dir, png_name)

        write_geotiff(local_tif, clipped_crop, profile, new_transform)
        plot_5class_png(clipped_crop, local_png, title_from_filename(tif_name[:-4]))

        s3_tif_key = f"{out_prefix}/{tif_name}"
        s3_png_key = f"{out_prefix}/{png_name}"

        s3_upload(bucket, local_tif, s3_tif_key)
        s3_upload(bucket, local_png, s3_png_key)

        uploaded.append(f"s3://{bucket}/{s3_tif_key}")
        uploaded.append(f"s3://{bucket}/{s3_png_key}")

        for fp in [local_model, local_tif, local_png]:
            try:
                os.remove(fp)
            except Exception:
                pass

        del arr, ds_mask, keep_mask, clipped, clipped_crop
        gc.collect()

    try:
        os.remove(local_aoi)
    except Exception:
        pass

    print("\n[4/4] Done ✅ Uploaded:")
    for x in uploaded:
        print(x)


if __name__ == "__main__":
    main()
