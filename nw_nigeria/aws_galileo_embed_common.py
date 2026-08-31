#!/usr/bin/env python3
import os
import sys
import gc
import json
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import Affine
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.format import open_memmap


s3 = boto3.client("s3")


def log(msg: str):
    print(msg, flush=True)


def s3_download(bucket: str, key: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    s3.download_file(bucket, key, out_path)


def s3_upload(bucket: str, local_path: str, key: str):
    s3.upload_file(local_path, bucket, key)


def infer_transform_from_xy(ds: xr.Dataset):
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("Dataset missing x/y coordinates; cannot infer transform.")

    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)

    if len(x) < 2 or len(y) < 2:
        raise ValueError("x/y coordinates too short to infer transform.")

    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))

    x0 = float(x[0] - dx / 2.0)
    y0 = float(y[0] - dy / 2.0)

    return Affine(dx, 0.0, x0, 0.0, dy, y0)


def get_crs_transform(ds: xr.Dataset):
    crs = None
    transform = None

    if "spatial_ref" in ds.coords:
        sp = ds["spatial_ref"]
        wkt = sp.attrs.get("crs_wkt") or sp.attrs.get("spatial_ref")
        gt = sp.attrs.get("GeoTransform")

        if wkt:
            try:
                crs = rasterio.crs.CRS.from_wkt(wkt)
            except Exception:
                crs = None

        if gt:
            try:
                parts = [float(v) for v in str(gt).strip().split()]
                transform = Affine.from_gdal(*parts)
            except Exception:
                transform = None

    if crs is None:
        for key in ["crs_wkt", "spatial_ref", "crs"]:
            val = ds.attrs.get(key)
            if val:
                try:
                    if str(val).startswith(("EPSG:", "epsg:")):
                        crs = rasterio.crs.CRS.from_string(str(val))
                    else:
                        crs = rasterio.crs.CRS.from_wkt(str(val))
                    break
                except Exception:
                    pass

    if transform is None:
        gt = ds.attrs.get("GeoTransform")
        if gt:
            try:
                parts = [float(v) for v in str(gt).strip().split()]
                transform = Affine.from_gdal(*parts)
            except Exception:
                transform = None

    if transform is None:
        transform = infer_transform_from_xy(ds)

    if crs is None:
        crs = rasterio.crs.CRS.from_epsg(3857)
        log("WARNING: CRS not found in NetCDF metadata; defaulting to EPSG:3857.")

    return crs, transform


def to_2d(da: xr.DataArray) -> np.ndarray:
    da2 = da.squeeze(drop=True)
    while len(da2.dims) > 2:
        extra = [d for d in da2.dims if d not in ("y", "x")][0]
        da2 = da2.isel({extra: 0}).squeeze(drop=True)
    if da2.dims != ("y", "x"):
        da2 = da2.transpose("y", "x")
    return da2.values.astype("float32")


def robust_zscore(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    vals = out[valid_mask]
    if vals.size == 0:
        return np.zeros_like(out, dtype=np.float32)

    median = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - median))

    if not np.isfinite(mad) or mad < 1e-8:
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if not np.isfinite(std) or std < 1e-8:
            out[valid_mask] = 0.0
            out[~valid_mask] = 0.0
            return out
        out[valid_mask] = (out[valid_mask] - mean) / std
    else:
        out[valid_mask] = (out[valid_mask] - median) / (1.4826 * mad)

    out = np.clip(out, -5.0, 5.0)
    out[~valid_mask] = 0.0
    return out


def load_galileo_encoder(repo_dir: str, model_dir: str, device: str = "cpu"):
    sys.path.insert(0, repo_dir)

    import torch
    from single_file_galileo import Encoder, SPACE_BANDS, SPACE_TIME_BANDS, TIME_BANDS, STATIC_BANDS

    model = Encoder.load_from_folder(Path(model_dir), device=torch.device(device))
    model.eval()

    dims = {
        "space_bands": len(SPACE_BANDS),
        "space_time_bands": len(SPACE_TIME_BANDS),
        "time_bands": len(TIME_BANDS),
        "static_bands": len(STATIC_BANDS),
    }
    return model, dims


def assign_features_to_galileo_slots(feature_names, dims):
    mapping = {
        "space": [],
        "space_time": [],
        "time": [],
        "static": [],
    }

    sb = dims["space_bands"]
    stb = dims["space_time_bands"]

    use_space = min(len(feature_names), sb)
    for i in range(use_space):
        mapping["space"].append((i, feature_names[i]))

    remaining = feature_names[use_space:]
    use_st = min(len(remaining), stb)
    for j in range(use_st):
        mapping["space_time"].append((j, remaining[j]))

    if len(remaining) > stb:
        extras = remaining[stb:]
        log(f"WARNING: features not placed into Galileo slots: {extras}")

    return mapping


def batched_pixel_iterator(X, valid_mask, transform, batch_size=512):
    rows, cols = np.where(valid_mask)
    total = len(rows)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        r = rows[start:end]
        c = cols[start:end]
        feats = X[:, r, c].T.astype(np.float32)
        xs, ys = rasterio.transform.xy(transform, r, c, offset="center")
        yield start, end, r, c, np.asarray(xs), np.asarray(ys), feats, total


def get_embedding_dim(model, dims, mapping, feature_names, X, valid_mask, transform, months_value=6, device="cpu"):
    import torch

    feat_idx = {f: i for i, f in enumerate(feature_names)}
    start, end, rows, cols, xs, ys, feats, total = next(
        batched_pixel_iterator(X, valid_mask, transform, batch_size=1)
    )
    B = feats.shape[0]

    space_x = torch.zeros((B, 1, 1, dims["space_bands"]), dtype=torch.float32, device=device)
    space_time_x = torch.zeros((B, 1, 1, 1, dims["space_time_bands"]), dtype=torch.float32, device=device)
    time_x = torch.zeros((B, 1, dims["time_bands"]), dtype=torch.float32, device=device)
    static_x = torch.zeros((B, dims["static_bands"]), dtype=torch.float32, device=device)

    space_mask = torch.ones((B, 1, 1, dims["space_bands"]), dtype=torch.int64, device=device)
    space_time_mask = torch.ones((B, 1, 1, 1, dims["space_time_bands"]), dtype=torch.int64, device=device)
    time_mask = torch.ones((B, 1, dims["time_bands"]), dtype=torch.int64, device=device)
    static_mask = torch.ones((B, dims["static_bands"]), dtype=torch.int64, device=device)
    months = torch.full((B, 1), int(months_value), dtype=torch.int64, device=device)

    for slot_idx, feat_name in mapping["space"]:
        col_idx = feat_idx[feat_name]
        vals = torch.from_numpy(feats[:, col_idx]).to(device)
        space_x[:, 0, 0, slot_idx] = vals
        space_mask[:, 0, 0, slot_idx] = 0

    for slot_idx, feat_name in mapping["space_time"]:
        col_idx = feat_idx[feat_name]
        vals = torch.from_numpy(feats[:, col_idx]).to(device)
        space_time_x[:, 0, 0, 0, slot_idx] = vals
        space_time_mask[:, 0, 0, 0, slot_idx] = 0

    with torch.no_grad():
        model_output = model(
            space_time_x.float(),
            space_x.float(),
            time_x.float(),
            static_x.float(),
            space_time_mask,
            space_mask,
            time_mask,
            static_mask,
            months.long(),
            patch_size=1,
        )
        emb = model.average_tokens(*model_output[:-1]).detach().cpu().numpy()

    emb_dim = emb.shape[1]
    del space_x, space_time_x, time_x, static_x
    del space_mask, space_time_mask, time_mask, static_mask
    del months, model_output, emb
    gc.collect()
    return emb_dim, total


def make_pca_png_from_npy(npy_path, metadata_csv, H, W, out_png, max_fit_samples=5000, chunk=10000):
    meta = pd.read_csv(metadata_csv)
    arr = np.load(npy_path, mmap_mode="r")
    total_rows = arr.shape[0]

    fit_idx = np.linspace(0, total_rows - 1, min(max_fit_samples, total_rows), dtype=int)
    pca = PCA(n_components=3, random_state=42)
    pca.fit(np.array(arr[fit_idx], dtype=np.float32))

    rgb = np.empty((total_rows, 3), dtype=np.float32)
    for start in range(0, total_rows, chunk):
        end = min(start + chunk, total_rows)
        rgb[start:end] = pca.transform(np.array(arr[start:end], dtype=np.float32)).astype(np.float32)

    for k in range(3):
        ch = rgb[:, k]
        lo, hi = np.percentile(ch, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
        rgb[:, k] = np.clip((ch - lo) / (hi - lo), 0, 1)

    canvas = np.full((H, W, 3), np.nan, dtype=np.float32)
    rr = meta["row"].to_numpy(dtype=int)
    cc = meta["col"].to_numpy(dtype=int)
    canvas[rr, cc, :] = rgb

    plt.figure(figsize=(10, 8), dpi=180)
    plt.imshow(canvas)
    plt.title("Galileo Embedding PCA Preview")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    del arr, rgb, canvas, meta
    gc.collect()


def write_summary_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_embedding_job(
    bucket: str,
    fused_key: str,
    out_prefix: str,
    feature_names: list,
    galileo_repo: str,
    galileo_model_dir: str,
    workdir: str = "/tmp/galileo_embed_job",
    batch_size: int = 512,
    months_value: int = 6,
    device: str = "cpu",
    save_pca_png: bool = False,
):
    import torch

    os.makedirs(workdir, exist_ok=True)
    local_nc = os.path.join(workdir, os.path.basename(fused_key))
    s3_download(bucket, fused_key, local_nc)

    ds = xr.open_dataset(local_nc)
    crs, transform = get_crs_transform(ds)

    arrays = []
    for name in feature_names:
        if name not in ds:
            raise KeyError(f"Feature '{name}' not found in fused dataset.")
        arrays.append(to_2d(ds[name]))

    X = np.stack(arrays, axis=0)
    _, H, W = X.shape
    valid_mask = np.isfinite(X).all(axis=0)

    # once arrays are loaded into RAM, delete local nc to save disk
    ds.close()
    del ds
    try:
        os.remove(local_nc)
    except OSError:
        pass

    Xn = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        Xn[i] = robust_zscore(X[i], valid_mask)

    model, dims = load_galileo_encoder(
        repo_dir=galileo_repo,
        model_dir=galileo_model_dir,
        device=device,
    )

    mapping = assign_features_to_galileo_slots(feature_names, dims)
    emb_dim, total_rows = get_embedding_dim(
        model, dims, mapping, feature_names, Xn, valid_mask, transform,
        months_value=months_value, device=device
    )

    out_dir = os.path.join(workdir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    final_npy = os.path.join(out_dir, "galileo_embeddings.npy")
    final_csv = os.path.join(out_dir, "galileo_metadata.csv")
    final_png = os.path.join(out_dir, "galileo_embedding_rgb_pca.png")
    final_json = os.path.join(out_dir, "galileo_summary.json")

    # create final .npy directly, as float16 to reduce disk
    emb_out = open_memmap(final_npy, mode="w+", dtype=np.float16, shape=(total_rows, emb_dim))

    feat_idx = {f: i for i, f in enumerate(feature_names)}
    header_written = False

    for start, end, rows, cols, xs, ys, feats, total in batched_pixel_iterator(
        Xn, valid_mask, transform, batch_size=batch_size
    ):
        B = feats.shape[0]

        space_x = torch.zeros((B, 1, 1, dims["space_bands"]), dtype=torch.float32, device=device)
        space_time_x = torch.zeros((B, 1, 1, 1, dims["space_time_bands"]), dtype=torch.float32, device=device)
        time_x = torch.zeros((B, 1, dims["time_bands"]), dtype=torch.float32, device=device)
        static_x = torch.zeros((B, dims["static_bands"]), dtype=torch.float32, device=device)

        space_mask = torch.ones((B, 1, 1, dims["space_bands"]), dtype=torch.int64, device=device)
        space_time_mask = torch.ones((B, 1, 1, 1, dims["space_time_bands"]), dtype=torch.int64, device=device)
        time_mask = torch.ones((B, 1, dims["time_bands"]), dtype=torch.int64, device=device)
        static_mask = torch.ones((B, dims["static_bands"]), dtype=torch.int64, device=device)

        months = torch.full((B, 1), int(months_value), dtype=torch.int64, device=device)

        for slot_idx, feat_name in mapping["space"]:
            col_idx = feat_idx[feat_name]
            vals = torch.from_numpy(feats[:, col_idx]).to(device)
            space_x[:, 0, 0, slot_idx] = vals
            space_mask[:, 0, 0, slot_idx] = 0

        for slot_idx, feat_name in mapping["space_time"]:
            col_idx = feat_idx[feat_name]
            vals = torch.from_numpy(feats[:, col_idx]).to(device)
            space_time_x[:, 0, 0, 0, slot_idx] = vals
            space_time_mask[:, 0, 0, 0, slot_idx] = 0

        with torch.no_grad():
            model_output = model(
                space_time_x.float(),
                space_x.float(),
                time_x.float(),
                static_x.float(),
                space_time_mask,
                space_mask,
                time_mask,
                static_mask,
                months.long(),
                patch_size=1,
            )
            emb = model.average_tokens(*model_output[:-1]).detach().cpu().numpy().astype(np.float16)

        emb_out[start:end] = emb
        emb_out.flush()

        meta_df = pd.DataFrame({
            "row": rows,
            "col": cols,
            "x": xs,
            "y": ys,
            "embedding_index": np.arange(start, end, dtype=int),
        })
        meta_df.to_csv(final_csv, index=False, mode="a", header=(not header_written))
        header_written = True

        log(f"Processed {end}/{total} pixels")

        del space_x, space_time_x, time_x, static_x
        del space_mask, space_time_mask, time_mask, static_mask
        del months, model_output, emb, meta_df
        gc.collect()

    del emb_out
    gc.collect()

    if save_pca_png:
        make_pca_png_from_npy(final_npy, final_csv, H, W, final_png)

    summary = {
        "bucket": bucket,
        "fused_key": fused_key,
        "out_prefix": out_prefix,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "height": int(H),
        "width": int(W),
        "valid_pixels": int(valid_mask.sum()),
        "embedding_dim": int(emb_dim),
        "dtype": "float16",
        "months_value": int(months_value),
        "batch_size": int(batch_size),
        "galileo_dims": dims,
        "slot_mapping": mapping,
        "crs": str(crs),
        "pca_png_saved": bool(save_pca_png),
        "note": "Lightweight Galileo embeddings generated specifically for this Josef 3-model subset.",
    }
    write_summary_json(final_json, summary)

    s3_upload(bucket, final_npy, f"{out_prefix}/galileo_embeddings.npy")
    s3_upload(bucket, final_csv, f"{out_prefix}/galileo_metadata.csv")
    s3_upload(bucket, final_json, f"{out_prefix}/galileo_summary.json")

    if save_pca_png and os.path.exists(final_png):
        s3_upload(bucket, final_png, f"{out_prefix}/galileo_embedding_rgb_pca.png")

    # cleanup local files aggressively
    for p in [final_npy, final_csv, final_json, final_png]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass

    log(f"Uploaded outputs to s3://{bucket}/{out_prefix}/")
