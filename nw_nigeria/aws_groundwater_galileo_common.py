#!/usr/bin/env python3
import os
import gc
import json
import time
import tempfile
import warnings

import boto3
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import rasterio
from rasterio.transform import Affine
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import distance_transform_edt, convolve

warnings.filterwarnings("ignore")

s3 = boto3.client("s3")
CLASS_COLORS = ["#d7191c", "#fdae61", "#ffff66", "#a6d96a", "#1a9641"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def s3_download(bucket, key, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    s3.download_file(bucket, key, out_path)


def s3_upload(bucket, local_path, key):
    s3.upload_file(local_path, bucket, key)


def infer_transform_from_xy(ds: xr.Dataset):
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("Dataset missing x/y coordinates; cannot infer transform.")
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
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


def to_2d(da):
    da = da.squeeze(drop=True)
    while len(da.dims) > 2:
        extra = [d for d in da.dims if d not in ("y", "x")][0]
        da = da.isel({extra: 0}).squeeze(drop=True)
    if da.dims != ("y", "x"):
        da = da.transpose("y", "x")
    return da.values.astype("float32")


def fill_nodata_nearest(arr, fill_mask):
    out = arr.copy()
    valid = np.isfinite(out) & (~fill_mask)
    inds = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    out[fill_mask] = out[inds[0][fill_mask], inds[1][fill_mask]]
    return out


def build_filled_stack(raw_layers):
    stack_raw = np.stack(raw_layers, axis=0)
    support_mask = np.isfinite(stack_raw).any(axis=0)

    filled_layers = []
    for arr in raw_layers:
        arr2 = arr.copy()
        missing_inside = support_mask & (~np.isfinite(arr2))
        if missing_inside.sum() > 0:
            arr2 = fill_nodata_nearest(arr2, missing_inside)
        filled_layers.append(arr2.astype("float32"))

    stack_filled = np.stack(filled_layers, axis=0)
    final_mask = support_mask & np.isfinite(stack_filled).all(axis=0)
    return stack_filled, final_mask


def spatial_block_split(rows, cols, H, W, n_blocks=6, seed=42):
    rng = np.random.default_rng(seed)
    br = np.floor(rows / (H / n_blocks)).astype(int)
    bc = np.floor(cols / (W / n_blocks)).astype(int)
    tile_id = br * n_blocks + bc
    unique_tiles = np.unique(tile_id)
    rng.shuffle(unique_tiles)
    n_test_tiles = max(1, int(0.2 * len(unique_tiles)))
    test_tiles = set(unique_tiles[:n_test_tiles])
    is_test = np.array([t in test_tiles for t in tile_id], dtype=bool)
    return ~is_test, is_test


def safe_auc(y_true, prob):
    return float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, prob))


def write_geotiff(path, arr, crs, transform, nodata=0):
    h, w = arr.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": arr.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def plot_class_map(arr, out_png, title):
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
    cbar.set_ticklabels(["Very High", "High", "Moderate", "Low", "Very Low"])
    cbar.set_label("Groundwater Potential Class")
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_conf_matrix(y_true, y_pred, out_png, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)

    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title(f"{title}\nRaw")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=10)

    axes[1].imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title(f"{title}\nNormalized")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center", fontsize=10)

    for ax in axes:
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(df, out_csv):
    df = df.sort_values(["model", "importance"], ascending=[True, False])
    df.to_csv(out_csv, index=False)


def high_zone_stats(class_map):
    mask = np.isin(class_map, [1, 2])
    return {
        "high_plus_veryhigh_pixels": int(mask.sum()),
        "high_plus_veryhigh_fraction": float(mask.mean())
    }


def prob_to_5_classes(prob_vals):
    qs = np.quantile(prob_vals, [0.2, 0.4, 0.6, 0.8])
    out = np.full(prob_vals.shape, 3, dtype=np.uint8)
    out[prob_vals >= qs[3]] = 1
    out[(prob_vals >= qs[2]) & (prob_vals < qs[3])] = 2
    out[(prob_vals >= qs[1]) & (prob_vals < qs[2])] = 3
    out[(prob_vals >= qs[0]) & (prob_vals < qs[1])] = 4
    out[prob_vals < qs[0]] = 5
    return out


def fill_internal_class_holes(class_map, aoi_mask):
    out = class_map.copy()
    hole_mask = (aoi_mask == 1) & (out == 0)
    if not hole_mask.any():
        return out

    valid = (out > 0)
    inds = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    out[hole_mask] = out[inds[0][hole_mask], inds[1][hole_mask]]
    return out


def conservative_isolated_pixel_cleanup(class_map, aoi_mask):
    out = class_map.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)

    counts = []
    for c in range(1, 6):
        counts.append(convolve((out == c).astype(np.uint8), kernel, mode="constant", cval=0))
    counts = np.stack(counts, axis=0)

    current_idx = np.clip(out - 1, 0, 4)
    same_count = np.take_along_axis(counts, current_idx[None, :, :], axis=0)[0]
    majority_idx = np.argmax(counts, axis=0)
    majority_count = np.max(counts, axis=0)

    replace_mask = (
        (aoi_mask == 1) &
        (out > 0) &
        (same_count <= 2) &
        (majority_count >= 7)
    )
    out[replace_mask] = (majority_idx[replace_mask] + 1).astype(np.uint8)
    return out


def build_final_class_map_from_probs(prob_vals, rows, cols, H, W, aoi_mask):
    class_valid = prob_to_5_classes(prob_vals.astype(np.float32))

    full_arr = np.zeros((H, W), dtype=np.uint8)
    full_arr[rows, cols] = class_valid

    full_arr = fill_internal_class_holes(full_arr, aoi_mask)
    full_arr = conservative_isolated_pixel_cleanup(full_arr, aoi_mask)
    full_arr[aoi_mask == 0] = 0
    return full_arr


def build_pseudolabels_from_raw(ds, feature_order, weights):
    raw_layers = [to_2d(ds[v]) for v in feature_order]
    stack_filled, final_mask = build_filled_stack(raw_layers)
    rows, cols = np.where(final_mask)
    H, W = stack_filled.shape[1], stack_filled.shape[2]

    X_raw = stack_filled[:, final_mask].T.astype("float32")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_raw).astype("float32")

    log("Running GMM for pseudo-labels...")
    gmm = GaussianMixture(
        n_components=len(feature_order),
        covariance_type="diag",
        reg_covar=1e-4,
        random_state=42,
        n_init=2,
        max_iter=150,
    )
    gmm.fit(Xs.astype("float64"))
    gmm_labels = gmm.predict(Xs.astype("float64"))

    df = pd.DataFrame({"cluster": gmm_labels + 1})
    for i, v in enumerate(feature_order):
        df[v] = X_raw[:, i]

    cluster_means = df.groupby("cluster")[feature_order].mean().reset_index()
    for v in feature_order:
        mu = cluster_means[v].mean()
        sd = cluster_means[v].std(ddof=0)
        cluster_means[f"z_{v}"] = 0.0 if (sd == 0 or not np.isfinite(sd)) else (cluster_means[v] - mu) / sd

    cluster_means["gw_score"] = 0.0
    for v in feature_order:
        cluster_means["gw_score"] += weights[v] * cluster_means[f"z_{v}"]

    cluster_means = cluster_means.sort_values("gw_score", ascending=False).reset_index(drop=True)
    cluster_means["rank_1_best"] = np.arange(1, len(cluster_means) + 1)
    cluster_to_rank = dict(zip(cluster_means["cluster"].astype(int), cluster_means["rank_1_best"].astype(int)))
    pseudo_rank = np.array([cluster_to_rank[int(c)] for c in (gmm_labels + 1)], dtype=np.uint16)

    top_k = max(1, int(np.ceil(len(feature_order) * 0.20)))
    y = (pseudo_rank <= top_k).astype(np.uint8)

    return {
        "rows": rows,
        "cols": cols,
        "H": H,
        "W": W,
        "y": y,
        "final_mask": final_mask,
    }


def run_galileo_model_job(
    bucket,
    fused_key,
    emb_key,
    meta_key,
    out_prefix,
    feature_order,
    weights,
    subset_name,
    max_train_samples=120000,
    rf_trees=300,
    xgb_estimators=400,
    cat_iters=500,
):
    with tempfile.TemporaryDirectory() as td:
        fused_local = os.path.join(td, f"fused_{subset_name}.nc4")
        emb_local = os.path.join(td, "galileo_embeddings.npy")
        meta_local = os.path.join(td, "galileo_metadata.csv")

        log("Downloading fused dataset, embeddings, metadata...")
        s3_download(bucket, fused_key, fused_local)
        s3_download(bucket, emb_key, emb_local)
        s3_download(bucket, meta_key, meta_local)

        ds = xr.open_dataset(fused_local)
        crs, transform = get_crs_transform(ds)

        pseudo = build_pseudolabels_from_raw(ds, feature_order, weights)
        rows = pseudo["rows"]
        cols = pseudo["cols"]
        H = pseudo["H"]
        W = pseudo["W"]
        y_raw = pseudo["y"]

        aoi_mask = np.zeros((H, W), dtype=np.uint8)
        aoi_mask[rows, cols] = 1

        log("Loading Galileo embeddings and metadata...")
        X_emb = np.load(emb_local, mmap_mode="r")
        meta = pd.read_csv(meta_local)

        merged = pd.DataFrame({
            "row": rows.astype(int),
            "col": cols.astype(int),
            "y": y_raw.astype(np.uint8),
        }).merge(meta[["row", "col", "embedding_index"]], on=["row", "col"], how="inner")

        if len(merged) == 0:
            raise RuntimeError("No aligned samples between pseudo-label pixels and Galileo metadata.")

        log(f"Aligned Galileo samples: {len(merged):,}")

        emb_idx = merged["embedding_index"].to_numpy(dtype=int)
        y = merged["y"].to_numpy(dtype=np.uint8)
        rr = merged["row"].to_numpy(dtype=int)
        cc = merged["col"].to_numpy(dtype=int)

        X = np.array(X_emb[emb_idx], dtype=np.float32)

        is_train, is_test = spatial_block_split(rr, cc, H, W, seed=42)
        X_train, X_test = X[is_train], X[is_test]
        y_train, y_test = y[is_train], y[is_test]

        if max_train_samples is not None and len(X_train) > max_train_samples:
            rng = np.random.default_rng(42)
            keep = rng.choice(len(X_train), size=max_train_samples, replace=False)
            X_train = X_train[keep]
            y_train = y_train[keep]

        cw = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
        class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}

        log("Training RF...")
        rf = RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=2,
            random_state=42,
            class_weight=class_weight_dict
        )
        rf.fit(X_train, y_train)
        rf_prob_test = rf.predict_proba(X_test)[:, 1]
        rf_prob_all = rf.predict_proba(X)[:, 1]

        log("Training XGB...")
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        xgb = XGBClassifier(
            n_estimators=xgb_estimators,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2,
            eval_metric="logloss",
            tree_method="hist",
            max_bin=128,
            scale_pos_weight=neg / pos,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        xgb_prob_test = xgb.predict_proba(X_test)[:, 1]
        xgb_prob_all = xgb.predict_proba(X)[:, 1]

        log("Training CatBoost...")
        cat = CatBoostClassifier(
            iterations=cat_iters,
            depth=4,
            learning_rate=0.06,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            thread_count=2,
            verbose=False
        )
        cat.fit(X_train, y_train)
        cat_prob_test = cat.predict_proba(X_test)[:, 1]
        cat_prob_all = cat.predict_proba(X)[:, 1]

        ens_cat_rf_test = 0.5 * (cat_prob_test + rf_prob_test)
        ens_cat_xgb_test = 0.5 * (cat_prob_test + xgb_prob_test)
        ens_cat_rf_all = 0.5 * (cat_prob_all + rf_prob_all)
        ens_cat_xgb_all = 0.5 * (cat_prob_all + xgb_prob_all)

        pred_dict_test = {
            "rf": (rf_prob_test >= 0.5).astype(np.uint8),
            "xgb": (xgb_prob_test >= 0.5).astype(np.uint8),
            "catboost": (cat_prob_test >= 0.5).astype(np.uint8),
            "ensemble_cat_rf": (ens_cat_rf_test >= 0.5).astype(np.uint8),
            "ensemble_cat_xgb": (ens_cat_xgb_test >= 0.5).astype(np.uint8),
        }

        metrics = {}
        for name, prob_test in {
            "rf": rf_prob_test,
            "xgb": xgb_prob_test,
            "catboost": cat_prob_test,
            "ensemble_cat_rf": ens_cat_rf_test,
            "ensemble_cat_xgb": ens_cat_xgb_test,
        }.items():
            metrics[name] = {
                "accuracy": float(accuracy_score(y_test, pred_dict_test[name])),
                "f1": float(f1_score(y_test, pred_dict_test[name])),
                "auc": safe_auc(y_test, prob_test),
            }

        metrics["note"] = "Metrics are against pseudo-labels, not groundwater ground truth."
        metrics["representation"] = "galileo_embeddings"
        metrics["feature_set"] = subset_name
        metrics["features"] = feature_order
        metrics["postprocess"] = {
            "hole_fill_inside_aoi": True,
            "isolated_pixel_cleanup": True,
            "gaussian_smoothing": False,
            "median_smoothing": False,
            "blob_smoothing": False,
        }

        rf_imp = pd.DataFrame({
            "feature": [f"emb_{i}" for i in range(X.shape[1])],
            "importance": rf.feature_importances_,
            "model": "rf"
        })
        xgb_imp = pd.DataFrame({
            "feature": [f"emb_{i}" for i in range(X.shape[1])],
            "importance": xgb.feature_importances_,
            "model": "xgb"
        })
        cat_imp = pd.DataFrame({
            "feature": [f"emb_{i}" for i in range(X.shape[1])],
            "importance": cat.get_feature_importance(),
            "model": "catboost"
        })
        feat_imp = pd.concat([rf_imp, xgb_imp, cat_imp], ignore_index=True)

        for name, ypred in pred_dict_test.items():
            cm_png = os.path.join(td, f"confusion_{name}.png")
            plot_conf_matrix(y_test, ypred, cm_png, f"{name.upper()} confusion")
            s3_upload(bucket, cm_png, f"{out_prefix}/{os.path.basename(cm_png)}")

        prob_dict_all = {
            f"RF_GALILEO_{subset_name.upper()}_5class": rf_prob_all,
            f"XGB_GALILEO_{subset_name.upper()}_5class": xgb_prob_all,
            f"CAT_GALILEO_{subset_name.upper()}_5class": cat_prob_all,
            f"CAT_RF_GALILEO_{subset_name.upper()}_ENSEMBLE_5class": ens_cat_rf_all,
            f"CAT_XGB_GALILEO_{subset_name.upper()}_ENSEMBLE_5class": ens_cat_xgb_all,
        }

        zone_stats = {}
        for name, probs in prob_dict_all.items():
            full_arr = build_final_class_map_from_probs(
                prob_vals=probs,
                rows=rr,
                cols=cc,
                H=H,
                W=W,
                aoi_mask=aoi_mask,
            )

            tif_path = os.path.join(td, f"{name}.tif")
            png_path = os.path.join(td, f"{name}.png")
            write_geotiff(tif_path, full_arr, crs, transform, nodata=0)
            plot_class_map(full_arr, png_path, name.replace("_", " "))

            s3_upload(bucket, tif_path, f"{out_prefix}/{os.path.basename(tif_path)}")
            s3_upload(bucket, png_path, f"{out_prefix}/{os.path.basename(png_path)}")

            zone_stats[name] = high_zone_stats(full_arr)

        metrics_path = os.path.join(td, f"metrics_galileo_{subset_name}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        s3_upload(bucket, metrics_path, f"{out_prefix}/metrics_galileo.json")

        feat_path = os.path.join(td, f"feature_importance_galileo_{subset_name}.csv")
        save_feature_importance(feat_imp, feat_path)
        s3_upload(bucket, feat_path, f"{out_prefix}/feature_importance_galileo.csv")

        zone_path = os.path.join(td, f"high_zone_stats_galileo_{subset_name}.json")
        with open(zone_path, "w") as f:
            json.dump(zone_stats, f, indent=2)
        s3_upload(bucket, zone_path, f"{out_prefix}/high_zone_stats_galileo.json")

        ds.close()
        del ds, X_emb, meta, merged, X, X_train, X_test, y_train, y_test
        gc.collect()

        log("Done.")
