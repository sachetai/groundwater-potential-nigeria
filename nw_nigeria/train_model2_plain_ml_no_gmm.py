#!/usr/bin/env python3

import os, json, warnings
from pathlib import Path
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

warnings.filterwarnings("ignore")

# ================= CONFIG =================
BUCKET = "sachet-imerg-nigeria"

S3_OUT_PREFIX = "output/final_models/model2_plain_ml_no_gmm"

WORK = Path("/tmp/model2_plain_ml_no_gmm")
OUT = WORK / "outputs"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MAX_TRAIN_SAMPLES = 180000
PRED_CHUNK = 80000

# Model 2 thematic layers only, NO GMM, NO VES
LAYERS = {
    "Rainfall": "datasets_2024/NW_Rainfall_2024_total.tif",
    "Lithology": "datasets_2024/Lithology.tif",
    "Lineament_Density": "datasets_2024/Lineament_Density.tif",
    "TWI": "datasets_2024/TWI.tif",
    "Slope": "datasets_2024/Slope.tif",
    "Soil": "datasets_2024/Soil.tif",
    "NDWI": "datasets_2024/NDWI.tif",
    "Drainage_Density": "datasets_2024/Drainage_Density.tif",
    "NDVI": "datasets_2024/NDVI.tif",
    "LULC": "datasets_2024/LULC.tif",
}

AOI_KEY = "datasets_2024/NW_AOI_Original_Boundary.tif"

# Domain direction
POSITIVE = ["Rainfall", "Lineament_Density", "TWI", "NDWI", "NDVI"]
NEGATIVE = ["Slope", "Drainage_Density"]
CATEGORICAL = ["Lithology", "Soil", "LULC"]

# Advisor priority-based relative importance
RAW_WEIGHTS = {
    "Rainfall": 10,
    "Lithology": 9,
    "Lineament_Density": 8,
    "TWI": 7,
    "Slope": 6,
    "Soil": 5,
    "NDWI": 4,
    "Drainage_Density": 3,
    "NDVI": 2,
    "LULC": 1,
}

CLASS_NAMES = {
    1: "Very High",
    2: "High",
    3: "Moderate",
    4: "Low",
    5: "Very Low",
}

COLORS = ["#e31a1c", "#fdae61", "#ffff66", "#a6d96a", "#1a9850"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], CMAP.N)

s3 = boto3.client("s3")


# ================= HELPERS =================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def s3_download(key, local_name):
    path = WORK / local_name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        log(f"Downloading s3://{BUCKET}/{key}")
        s3.download_file(BUCKET, key, str(path))
    return path


def upload_folder(local_dir, s3_prefix):
    for p in Path(local_dir).rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            s3.upload_file(str(p), BUCKET, f"{s3_prefix}/{rel}")


def safe_norm(x, mask):
    vals = x[mask & np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.nanpercentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1).astype(np.float32)


def read_reference_aoi(aoi_path):
    with rasterio.open(aoi_path) as src:
        aoi = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    mask = np.isfinite(aoi)
    if np.nanmax(aoi) > 0:
        mask = mask & (aoi > 0)

    if mask.sum() == 0:
        mask = np.isfinite(aoi)

    return mask, profile, transform, crs, aoi.shape


def align_to_reference(src_path, ref_profile, ref_transform, ref_crs, ref_shape, resampling):
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        dst = np.full(ref_shape, np.nan, dtype=np.float32)

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return dst


def write_single_tif(path, arr, profile, dtype="uint8", nodata=0):
    prof = profile.copy()
    prof.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def write_float_tif(path, arr, profile):
    prof = profile.copy()
    prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=2)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)


def write_multiband_tif(path, stack, profile):
    prof = profile.copy()
    prof.update(count=stack.shape[0], dtype="float32", nodata=np.nan, compress="deflate", predictor=2)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(stack.astype(np.float32))


def save_map_png(path, arr, title):
    masked = np.ma.masked_where(arr == 0, arr)
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(masked, cmap=CMAP, norm=NORM)
    ax.set_title(title, fontsize=18)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03, ticks=[1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels(["Very High", "High", "Moderate", "Low", "Very Low"])
    cbar.set_label("Groundwater Potential Class", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_confusion_png(path, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(["VH", "H", "M", "L", "VL"])
    ax.set_yticklabels(["VH", "H", "M", "L", "VL"])

    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]*100:.1f}%", ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def one_vs_rest(y_true, y_pred):
    total = len(y_true)
    rows = []
    for c in [1, 2, 3, 4, 5]:
        t = y_true == c
        p = y_pred == c
        tp = int(np.sum(t & p))
        tn = int(np.sum(~t & ~p))
        fp = int(np.sum(~t & p))
        fn = int(np.sum(t & ~p))
        rows.append({
            "class": c,
            "label": CLASS_NAMES[c],
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP_percent": round(tp * 100 / total, 3),
            "TN_percent": round(tn * 100 / total, 3),
            "FP_percent": round(fp * 100 / total, 3),
            "FN_percent": round(fn * 100 / total, 3),
        })
    return pd.DataFrame(rows)


# ================= SCORE WITHOUT GMM =================
def categorical_score(layer, mask, feature_name):
    """
    No fixed external class table used.
    Data-driven categorical ranking:
    each category gets score based on mean of favorable continuous context later.
    For now returns normalized category codes as weak signal.
    """
    vals = layer[mask & np.isfinite(layer)]
    if vals.size == 0:
        return np.zeros_like(layer, dtype=np.float32)

    unique_vals = np.unique(vals)
    out = np.zeros_like(layer, dtype=np.float32)

    if unique_vals.size <= 1:
        return out

    # normalize category IDs only as neutral weak feature
    ranks = {v: i / (unique_vals.size - 1) for i, v in enumerate(sorted(unique_vals))}
    for v, s in ranks.items():
        out[layer == v] = s

    return np.clip(out, 0, 1).astype(np.float32)


def build_direct_score(norm_layers, mask):
    weights = {k: v / sum(RAW_WEIGHTS.values()) for k, v in RAW_WEIGHTS.items()}

    score = np.zeros(mask.shape, dtype=np.float32)
    score_parts = []

    for name, x in norm_layers.items():
        w = weights[name]

        if name in POSITIVE:
            contribution = x
            direction = "positive"
        elif name in NEGATIVE:
            contribution = 1.0 - x
            direction = "negative"
        else:
            contribution = x
            direction = "categorical_data_driven"

        score += w * contribution

        score_parts.append({
            "feature": name,
            "weight": w,
            "direction": direction,
            "s3_key": LAYERS[name],
        })

    score = safe_norm(score, mask)
    return score, pd.DataFrame(score_parts)


def score_to_5class(score, mask):
    vals = score[mask & np.isfinite(score)]

    # Quantile-based 5 classes ensures all classes exist
    q80, q60, q40, q20 = np.nanpercentile(vals, [80, 60, 40, 20])

    labels = np.zeros(score.shape, dtype=np.uint8)
    labels[mask & (score >= q80)] = 1
    labels[mask & (score < q80) & (score >= q60)] = 2
    labels[mask & (score < q60) & (score >= q40)] = 3
    labels[mask & (score < q40) & (score >= q20)] = 4
    labels[mask & (score < q20)] = 5

    return labels


# ================= ML =================
def prepare_ml_data(stack, labels, mask):
    X = stack[:, mask].T.astype(np.float32)
    y = labels[mask].astype(np.uint8)

    good = np.isfinite(X).all(axis=1) & np.isin(y, [1, 2, 3, 4, 5])
    X = X[good]
    y = y[good]

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    rng = np.random.default_rng(RANDOM_STATE)
    train_ids = []
    y0 = y - 1
    per_class = max(3000, MAX_TRAIN_SAMPLES // 5)

    for c in range(5):
        ids = np.where(y0 == c)[0]
        take = min(per_class, ids.size)
        train_ids.append(rng.choice(ids, size=take, replace=False))

    train_ids = np.concatenate(train_ids)
    rng.shuffle(train_ids)

    X_train = X[train_ids]
    y_train = y0[train_ids]

    log(f"Training samples: {len(X_train):,}")
    log(f"Prediction pixels: {len(X):,}")

    return X_train, y_train, X, y, good


def train_models(X_train, y_train):
    models = {}

    log("Training RF")
    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=16,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    models["rf"] = rf

    log("Training XGB")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train)
    models["xgb"] = xgb

    log("Training CatBoost")
    cat = CatBoostClassifier(
        iterations=350,
        depth=7,
        learning_rate=0.055,
        l2_leaf_reg=5,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    cat.fit(X_train, y_train)
    models["catboost"] = cat

    return models


def predict_chunked(model, X, name):
    parts = []
    for i in tqdm(range(0, len(X), PRED_CHUNK), desc=f"Predicting {name}"):
        parts.append(model.predict_proba(X[i:i + PRED_CHUNK]).astype(np.float32))
    return np.vstack(parts)


def compute_metrics(y_true1, probs, preds):
    rows = []
    y0 = y_true1 - 1

    for name in preds:
        p1 = preds[name]
        p0 = p1 - 1
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y0, p0),
            "f1_macro": f1_score(y0, p0, average="macro"),
            "f1_weighted": f1_score(y0, p0, average="weighted"),
            "precision_macro": precision_score(y0, p0, average="macro", zero_division=0),
            "recall_macro": recall_score(y0, p0, average="macro", zero_division=0),
            "mean_confidence": float(np.mean(np.max(probs[name], axis=1))),
        })

    return pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False)


def map_from_valid(pred_valid, final_mask, shape):
    arr = np.zeros(shape, dtype=np.uint8)
    arr[final_mask] = pred_valid.astype(np.uint8)
    return arr


# ================= MAIN =================
def main():
    log("Downloading AOI")
    aoi_path = s3_download(AOI_KEY, "aoi.tif")
    aoi_mask, ref_profile, ref_transform, ref_crs, ref_shape = read_reference_aoi(aoi_path)

    aligned_layers = {}
    norm_layers = {}

    log("Downloading and aligning Model 2 layers")
    for name, key in tqdm(LAYERS.items(), desc="Layers"):
        local_path = s3_download(key, f"inputs/{Path(key).name}")

        resampling = Resampling.nearest if name in CATEGORICAL else Resampling.bilinear
        arr = align_to_reference(local_path, ref_profile, ref_transform, ref_crs, ref_shape, resampling)

        aligned_layers[name] = arr

    valid = aoi_mask.copy()
    for arr in aligned_layers.values():
        valid &= np.isfinite(arr)

    log(f"Valid AOI pixels: {valid.sum():,}")

    # Normalize each input
    for name, arr in aligned_layers.items():
        if name in CATEGORICAL:
            norm_layers[name] = categorical_score(arr, valid, name)
        else:
            norm_layers[name] = safe_norm(arr, valid)

    stack = np.stack([norm_layers[name] for name in LAYERS.keys()], axis=0).astype(np.float32)

    # Save fused 10-layer input
    fused_dir = OUT / "fused_model2_10_layers"
    fused_dir.mkdir(parents=True, exist_ok=True)
    write_multiband_tif(fused_dir / "model2_10layer_aoi_aligned_normalized_stack.tif", stack, ref_profile)

    with open(fused_dir / "model2_10layer_band_names.txt", "w") as f:
        for i, name in enumerate(LAYERS.keys(), 1):
            f.write(f"{i}: {name} | {LAYERS[name]}\n")

    # Direct non-GMM score and labels
    log("Creating direct non-GMM groundwater score")
    score, weight_df = build_direct_score(norm_layers, valid)
    direct_labels = score_to_5class(score, valid)

    weight_df.to_csv(OUT / "model2_plain_ml_no_gmm_feature_weights.csv", index=False)

    write_float_tif(OUT / "model2_plain_ml_no_gmm_direct_score.tif", score, ref_profile)
    write_single_tif(OUT / "model2_plain_ml_no_gmm_direct_5class_labels.tif", direct_labels, ref_profile)
    save_map_png(OUT / "model2_plain_ml_no_gmm_direct_5class_labels.png", direct_labels,
                 "Model 2 Direct 10-Layer Score Labels (No GMM)")

    # ML training
    X_train, y_train, X_all, y_all1, good = prepare_ml_data(stack, direct_labels, valid)

    valid_flat = np.where(valid.ravel())[0]
    final_flat = np.zeros(valid.size, dtype=bool)
    final_flat[valid_flat[good]] = True
    final_mask = final_flat.reshape(valid.shape)

    models = train_models(X_train, y_train)

    log("Predicting base models")
    probs = {}
    for name, model in models.items():
        probs[name] = predict_chunked(model, X_all, name)

    log("Creating ensembles")
    probs["cat_xgb"] = 0.60 * probs["catboost"] + 0.40 * probs["xgb"]
    probs["cat_rf"] = 0.70 * probs["catboost"] + 0.30 * probs["rf"]

    preds = {}
    confs = {}
    for name in probs:
        preds[name] = np.argmax(probs[name], axis=1).astype(np.uint8) + 1
        confs[name] = np.max(probs[name], axis=1).astype(np.float32)

    summary = compute_metrics(y_all1, probs, preds)
    summary.to_csv(OUT / "model2_plain_ml_no_gmm_model_comparison_summary.csv", index=False)

    best_model = str(summary.iloc[0]["model"])
    log(f"Best model: {best_model}")

    # Save outputs per model
    for name in tqdm(preds.keys(), desc="Saving outputs"):
        model_dir = OUT / name
        model_dir.mkdir(parents=True, exist_ok=True)

        pred_map = map_from_valid(preds[name], final_mask, valid.shape)
        pred_map_clean = pred_map.copy()
        pred_map_clean[final_mask] = median_filter(pred_map, size=3)[final_mask]

        conf_map = np.full(valid.shape, np.nan, dtype=np.float32)
        conf_map[final_mask] = confs[name]

        write_single_tif(model_dir / f"model2_plain_ml_no_gmm_{name}_5class_map.tif", pred_map_clean, ref_profile)
        write_float_tif(model_dir / f"model2_plain_ml_no_gmm_{name}_confidence.tif", conf_map, ref_profile)

        save_map_png(
            model_dir / f"model2_plain_ml_no_gmm_{name}_5class_map.png",
            pred_map_clean,
            f"Model 2 Plain ML No GMM - {name.upper()}"
        )

        save_confusion_png(
            model_dir / f"model2_plain_ml_no_gmm_{name}_confusion_matrix.png",
            y_all1,
            preds[name],
            f"Model 2 Plain ML No GMM {name.upper()} Confusion Matrix"
        )

        one_vs_rest(y_all1, preds[name]).to_csv(
            model_dir / f"model2_plain_ml_no_gmm_{name}_tp_tn_fp_fn_one_vs_rest.csv",
            index=False
        )

        report = classification_report(
            y_all1,
            preds[name],
            labels=[1, 2, 3, 4, 5],
            target_names=[CLASS_NAMES[i] for i in [1, 2, 3, 4, 5]],
            output_dict=True,
            zero_division=0,
        )

        with open(model_dir / f"model2_plain_ml_no_gmm_{name}_classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

    with open(OUT / "README.txt", "w") as f:
        f.write("Model 2 Plain ML without GMM\n")
        f.write("============================\n\n")
        f.write("This run uses only the 10 thematic Model 2 layers:\n")
        for name, key in LAYERS.items():
            f.write(f"- {name}: {key}\n")
        f.write("\nNo GMM was used.\n")
        f.write("No VES was used.\n")
        f.write("A direct groundwater favorability score was created from the 10 inputs using domain directions.\n")
        f.write("The 5-class labels were created using quantiles of this direct score.\n")
        f.write("RF, XGB, CatBoost, CAT_XGB, and CAT_RF were trained on these direct labels.\n\n")
        f.write(f"Best model by metrics: {best_model}\n")

    log("Uploading outputs to S3")
    upload_folder(OUT, S3_OUT_PREFIX)

    log(f"DONE: s3://{BUCKET}/{S3_OUT_PREFIX}/")


if __name__ == "__main__":
    main()
