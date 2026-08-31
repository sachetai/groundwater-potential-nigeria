#!/usr/bin/env python3
import os
import json
import warnings
from pathlib import Path
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter, median_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

warnings.filterwarnings("ignore")

# ================= CONFIG =================
BUCKET = "sachet-imerg-nigeria"

S3_FUSED = "output/final_labels/fused_inputs/final_fused_13_layers_aoi_aligned.tif"
S3_LABELS = "output/final_labels/model2_extended/model2_extended_final_5class_pseudo_labels.tif"
S3_VES_SCORE = "output/final_labels/model2_extended/model2_extended_ves_calibrated_score.tif"
S3_VES_CONF = "output/final_labels/model2_extended/model2_extended_ves_confidence.tif"

S3_OUT_PREFIX = "output/final_models/galileo_models"

WORK = Path("/tmp/galileo_model2_optimized")
OUT = WORK / "outputs"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

MAX_TRAIN_SAMPLES = 150000
MAX_PCA_FIT_PIXELS = 180000
PCA_COMPONENTS = 28
PRED_CHUNK = 38000

SMOOTH_KERNEL = 3
MEDIAN_SIZE = 3
VES_WEIGHT_MULTIPLIER = 2.5

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
    if not path.exists() or path.stat().st_size == 0:
        log(f"Downloading s3://{BUCKET}/{key}")
        s3.download_file(BUCKET, key, str(path))
    return path


def upload_folder(local_dir, s3_prefix):
    for p in Path(local_dir).rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            s3.upload_file(str(p), BUCKET, f"{s3_prefix}/{rel}")


def read_stack(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def read_single(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


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


def safe_norm(x, mask):
    vals = x[mask & np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.nanpercentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1).astype(np.float32)


def fill_nan(x, mask):
    y = x.copy().astype(np.float32)
    vals = y[mask & np.isfinite(y)]
    fill = float(np.nanmedian(vals)) if vals.size else 0.0
    y[~np.isfinite(y)] = fill
    return y


# ================= EMBEDDINGS =================
def make_galileo_embeddings(fused, ves_score, ves_conf, mask):
    log("Creating Galileo-style embeddings")

    channels = []

    for i in tqdm(range(fused.shape[0]), desc="Embedding raw/local"):
        x = fill_nan(fused[i], mask)
        x = safe_norm(x, mask)

        channels.append(x)
        channels.append(uniform_filter(x, size=SMOOTH_KERNEL).astype(np.float32))

    important_band_ids = list(range(min(6, fused.shape[0])))

    for i in tqdm(important_band_ids, desc="Embedding gradients"):
        x = fill_nan(fused[i], mask)
        x = safe_norm(x, mask)

        gy, gx = np.gradient(x)
        grad = np.sqrt(gx * gx + gy * gy)

        channels.append(safe_norm(grad, mask))

    ves_score_n = safe_norm(fill_nan(ves_score, mask), mask)
    ves_conf_n = np.clip(np.nan_to_num(ves_conf, nan=0.0), 0, 1).astype(np.float32)

    channels.append(ves_score_n)
    channels.append(ves_conf_n)

    return np.stack(channels).astype(np.float32), ves_score_n, ves_conf_n


def reduce_pca_hybrid(emb, fused, ves_score_n, ves_conf_n, mask):
    """
    Hybrid features:
    Galileo PCA embeddings + original raw normalized input layers + VES score/confidence.
    This usually improves accuracy because the model keeps raw thematic signal plus spatial context.
    """
    X_emb = emb[:, mask].T.astype(np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_emb).astype(np.float32)

    rng = np.random.default_rng(RANDOM_STATE)
    n = min(MAX_PCA_FIT_PIXELS, Xs.shape[0])
    idx = rng.choice(Xs.shape[0], size=n, replace=False)

    n_comp = min(PCA_COMPONENTS, Xs.shape[1])
    log(f"PCA input dims={Xs.shape[1]}, components={n_comp}")

    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    pca.fit(Xs[idx])

    Z_emb = pca.transform(Xs).astype(np.float32)
    log(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    raw_channels = []
    for i in tqdm(range(fused.shape[0]), desc="Adding raw hybrid features"):
        x = fill_nan(fused[i], mask)
        x = safe_norm(x, mask)
        raw_channels.append(x)

    raw_channels.append(ves_score_n)
    raw_channels.append(ves_conf_n)

    raw_stack = np.stack(raw_channels).astype(np.float32)
    X_raw = raw_stack[:, mask].T.astype(np.float32)

    Z = np.hstack([Z_emb, X_raw]).astype(np.float32)

    log(f"Final hybrid feature dimensions: {Z.shape[1]}")
    return Z, scaler, pca


# ================= TRAINING DATA =================
def prepare_data(Z, labels, ves_conf, mask):
    y = labels[mask].astype(np.int32)
    conf = ves_conf[mask].astype(np.float32)

    good = np.isfinite(Z).all(axis=1) & np.isin(y, [1, 2, 3, 4, 5])
    Z = Z[good]
    y = y[good]
    conf = np.clip(np.nan_to_num(conf[good], nan=0.0), 0, 1)

    y0 = y - 1

    rng = np.random.default_rng(RANDOM_STATE)
    train_ids = []

    per_class = max(3000, MAX_TRAIN_SAMPLES // 5)

    for c in range(5):
        ids = np.where(y0 == c)[0]
        take = min(per_class, ids.size)
        if take > 0:
            train_ids.append(rng.choice(ids, size=take, replace=False))

    train_ids = np.concatenate(train_ids)
    rng.shuffle(train_ids)

    X_train = Z[train_ids]
    y_train = y0[train_ids]
    w_train = 1.0 + VES_WEIGHT_MULTIPLIER * conf[train_ids]

    log(f"Training samples: {len(X_train):,}")
    log(f"Prediction pixels: {len(Z):,}")

    return X_train, y_train, w_train, Z, y, good


# ================= MODELS =================
def train_models(X, y, w):
    models = {}

    log("Training RF")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=6,
        min_samples_split=14,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X, y, sample_weight=w)
    models["rf"] = rf

    log("Training XGB")
    xgb = XGBClassifier(
        n_estimators=320,
        max_depth=6,
        learning_rate=0.045,
        subsample=0.88,
        colsample_bytree=0.88,
        reg_lambda=1.8,
        reg_alpha=0.45,
        gamma=0.05,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    xgb.fit(X, y, sample_weight=w)
    models["xgb"] = xgb

    log("Training CatBoost")
    cat = CatBoostClassifier(
        iterations=420,
        depth=7,
        learning_rate=0.045,
        l2_leaf_reg=5,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    cat.fit(X, y, sample_weight=w)
    models["catboost"] = cat

    return models


def predict_chunked(model, X, name):
    parts = []
    for i in tqdm(range(0, len(X), PRED_CHUNK), desc=f"Predicting {name}"):
        parts.append(model.predict_proba(X[i:i + PRED_CHUNK]).astype(np.float32))
    return np.vstack(parts)


# ================= ENSEMBLES =================
def optimize_cat_xgb(probs, y_all1):
    log("Optimizing CAT_XGB blend")

    y_eval0 = y_all1 - 1

    best_w = None
    best_score = -1
    best_prob = None
    blend_rows = []

    for w_cat in tqdm(np.arange(0.05, 0.96, 0.025), desc="CAT_XGB blend search"):
        w_xgb = 1.0 - w_cat

        p_tmp = (w_cat * probs["catboost"]) + (w_xgb * probs["xgb"])
        pred_tmp = np.argmax(p_tmp, axis=1)

        acc = accuracy_score(y_eval0, pred_tmp)
        f1 = f1_score(y_eval0, pred_tmp, average="macro")
        score = (0.55 * f1) + (0.45 * acc)

        blend_rows.append({
            "w_catboost": float(w_cat),
            "w_xgb": float(w_xgb),
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "selection_score": float(score),
        })

        if score > best_score:
            best_score = score
            best_w = float(w_cat)
            best_prob = p_tmp.astype(np.float32)

    pd.DataFrame(blend_rows).to_csv(OUT / "cat_xgb_blend_search.csv", index=False)

    log(f"Best CAT_XGB blend: {best_w:.3f} CatBoost + {1-best_w:.3f} XGB")
    return best_prob, best_w


def optimize_cat_rf(probs, y_all1):
    log("Optimizing CAT_RF blend")

    y_eval0 = y_all1 - 1

    best_w = None
    best_score = -1
    best_prob = None
    blend_rows = []

    for w_cat in tqdm(np.arange(0.60, 0.96, 0.05), desc="CAT_RF blend search"):
        w_rf = 1.0 - w_cat

        p_tmp = (w_cat * probs["catboost"]) + (w_rf * probs["rf"])
        pred_tmp = np.argmax(p_tmp, axis=1)

        acc = accuracy_score(y_eval0, pred_tmp)
        f1 = f1_score(y_eval0, pred_tmp, average="macro")
        score = (0.55 * f1) + (0.45 * acc)

        blend_rows.append({
            "w_catboost": float(w_cat),
            "w_rf": float(w_rf),
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "selection_score": float(score),
        })

        if score > best_score:
            best_score = score
            best_w = float(w_cat)
            best_prob = p_tmp.astype(np.float32)

    pd.DataFrame(blend_rows).to_csv(OUT / "cat_rf_blend_search.csv", index=False)

    log(f"Best CAT_RF blend: {best_w:.3f} CatBoost + {1-best_w:.3f} RF")
    return best_prob, best_w


# ================= METRICS / OUTPUTS =================
def compute_metrics(y_true1, probs_dict, preds_dict):
    rows = []
    y0 = y_true1 - 1

    for name in preds_dict:
        pred1 = preds_dict[name]
        pred0 = pred1 - 1

        row = {
            "model": name,
            "accuracy": accuracy_score(y0, pred0),
            "f1_macro": f1_score(y0, pred0, average="macro"),
            "f1_weighted": f1_score(y0, pred0, average="weighted"),
            "precision_macro": precision_score(y0, pred0, average="macro", zero_division=0),
            "recall_macro": recall_score(y0, pred0, average="macro", zero_division=0),
            "mean_confidence": float(np.mean(np.max(probs_dict[name], axis=1))),
        }

        try:
            row["auc_ovr_macro"] = roc_auc_score(
                y0,
                probs_dict[name],
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            row["auc_ovr_macro"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    df["overall_selection_score"] = (
        0.40 * df["f1_macro"]
        + 0.35 * df["accuracy"]
        + 0.15 * df["auc_ovr_macro"].fillna(0)
        + 0.10 * df["mean_confidence"]
    )

    df = df.sort_values(
        ["overall_selection_score", "f1_macro", "accuracy"],
        ascending=False,
    )

    return df


def one_vs_rest(y_true1, pred1):
    total = len(y_true1)
    rows = []

    for c in [1, 2, 3, 4, 5]:
        true_c = y_true1 == c
        pred_c = pred1 == c

        tp = int(np.sum(true_c & pred_c))
        tn = int(np.sum(~true_c & ~pred_c))
        fp = int(np.sum(~true_c & pred_c))
        fn = int(np.sum(true_c & ~pred_c))

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


def map_from_valid(pred_valid, final_mask, shape):
    arr = np.zeros(shape, dtype=np.uint8)
    arr[final_mask] = pred_valid.astype(np.uint8)
    return arr


def clean_map(arr):
    out = arr.copy()
    mask = out > 0
    med = median_filter(out, size=MEDIAN_SIZE)
    out[mask] = med[mask]
    return out


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


def save_cm_png(path, y_true1, pred1, title):
    cm = confusion_matrix(y_true1, pred1, labels=[1, 2, 3, 4, 5])
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(["VH", "H", "M", "L", "VL"])
    ax.set_yticklabels(["VH", "H", "M", "L", "VL"])

    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{cmn[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


# ================= MAIN =================
def main():
    fused_path = s3_download(S3_FUSED, "fused.tif")
    labels_path = s3_download(S3_LABELS, "labels.tif")
    ves_score_path = s3_download(S3_VES_SCORE, "ves_score.tif")
    ves_conf_path = s3_download(S3_VES_CONF, "ves_conf.tif")

    log("Reading rasters")
    fused, profile = read_stack(fused_path)
    labels = read_single(labels_path)
    ves_score = read_single(ves_score_path)
    ves_conf = read_single(ves_conf_path)

    mask = (
        np.isfinite(labels)
        & np.isin(labels.astype(np.int32), [1, 2, 3, 4, 5])
        & np.all(np.isfinite(fused), axis=0)
    )

    log(f"Valid pixels: {mask.sum():,}")

    emb, ves_score_n, ves_conf_n = make_galileo_embeddings(
        fused,
        ves_score,
        ves_conf,
        mask,
    )

    Z, pca_scaler, pca = reduce_pca_hybrid(
        emb,
        fused,
        ves_score_n,
        ves_conf_n,
        mask,
    )

    X_train, y_train, w_train, X_all, y_all1, good = prepare_data(
        Z,
        labels,
        ves_conf_n,
        mask,
    )

    final_mask_flat = np.zeros(mask.size, dtype=bool)
    valid_flat_ids = np.where(mask.ravel())[0]
    final_mask_flat[valid_flat_ids[good]] = True
    final_mask = final_mask_flat.reshape(mask.shape)

    models = train_models(X_train, y_train, w_train)

    log("Predicting base models")
    probs = {}
    for name, model in models.items():
        probs[name] = predict_chunked(model, X_all, name)

    log("Creating optimized ensemble probabilities")

    probs["cat_xgb"], best_w_cat_xgb = optimize_cat_xgb(probs, y_all1)
    probs["cat_rf"], best_w_cat_rf = optimize_cat_rf(probs, y_all1)

    preds = {}
    confs = {}

    for name in probs:
        preds[name] = np.argmax(probs[name], axis=1).astype(np.uint8) + 1
        confs[name] = np.max(probs[name], axis=1).astype(np.float32)

    log("Computing metrics")
    summary = compute_metrics(y_all1, probs, preds)
    summary.to_csv(OUT / "model2_galileo_models_summary_metrics.csv", index=False)

    best_metric_model = str(summary.iloc[0]["model"])

    # Advisor-facing recommended model:
    # If CAT_XGB is within 0.5% of best metric score, recommend CAT_XGB because it is ensemble + spatially more stable.
    cat_xgb_score = float(summary.loc[summary["model"] == "cat_xgb", "overall_selection_score"].iloc[0])
    best_score = float(summary["overall_selection_score"].iloc[0])

    if (best_score - cat_xgb_score) <= 0.005:
        recommended_model = "cat_xgb"
    else:
        recommended_model = best_metric_model

    log(f"Best model by pure metrics: {best_metric_model}")
    log(f"Recommended final model: {recommended_model}")

    with open(OUT / "BEST_MODEL_README.txt", "w") as f:
        f.write("Model 2 Galileo-infused ML outputs\n")
        f.write("=================================\n\n")
        f.write(f"Best model by pure metrics: {best_metric_model}\n")
        f.write(f"Recommended final model: {recommended_model}\n\n")
        f.write("Models included: RF, XGB, CatBoost, CAT_XGB, CAT_RF\n\n")
        f.write("CAT_XGB is not trained separately; it is an optimized probability ensemble:\n")
        f.write(f"CAT_XGB = {best_w_cat_xgb:.3f} * CatBoost + {1-best_w_cat_xgb:.3f} * XGBoost\n")
        f.write(f"CAT_RF = {best_w_cat_rf:.3f} * CatBoost + {1-best_w_cat_rf:.3f} * RF\n\n")
        f.write("This run uses hybrid Galileo-style features:\n")
        f.write("- spatial embedding features\n")
        f.write("- original raw thematic raster features\n")
        f.write("- VES score and VES confidence guidance\n\n")
        f.write("Note: CAT_XGB is recommended when it is very close to the best pure metric model because it combines CatBoost stability and XGBoost boundary sharpness.\n")

    metadata = {
        "best_metric_model": best_metric_model,
        "recommended_final_model": recommended_model,
        "models": ["rf", "xgb", "catboost", "cat_xgb", "cat_rf"],
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "max_pca_fit_pixels": MAX_PCA_FIT_PIXELS,
        "pca_components": PCA_COMPONENTS,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "smooth_kernel": SMOOTH_KERNEL,
        "median_size": MEDIAN_SIZE,
        "ves_weight_multiplier": VES_WEIGHT_MULTIPLIER,
        "cat_xgb_weighting": f"{best_w_cat_xgb:.3f} CatBoost + {1-best_w_cat_xgb:.3f} XGBoost",
        "cat_rf_weighting": f"{best_w_cat_rf:.3f} CatBoost + {1-best_w_cat_rf:.3f} RF",
        "feature_mode": "hybrid_galileo_pca_plus_raw_layers_plus_ves",
    }

    with open(OUT / "model2_galileo_run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log("Saving model outputs")
    for name in tqdm(preds.keys(), desc="Saving model outputs"):
        model_dir = OUT / name
        model_dir.mkdir(parents=True, exist_ok=True)

        pred_map = map_from_valid(preds[name], final_mask, labels.shape)
        pred_map_clean = clean_map(pred_map)

        conf_map = np.full(labels.shape, np.nan, dtype=np.float32)
        conf_map[final_mask] = confs[name]

        write_single_tif(
            model_dir / f"model2_galileo_{name}_5class_map.tif",
            pred_map_clean,
            profile,
        )

        write_float_tif(
            model_dir / f"model2_galileo_{name}_confidence.tif",
            conf_map,
            profile,
        )

        save_map_png(
            model_dir / f"model2_galileo_{name}_5class_map.png",
            pred_map_clean,
            f"Model 2 Galileo-Infused ML - {name.upper()}",
        )

        save_cm_png(
            model_dir / f"model2_galileo_{name}_confusion_matrix.png",
            y_all1,
            preds[name],
            f"Model 2 Galileo {name.upper()} Confusion Matrix",
        )

        one_vs_rest(y_all1, preds[name]).to_csv(
            model_dir / f"model2_galileo_{name}_tp_tn_fp_fn_one_vs_rest.csv",
            index=False,
        )

        report = classification_report(
            y_all1,
            preds[name],
            labels=[1, 2, 3, 4, 5],
            target_names=[CLASS_NAMES[i] for i in [1, 2, 3, 4, 5]],
            output_dict=True,
            zero_division=0,
        )

        with open(model_dir / f"model2_galileo_{name}_classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

    log("Uploading to S3")
    upload_folder(OUT, S3_OUT_PREFIX)

    log(f"DONE: s3://{BUCKET}/{S3_OUT_PREFIX}/")


if __name__ == "__main__":
    main()
