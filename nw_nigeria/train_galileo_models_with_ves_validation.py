#!/usr/bin/env python3

import gc
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from pyproj import Transformer
from rasterio.transform import rowcol
from scipy.ndimage import uniform_filter, gaussian_gradient_magnitude

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm


BUCKET = "sachet-imerg-nigeria"

FUSED_RASTER_S3 = f"s3://{BUCKET}/output/final_labels/fused_inputs/final_fused_13_layers_aoi_aligned.tif"
LABEL_RASTER_S3 = f"s3://{BUCKET}/output/final_labels/model2_extended/model2_extended_final_5class_pseudo_labels.tif"

VES_CSV_S3 = f"s3://{BUCKET}/datasets_2024/NW_VES_DATE.csv"
VES_SCORE_RASTER_S3 = f"s3://{BUCKET}/output/final_labels/model2_extended/model2_extended_ves_calibrated_score.tif"
VES_CONF_RASTER_S3 = f"s3://{BUCKET}/output/final_labels/model2_extended/model2_extended_ves_confidence.tif"

S3_OUT_PREFIX = f"s3://{BUCKET}/output/final_models/galileo_models"

WORK = Path("/tmp/galileo_models_ves_validation")
OUT = WORK / "outputs"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MAX_TRAIN_SAMPLES = 180000
MAX_PCA_FIT_PIXELS = 200000
PCA_COMPONENTS = 28
PRED_CHUNK = 25000

SMOOTH_KERNEL = 3
VES_WEIGHT_MULTIPLIER = 2.5

CLASS_IDS = [1, 2, 3, 4, 5]
CLASS_NAMES = {
    1: "Very High",
    2: "High",
    3: "Moderate",
    4: "Low",
    5: "Very Low",
}
SHORT_LABELS = ["VH", "H", "M", "L", "VL"]
COLORS = ["#e31a1c", "#fdae61", "#ffff66", "#a6d96a", "#1a9850"]


def log(msg):
    print(msg, flush=True)


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def s3_cp(src, dst):
    log(f"Downloading {src}")
    run_cmd(["aws", "s3", "cp", src, str(dst)])


def s3_sync(local_dir, s3_prefix):
    log(f"Uploading outputs to {s3_prefix}")
    run_cmd(["aws", "s3", "sync", str(local_dir), s3_prefix])


def clean_array(a, nodata=None):
    a = a.astype(np.float32)
    if nodata is not None:
        a[a == nodata] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a


def detect_lat_lon_cols(df):
    lower = {c.lower().strip(): c for c in df.columns}
    lat_col = None
    lon_col = None

    for c in ["latitude", "lat", "y"]:
        if c in lower:
            lat_col = lower[c]
            break

    for c in ["longitude", "lon", "long", "x"]:
        if c in lower:
            lon_col = lower[c]
            break

    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not detect latitude/longitude columns. Found: {list(df.columns)}")

    return lat_col, lon_col


def derive_ves_groundtruth(df):
    class_cols = [
        "ves_class", "VES_class", "VES_Class",
        "class", "Class",
        "gw_class", "GW_Class",
        "groundwater_class", "Groundwater_Class",
    ]

    for c in class_cols:
        if c in df.columns:
            raw = df[c]
            if raw.dtype == object:
                mapping = {
                    "very high": 1,
                    "high": 2,
                    "moderate": 3,
                    "medium": 3,
                    "low": 4,
                    "very low": 5,
                }
                return raw.astype(str).str.lower().str.strip().map(mapping).astype("Int64")
            return pd.to_numeric(raw, errors="coerce").round().astype("Int64")

    score_cols = [
        "VES_score", "ves_score", "score", "Score",
        "VES_SCORE", "groundwater_score", "GW_score", "gw_score",
    ]

    score = None
    for c in score_cols:
        if c in df.columns:
            score = pd.to_numeric(df[c], errors="coerce")
            break

    if score is None:
        lat_col, lon_col = detect_lat_lon_cols(df)

        numeric_cols = []
        for c in df.columns:
            if c in [lat_col, lon_col]:
                continue
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() >= max(5, int(0.3 * len(df))):
                numeric_cols.append(c)

        if not numeric_cols:
            raise ValueError("No VES class/score column and no usable numeric columns found.")

        parts = []
        for c in numeric_cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            name = c.lower()

            vmin = vals.quantile(0.02)
            vmax = vals.quantile(0.98)
            vals = vals.clip(vmin, vmax)

            norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

            if any(k in name for k in ["depth", "deep"]):
                norm = 1.0 - norm

            parts.append(norm)

        score = pd.concat(parts, axis=1).mean(axis=1)

    ranks = score.rank(method="average", pct=True)

    out = pd.Series(index=df.index, dtype="float32")
    out[ranks >= 0.80] = 1
    out[(ranks >= 0.60) & (ranks < 0.80)] = 2
    out[(ranks >= 0.40) & (ranks < 0.60)] = 3
    out[(ranks >= 0.20) & (ranks < 0.40)] = 4
    out[ranks < 0.20] = 5

    return out.astype("Int64")


def sample_prediction_at_ves(pred_map, profile, ves_df):
    lat_col, lon_col = detect_lat_lon_cols(ves_df)

    transformer = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)

    xs, ys = transformer.transform(
        ves_df[lon_col].astype(float).values,
        ves_df[lat_col].astype(float).values,
    )

    rows, cols = rowcol(profile["transform"], xs, ys)

    sampled = []
    inside = []
    h, w = pred_map.shape

    for r, c in zip(rows, cols):
        if 0 <= r < h and 0 <= c < w:
            sampled.append(pred_map[r, c])
            inside.append(True)
        else:
            sampled.append(np.nan)
            inside.append(False)

    return np.array(sampled), np.array(inside)


def write_class_tif(path, arr, profile):
    prof = profile.copy()
    prof.update(count=1, dtype="uint8", nodata=0, compress="deflate")

    out = np.where(np.isfinite(arr), arr, 0).astype(np.uint8)

    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def write_float_tif(path, arr, profile):
    prof = profile.copy()
    prof.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")

    out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)

    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def plot_map(arr, title, path):
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    plt.figure(figsize=(10, 8))
    masked = np.ma.masked_where(~np.isfinite(arr) | (arr == 0), arr)
    im = plt.imshow(masked, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=CLASS_IDS)
    cbar.ax.set_yticklabels([CLASS_NAMES[i] for i in CLASS_IDS])
    cbar.set_label("Groundwater Potential Class")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion(cm_norm, title, path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(5), SHORT_LABELS)
    plt.yticks(range(5), SHORT_LABELS)
    plt.colorbar()

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_norm[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def tp_tn_fp_fn_table(y_true, y_pred):
    rows = []

    for cls in CLASS_IDS:
        yt = y_true == cls
        yp = y_pred == cls

        tp = int(np.sum(yt & yp))
        tn = int(np.sum(~yt & ~yp))
        fp = int(np.sum(~yt & yp))
        fn = int(np.sum(yt & ~yp))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        rows.append({
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
        })

    return pd.DataFrame(rows)


def metrics_from_predictions(y_true, y_pred, proba=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if proba is not None:
        out["mean_confidence"] = float(np.max(proba, axis=1).mean())
        try:
            out["auc_ovr_macro"] = roc_auc_score(
                y_true.astype(int) - 1,
                proba,
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            out["auc_ovr_macro"] = np.nan

    return out


def predict_proba_chunked(model, X, name):
    parts = []

    for i in tqdm(range(0, len(X), PRED_CHUNK), desc=f"Predicting {name}"):
        parts.append(model.predict_proba(X[i:i + PRED_CHUNK]).astype(np.float32))

    out = np.vstack(parts)
    del parts
    gc.collect()
    return out


def save_eval_outputs(name, y_true, pred, proba, out_dir, prefix):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = metrics_from_predictions(y_true, pred, proba)
    pd.DataFrame([metrics]).to_csv(out_dir / f"{name}_metrics.csv", index=False)

    report = classification_report(
        y_true,
        pred,
        labels=CLASS_IDS,
        target_names=[CLASS_NAMES[i] for i in CLASS_IDS],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(out_dir / f"{name}_classification_report.csv")

    cm = confusion_matrix(y_true, pred, labels=CLASS_IDS)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    pd.DataFrame(
        cm,
        index=[CLASS_NAMES[i] for i in CLASS_IDS],
        columns=[CLASS_NAMES[i] for i in CLASS_IDS],
    ).to_csv(out_dir / f"{name}_confusion_matrix_5class.csv")

    pd.DataFrame(
        cm_norm,
        index=[CLASS_NAMES[i] for i in CLASS_IDS],
        columns=[CLASS_NAMES[i] for i in CLASS_IDS],
    ).to_csv(out_dir / f"{name}_confusion_matrix_5class_normalized.csv")

    tp_tn_fp_fn_table(y_true, pred).to_csv(
        out_dir / f"{name}_tp_tn_fp_fn_one_vs_rest.csv",
        index=False,
    )

    plot_confusion(
        cm_norm,
        f"{prefix} {name.upper()} Confusion Matrix",
        out_dir / f"{prefix.lower()}_{name}_confusion_matrix.png",
    )

    return metrics


def build_galileo_features(fused, ves_score, ves_conf, valid):
    layers = []

    log("Building Galileo-style features")

    for i in tqdm(range(fused.shape[0]), desc="Raw/local channels"):
        x = fused[i].astype(np.float32)
        layers.append(x)

        filled = x.copy()
        med = np.nanmedian(filled)
        filled[~np.isfinite(filled)] = med

        local_mean = uniform_filter(filled, size=SMOOTH_KERNEL).astype(np.float32)
        layers.append(local_mean)

    for i in tqdm(range(min(6, fused.shape[0])), desc="Gradient channels"):
        x = fused[i].astype(np.float32)
        filled = x.copy()
        med = np.nanmedian(filled)
        filled[~np.isfinite(filled)] = med

        grad = gaussian_gradient_magnitude(filled, sigma=1).astype(np.float32)
        layers.append(grad)

    layers.append(ves_score.astype(np.float32))
    layers.append(ves_conf.astype(np.float32))

    emb_stack = np.stack(layers, axis=0)
    X_raw = emb_stack[:, valid].T.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    n_components = min(PCA_COMPONENTS, X_scaled.shape[1])

    if len(X_scaled) > MAX_PCA_FIT_PIXELS:
        rng = np.random.default_rng(RANDOM_STATE)
        fit_idx = rng.choice(len(X_scaled), size=MAX_PCA_FIT_PIXELS, replace=False)
        X_fit = X_scaled[fit_idx]
    else:
        X_fit = X_scaled

    log(f"PCA input dims={X_scaled.shape[1]}, components={n_components}")

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(X_fit)

    X_pca = pca.transform(X_scaled).astype(np.float32)

    log(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    return X_pca


def build_sample_weights(ves_conf_valid):
    conf = np.nan_to_num(ves_conf_valid, nan=0.0)
    conf = np.clip(conf, 0, 1)
    return (1.0 + (VES_WEIGHT_MULTIPLIER - 1.0) * conf).astype(np.float32)


def train_models(X, y, w):
    models = {}

    log("Training RF")
    rf = RandomForestClassifier(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=4,
        min_samples_split=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X, y, sample_weight=w)
    models["rf"] = rf

    log("Training XGB")
    xgb = XGBClassifier(
        n_estimators=750,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=2.0,
        reg_alpha=0.3,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    xgb.fit(X, y - 1, sample_weight=w)
    models["xgb"] = xgb

    log("Training CatBoost")
    cat = CatBoostClassifier(
        iterations=750,
        depth=7,
        learning_rate=0.04,
        l2_leaf_reg=5,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=100,
        allow_writing_files=False,
    )
    cat.fit(X, y - 1, sample_weight=w)
    models["catboost"] = cat

    return models


def main():
    fused_local = WORK / "fused_inputs.tif"
    label_local = WORK / "labels.tif"
    ves_csv_local = WORK / "NW_VES_DATE.csv"
    ves_score_local = WORK / "ves_score.tif"
    ves_conf_local = WORK / "ves_confidence.tif"

    s3_cp(FUSED_RASTER_S3, fused_local)
    s3_cp(LABEL_RASTER_S3, label_local)
    s3_cp(VES_CSV_S3, ves_csv_local)
    s3_cp(VES_SCORE_RASTER_S3, ves_score_local)
    s3_cp(VES_CONF_RASTER_S3, ves_conf_local)

    log("Reading rasters")

    with rasterio.open(fused_local) as src:
        fused = src.read().astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    with rasterio.open(label_local) as src:
        labels = src.read(1).astype(np.float32)

    with rasterio.open(ves_score_local) as src:
        ves_score = src.read(1).astype(np.float32)

    with rasterio.open(ves_conf_local) as src:
        ves_conf = src.read(1).astype(np.float32)

    for b in range(fused.shape[0]):
        fused[b] = clean_array(fused[b], nodata)

    ves_score = clean_array(ves_score)
    ves_conf = clean_array(ves_conf)

    valid = np.isfinite(labels) & (labels >= 1) & (labels <= 5)
    valid &= np.all(np.isfinite(fused), axis=0)
    valid &= np.isfinite(ves_score)
    valid &= np.isfinite(ves_conf)

    log(f"Valid pixels: {valid.sum():,}")

    y_all = labels[valid].astype(np.uint8)
    X_all = build_galileo_features(fused, ves_score, ves_conf, valid)
    sample_weights_all = build_sample_weights(ves_conf[valid].astype(np.float32))

    if len(y_all) > MAX_TRAIN_SAMPLES:
        idx_pool, _ = train_test_split(
            np.arange(len(y_all)),
            train_size=MAX_TRAIN_SAMPLES,
            stratify=y_all,
            random_state=RANDOM_STATE,
        )
    else:
        idx_pool = np.arange(len(y_all))

    X_pool = X_all[idx_pool]
    y_pool = y_all[idx_pool]
    w_pool = sample_weights_all[idx_pool]

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X_pool,
        y_pool,
        w_pool,
        test_size=0.25,
        stratify=y_pool,
        random_state=RANDOM_STATE,
    )

    log(f"Training samples: {len(y_train):,}")
    log(f"Testing samples: {len(y_test):,}")

    models = train_models(X_train, y_train, w_train)

    log("Evaluating models")

    test_probs = {}
    test_preds = {}

    for name, model in models.items():
        proba = model.predict_proba(X_test).astype(np.float32)
        pred = np.argmax(proba, axis=1) + 1
        test_probs[name] = proba
        test_preds[name] = pred

    test_probs["cat_xgb"] = 0.75 * test_probs["xgb"] + 0.25 * test_probs["catboost"]
    test_preds["cat_xgb"] = np.argmax(test_probs["cat_xgb"], axis=1) + 1

    test_probs["cat_rf"] = 0.65 * test_probs["catboost"] + 0.35 * test_probs["rf"]
    test_preds["cat_rf"] = np.argmax(test_probs["cat_rf"], axis=1) + 1

    summary_rows = []

    for name in ["cat_xgb", "xgb", "catboost", "cat_rf", "rf"]:
        model_dir = OUT / name
        metrics = save_eval_outputs(
            name,
            y_test,
            test_preds[name],
            test_probs[name],
            model_dir,
            "Galileo",
        )
        metrics["model"] = name
        metrics["representation"] = "galileo_style_spatial_embeddings"
        summary_rows.append(metrics)

    pd.DataFrame(summary_rows).sort_values(
        "accuracy",
        ascending=False,
    ).to_csv(OUT / "model2_galileo_models_summary_metrics.csv", index=False)

    log("Predicting full maps")

    full_probs = {}
    for name in ["rf", "xgb", "catboost"]:
        full_probs[name] = predict_proba_chunked(models[name], X_all, name)

    full_probs["cat_xgb"] = 0.75 * full_probs["xgb"] + 0.25 * full_probs["catboost"]
    full_probs["cat_rf"] = 0.65 * full_probs["catboost"] + 0.35 * full_probs["rf"]

    ves_df = pd.read_csv(ves_csv_local)
    ves_df["ves_groundtruth_class"] = derive_ves_groundtruth(ves_df)

    ves_summary_rows = []

    for name in ["cat_xgb", "xgb", "catboost", "cat_rf", "rf"]:
        model_dir = OUT / name
        model_dir.mkdir(parents=True, exist_ok=True)

        pred_valid = np.argmax(full_probs[name], axis=1) + 1
        conf_valid = np.max(full_probs[name], axis=1)

        pred_map = np.zeros(labels.shape, dtype=np.uint8)
        conf_map = np.full(labels.shape, np.nan, dtype=np.float32)

        pred_map[valid] = pred_valid.astype(np.uint8)
        conf_map[valid] = conf_valid.astype(np.float32)

        write_class_tif(model_dir / f"{name}_groundwater_5class_map.tif", pred_map, profile)
        write_float_tif(model_dir / f"{name}_confidence.tif", conf_map, profile)

        plot_map(
            pred_map,
            f"Model 2 Galileo-Infused ML - {name.upper()}",
            model_dir / f"{name}_groundwater_5class_map.png",
        )

        sampled_pred, inside = sample_prediction_at_ves(pred_map, profile, ves_df)

        tmp = ves_df.copy()
        tmp["model"] = name
        tmp["inside_raster"] = inside
        tmp["predicted_class"] = sampled_pred
        tmp["match"] = tmp["ves_groundtruth_class"].astype(float) == tmp["predicted_class"].astype(float)

        tmp_valid = tmp[
            tmp["inside_raster"]
            & tmp["ves_groundtruth_class"].notna()
            & np.isfinite(tmp["predicted_class"])
        ].copy()

        if len(tmp_valid) > 0:
            yv = tmp_valid["ves_groundtruth_class"].astype(int).values
            pv = tmp_valid["predicted_class"].astype(int).values

            ves_acc = accuracy_score(yv, pv)
            ves_bal = balanced_accuracy_score(yv, pv)

            tp_tn_fp_fn_table(yv, pv).to_csv(
                model_dir / f"{name}_ves_tp_tn_fp_fn_one_vs_rest.csv",
                index=False,
            )

            cm = confusion_matrix(yv, pv, labels=CLASS_IDS)
            pd.DataFrame(
                cm,
                index=[CLASS_NAMES[i] for i in CLASS_IDS],
                columns=[CLASS_NAMES[i] for i in CLASS_IDS],
            ).to_csv(model_dir / f"{name}_ves_confusion_matrix_5class.csv")
        else:
            ves_acc = np.nan
            ves_bal = np.nan

        tmp.to_csv(model_dir / f"{name}_ves_point_validation.csv", index=False)

        ves_summary_rows.append({
            "model": name,
            "ves_points_total": len(ves_df),
            "ves_points_inside_raster": int(inside.sum()),
            "ves_points_used": int(len(tmp_valid)),
            "ves_agreement_accuracy": ves_acc,
            "ves_balanced_accuracy": ves_bal,
        })

        del pred_valid, conf_valid, pred_map, conf_map
        gc.collect()

    pd.DataFrame(ves_summary_rows).to_csv(
        OUT / "model2_galileo_ves_validation_summary.csv",
        index=False,
    )

    with open(OUT / "embedding_notes.txt", "w") as f:
        f.write("This script creates lightweight Galileo-style spatial embeddings.\n")
        f.write("It is not the official Galileo foundation model embedding.\n")
        f.write("Inputs: raw fused layers + local spatial means + gradients + VES score/confidence.\n")
        f.write(f"PCA_COMPONENTS={PCA_COMPONENTS}\n")
        f.write(f"SMOOTH_KERNEL={SMOOTH_KERNEL}\n")
        f.write(f"VES_WEIGHT_MULTIPLIER={VES_WEIGHT_MULTIPLIER}\n")
        f.write(f"FUSED_RASTER_S3={FUSED_RASTER_S3}\n")
        f.write(f"LABEL_RASTER_S3={LABEL_RASTER_S3}\n")
        f.write(f"VES_CSV_S3={VES_CSV_S3}\n")

    s3_sync(OUT, S3_OUT_PREFIX)
    log("DONE Galileo models")


if __name__ == "__main__":
    main()
