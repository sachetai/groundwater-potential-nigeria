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
import rasterio
from rasterio.warp import reproject, Resampling, transform as rio_transform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from scipy.ndimage import uniform_filter, median_filter, binary_opening, binary_closing, binary_fill_holes
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings("ignore")

BUCKET = "sachet-imerg-nigeria"
OUT = "output/final_labels"

AOI_KEY = "datasets_2024/NW_AOI_Original_Boundary.tif"
VES_KEY = "datasets_2024/NW_VES_DATE.csv"

s3 = boto3.client("s3")

RASTER_KEYS = {
    "rainfall": "datasets_2024/NW_Rainfall_2024_total.tif",
    "Drainage_Density": "datasets_2024/Drainage_Density.tif",
    "Lineament_Density": "datasets_2024/Lineament_Density.tif",
    "Lithology": "datasets_2024/Lithology.tif",
    "LST": "datasets_2024/LST.tif",
    "LULC": "datasets_2024/LULC.tif",
    "NDVI": "datasets_2024/NDVI.tif",
    "NDWI": "datasets_2024/NDWI.tif",
    "Slope": "datasets_2024/Slope.tif",
    "Soil": "datasets_2024/Soil.tif",
    "SPI": "datasets_2024/SPI.tif",
    "TWI": "datasets_2024/TWI.tif",
    "TPI": "datasets_2024/TPI.tif",
}

ALL_13 = [
    "rainfall", "Lithology", "Lineament_Density", "TWI", "Slope", "Soil",
    "SPI", "Drainage_Density", "TPI", "NDWI", "NDVI", "LST", "LULC"
]

CONFIGS = {
    "model1_core": [
        "rainfall", "Lithology", "Lineament_Density", "Slope",
        "Soil", "Drainage_Density", "LULC"
    ],
    "model2_extended": [
        "rainfall", "Lithology", "Lineament_Density", "TWI",
        "Slope", "Soil", "NDWI", "Drainage_Density", "NDVI", "LULC"
    ],
    "model3_full": [
        "rainfall", "Lithology", "Lineament_Density", "TWI",
        "Slope", "Soil", "SPI", "Drainage_Density", "TPI",
        "NDWI", "NDVI", "LST", "LULC"
    ],
}

BENEFIT = {"rainfall", "Lineament_Density", "TWI", "NDWI", "NDVI"}
COST = {"Slope", "Drainage_Density", "LST", "SPI", "TPI"}
CATEGORICAL = {"Lithology", "Soil", "LULC"}

CLASS_COLORS = ["#d7191c", "#fdae61", "#ffff66", "#a6d96a", "#1a9641"]
CLASS_LABELS = ["Very High", "High", "Moderate", "Low", "Very Low"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(key, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s3.download_file(BUCKET, key, path)


def upload(path, key):
    s3.upload_file(path, BUCKET, key)


def read_aoi_reference(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = arr.shape
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    positive_mask = np.isfinite(arr) & (arr > 0)

    profile.update(
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        transform=transform,
        crs=crs,
    )

    return arr, positive_mask, profile, transform, crs, shape


def read_align(path, shape, transform, crs, categorical=False):
    with rasterio.open(path) as src:
        src_arr = src.read(1).astype("float32")

        if src.nodata is not None:
            src_arr[src_arr == src.nodata] = np.nan

        dst = np.full(shape, np.nan, dtype="float32")

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )

    return dst.astype("float32")


def build_real_aoi_mask(raw_layers, aoi_positive):
    finite_count = np.zeros(next(iter(raw_layers.values())).shape, dtype=np.int16)
    nonzero_count = np.zeros_like(finite_count)

    for _, arr in raw_layers.items():
        finite = np.isfinite(arr)
        finite_count += finite.astype(np.int16)

        vals = arr[finite]
        if vals.size == 0:
            continue

        zero_frac = np.mean(vals == 0)
        useful = finite & (arr != 0) if zero_frac > 0.05 else finite
        nonzero_count += useful.astype(np.int16)

    support_mask = (finite_count >= 8) & (nonzero_count >= 5)

    if aoi_positive.sum() > 0:
        mask = support_mask & aoi_positive
        method = "AOI positive pixels + raster support"
    else:
        mask = support_mask
        method = "raster-support inferred AOI because AOI raster has no positive pixels"

    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)
    mask = binary_fill_holes(mask)

    if mask.sum() == 0:
        raise RuntimeError("AOI/support mask is empty.")

    return mask.astype(bool), method


def fill_missing(arr, mask):
    out = arr.copy()
    vals = out[mask & np.isfinite(out)]

    if vals.size == 0:
        out[mask] = 0.0
    else:
        out[mask & (~np.isfinite(out))] = np.nanmedian(vals)

    out[~mask] = np.nan
    return out.astype("float32")


def scale01(arr, mask):
    vals = arr[mask & np.isfinite(arr)]
    out = np.full(arr.shape, np.nan, dtype="float32")

    if vals.size < 20:
        out[mask] = 0.5
        return out

    p2, p98 = np.nanpercentile(vals, [2, 98])

    if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
        out[mask] = 0.5
    else:
        out[mask] = np.clip((arr[mask] - p2) / (p98 - p2), 0, 1)

    out[mask & (~np.isfinite(out))] = 0.5
    out[~mask] = np.nan
    return out.astype("float32")


def smooth_nan(arr, mask, size=3):
    v = np.where(mask & np.isfinite(arr), arr, 0).astype("float32")
    w = np.where(mask & np.isfinite(arr), 1, 0).astype("float32")

    sv = uniform_filter(v, size=size, mode="nearest")
    sw = uniform_filter(w, size=size, mode="nearest")

    out = np.divide(sv, sw, out=np.full_like(sv, np.nan), where=sw > 0)
    out[~mask] = np.nan
    return out.astype("float32")


def suitability(name, arr, mask):
    s = scale01(arr, mask)

    if name in BENEFIT:
        out = s
    elif name in COST:
        out = 1.0 - s
    else:
        out = s

    out[mask & (~np.isfinite(out))] = 0.5
    out[~mask] = np.nan
    return out.astype("float32")


def categorical_score(cat, base, mask):
    out = np.full(cat.shape, np.nan, dtype="float32")
    cats = np.unique(cat[mask & np.isfinite(cat)])

    rows = []

    for c in cats:
        cm = mask & (cat == c) & np.isfinite(base)
        if cm.sum() >= 20:
            rows.append((c, float(np.nanmean(base[cm]))))

    if not rows:
        out[mask] = 0.5
        out[~mask] = np.nan
        return out.astype("float32")

    vals = np.array([r[1] for r in rows])
    mn, mx = vals.min(), vals.max()

    for c, v in rows:
        sc = 0.5 if mx <= mn else (v - mn) / (mx - mn)
        sc = 0.5 + 0.35 * (sc - 0.5)
        out[mask & (cat == c)] = sc

    out[mask & (~np.isfinite(out))] = 0.5
    out[~mask] = np.nan
    return out.astype("float32")


def feature_strength(layer, mask):
    vals = layer[mask & np.isfinite(layer)]
    if vals.size < 20:
        return 1e-6
    return max(float(np.nanstd(vals)), 1e-6)


def make_hydro_score(layers, feature_order, mask):
    continuous = {}

    for f in feature_order:
        if f not in CATEGORICAL:
            continuous[f] = suitability(f, layers[f], mask)

    base = np.nanmean(np.stack(list(continuous.values())), axis=0)
    base[mask & (~np.isfinite(base))] = 0.5
    base[~mask] = np.nan

    scores = {}

    for f in feature_order:
        if f in CATEGORICAL:
            scores[f] = categorical_score(layers[f], base, mask)
        else:
            scores[f] = continuous[f]

    raw_weights = {}
    n = len(feature_order)

    for i, f in enumerate(feature_order):
        priority = (n - i) / n
        pattern = feature_strength(scores[f], mask)
        raw_weights[f] = priority * pattern

    total = np.sum(list(raw_weights.values()))
    weights = {f: float(raw_weights[f] / total) for f in feature_order}

    hydro = np.zeros(mask.shape, dtype="float32")

    for f in feature_order:
        layer = scores[f]
        layer[mask & (~np.isfinite(layer))] = 0.5
        hydro[mask] += weights[f] * layer[mask]

    hydro = smooth_nan(hydro, mask, size=3)
    hydro[mask & (~np.isfinite(hydro))] = 0.5
    hydro[~mask] = np.nan

    return hydro.astype("float32"), scores, weights


def robust_series_score(s, higher_better=True):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)

    valid = x[np.isfinite(x)]

    if valid.size < 5:
        out[:] = 0.5
        return out

    p5, p95 = np.nanpercentile(valid, [5, 95])

    if not np.isfinite(p5) or not np.isfinite(p95) or p95 <= p5:
        out[:] = 0.5
        return out

    out[:] = np.clip((x - p5) / (p95 - p5), 0, 1)

    if not higher_better:
        out[:] = 1 - out

    out = out.fillna(out.median())
    return out


def load_ves(ves_csv, profile):
    df = pd.read_csv(ves_csv)

    lon_col = "Longitude"
    lat_col = "Latitude"

    if lon_col not in df.columns or lat_col not in df.columns:
        raise RuntimeError("VES CSV must contain Longitude and Latitude columns.")

    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")

    df = df.dropna(subset=["Longitude", "Latitude"]).copy()

    thickness = robust_series_score(df.get("Aquifer_Thickness_m", pd.Series(np.nan, index=df.index)), True)
    resist = robust_series_score(np.log1p(pd.to_numeric(df.get("Aquifer_Resistivity_ohm_m", pd.Series(np.nan, index=df.index)), errors="coerce")), True)
    overburden = robust_series_score(df.get("Overburden_Thickness_m", pd.Series(np.nan, index=df.index)), False)
    top_depth = robust_series_score(df.get("Aquifer_Top_Depth_m", pd.Series(np.nan, index=df.index)), False)

    df["ves_score"] = (
        0.40 * thickness +
        0.35 * resist +
        0.15 * overburden +
        0.10 * top_depth
    ).clip(0, 1)

    xs, ys = rio_transform("EPSG:4326", profile["crs"], df["Longitude"].tolist(), df["Latitude"].tolist())
    df["x_proj"] = xs
    df["y_proj"] = ys

    rows, cols = rasterio.transform.rowcol(profile["transform"], xs, ys)
    df["row"] = rows
    df["col"] = cols

    h = profile["height"]
    w = profile["width"]

    df = df[(df["row"] >= 0) & (df["row"] < h) & (df["col"] >= 0) & (df["col"] < w)].copy()

    return df


def extract_raster_at_ves(df, layers, mask):
    rows = []

    for _, r in df.iterrows():
        rr = int(r["row"])
        cc = int(r["col"])

        inside = bool(mask[rr, cc]) if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] else False

        row = r.to_dict()
        row["inside_final_aoi_mask"] = inside

        for name, arr in layers.items():
            row[f"raster_{name}"] = float(arr[rr, cc]) if inside and np.isfinite(arr[rr, cc]) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def idw_ves_score(df, mask, profile, max_points=200):
    out = np.full(mask.shape, np.nan, dtype="float32")
    conf = np.zeros(mask.shape, dtype="float32")

    ves = df[df["inside_final_aoi_mask"] == True].copy()

    if len(ves) < 3:
        out[mask] = 0.5
        conf[mask] = 0.0
        return out, conf

    ves = ves.head(max_points)

    p_rows = ves["row"].astype(float).to_numpy()
    p_cols = ves["col"].astype(float).to_numpy()
    p_scores = ves["ves_score"].astype(float).to_numpy()

    rows, cols = np.where(mask)
    n = len(rows)

    block = 150000
    power = 2.0
    eps = 1e-6

    for start in range(0, n, block):
        end = min(start + block, n)

        rr = rows[start:end].astype(float)[:, None]
        cc = cols[start:end].astype(float)[:, None]

        d2 = (rr - p_rows[None, :]) ** 2 + (cc - p_cols[None, :]) ** 2
        d = np.sqrt(d2 + eps)

        w = 1.0 / (d ** power + eps)

        score = np.sum(w * p_scores[None, :], axis=1) / np.sum(w, axis=1)

        nearest = np.min(d, axis=1)
        local_conf = np.exp(-nearest / 75.0)

        out[rows[start:end], cols[start:end]] = score.astype("float32")
        conf[rows[start:end], cols[start:end]] = local_conf.astype("float32")

    out[~mask] = np.nan
    conf[~mask] = 0.0

    return out.astype("float32"), conf.astype("float32")


def calibrate_with_ves(hydro, ves_idw, ves_conf, mask):
    blend = np.clip(0.10 + 0.30 * ves_conf, 0.10, 0.40)
    calibrated = (1 - blend) * hydro + blend * ves_idw

    calibrated = smooth_nan(calibrated, mask, size=3)
    calibrated[mask & (~np.isfinite(calibrated))] = hydro[mask & (~np.isfinite(calibrated))]
    calibrated[~mask] = np.nan

    return calibrated.astype("float32")


def five_class_reclassification(score, mask):
    out = np.zeros(score.shape, dtype="uint8")
    vals = score[mask & np.isfinite(score)]

    if vals.size < 100:
        raise RuntimeError("Not enough valid pixels for five-class reclassification.")

    q10, q30, q70, q90 = np.nanpercentile(vals, [10, 30, 70, 90])

    out[mask & (score >= q90)] = 1
    out[mask & (score >= q70) & (score < q90)] = 2
    out[mask & (score >= q30) & (score < q70)] = 3
    out[mask & (score >= q10) & (score < q30)] = 4
    out[mask & (score < q10)] = 5

    out = median_filter(out, size=3)
    out[~mask] = 0

    return out.astype("uint8"), {
        "very_low_upper_q10": float(q10),
        "low_upper_q30": float(q30),
        "moderate_upper_q70": float(q70),
        "high_upper_q90": float(q90),
    }


def make_gmm_clusters(scores, feature_order, calibrated_score, ves_idw, ves_conf, mask):
    rows, cols = np.where(mask)

    parts = []

    for f in feature_order:
        parts.append(np.nan_to_num(scores[f][rows, cols], nan=0.5))

    parts.append(np.nan_to_num(calibrated_score[rows, cols], nan=0.5))
    parts.append(np.nan_to_num(ves_idw[rows, cols], nan=0.5))
    parts.append(np.nan_to_num(ves_conf[rows, cols], nan=0.0))

    X = np.column_stack(parts).astype("float64")
    X = RobustScaler().fit_transform(X)
    X = np.nan_to_num(X)

    stds = X.std(axis=0)
    X = X[:, stds > 1e-8] if np.any(stds > 1e-8) else np.zeros((X.shape[0], 1))

    labels = None
    method = {}

    attempts = [
        ("diag", 1e-2),
        ("spherical", 1e-2),
        ("diag", 1e-1),
        ("spherical", 1e-1),
    ]

    for cov, reg in attempts:
        try:
            gmm = GaussianMixture(
                n_components=5,
                covariance_type=cov,
                reg_covar=reg,
                random_state=42,
                n_init=5,
                max_iter=300,
                init_params="kmeans",
            )
            labels = gmm.fit_predict(X).astype("uint8") + 1
            method = {
                "method": "GaussianMixture",
                "covariance_type": cov,
                "reg_covar": reg,
                "converged": bool(gmm.converged_),
                "n_iter": int(gmm.n_iter_),
            }
            break
        except Exception as e:
            method = {"last_gmm_error": str(e)}

    if labels is None:
        log("WARNING: GMM failed. Falling back to MiniBatchKMeans.")
        km = MiniBatchKMeans(
            n_clusters=5,
            random_state=42,
            batch_size=8192,
            n_init=20,
            max_iter=500,
        )
        labels = km.fit_predict(X).astype("uint8") + 1
        method = {
            "method": "MiniBatchKMeans fallback",
            "reason": "GMM failed"
        }

    cluster_map = np.zeros(mask.shape, dtype="uint8")
    cluster_map[rows, cols] = labels

    return cluster_map, method


def cluster_to_pseudo_labels(cluster_map, score, mask):
    summary_rows = []

    for c in sorted(np.unique(cluster_map[mask])):
        cm = mask & (cluster_map == c)
        summary_rows.append({
            "cluster": int(c),
            "mean_calibrated_score": float(np.nanmean(score[cm])),
            "pixel_count": int(cm.sum()),
            "pixel_fraction": float(cm.sum() / mask.sum()),
        })

    df = pd.DataFrame(summary_rows).sort_values("mean_calibrated_score", ascending=False).reset_index(drop=True)
    df["rank_order"] = np.arange(1, len(df) + 1)
    df["cluster_based_pseudo_class"] = df["rank_order"]
    df["cluster_based_pseudo_name"] = df["cluster_based_pseudo_class"].map(lambda x: CLASS_LABELS[int(x) - 1])

    label_map = np.zeros(cluster_map.shape, dtype="uint8")

    for _, r in df.iterrows():
        label_map[cluster_map == int(r["cluster"])] = int(r["cluster_based_pseudo_class"])

    label_map[~mask] = 0

    return label_map.astype("uint8"), df


def class_summary(labels, score, mask):
    rows = []

    for c, name in enumerate(CLASS_LABELS, start=1):
        cm = mask & (labels == c)
        rows.append({
            "class": c,
            "label": name,
            "pixel_count": int(cm.sum()),
            "pixel_fraction": float(cm.sum() / mask.sum()) if mask.sum() else 0,
            "mean_calibrated_score": float(np.nanmean(score[cm])) if cm.sum() else np.nan,
        })

    return pd.DataFrame(rows)


def write_tif(path, arr, profile, dtype, nodata=0):
    p = profile.copy()
    p.update(driver="GTiff", count=1, dtype=dtype, compress="deflate", nodata=nodata)

    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)


def plot_map(arr, path, title, cluster=False):
    a = arr.astype(float)
    a[a == 0] = np.nan

    if cluster:
        colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
        labels = [f"Cluster {i}" for i in range(1, 6)]
    else:
        colors = CLASS_COLORS
        labels = CLASS_LABELS

    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    im = ax.imshow(a, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(labels)

    if not cluster:
        cbar.set_label("Groundwater Potential Class")

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_score(score, path, title):
    a = score.copy()
    a[~np.isfinite(a)] = np.nan

    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    im = ax.imshow(a, cmap="viridis", interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Calibrated score")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close()


def write_fused(layers, mask, profile, td):
    local_tif = os.path.join(td, "final_fused_13_layers_aoi_aligned.tif")
    local_txt = os.path.join(td, "final_fused_13_layers_band_names.txt")
    local_csv = os.path.join(td, "final_fused_13_layers_input_summary.csv")

    p = profile.copy()
    p.update(driver="GTiff", count=len(ALL_13), dtype="float32", compress="deflate", nodata=-9999)

    with rasterio.open(local_tif, "w", **p) as dst:
        for i, name in enumerate(ALL_13, 1):
            arr = layers[name].copy()
            arr[~mask] = -9999
            dst.write(arr.astype("float32"), i)
            dst.set_band_description(i, name)

    with open(local_txt, "w") as f:
        for i, name in enumerate(ALL_13, 1):
            f.write(f"Band {i}: {name} | {RASTER_KEYS[name]}\n")

    rows = []
    for name in ALL_13:
        vals = layers[name][mask]
        vals = vals[np.isfinite(vals)]
        rows.append({
            "layer": name,
            "s3_key": RASTER_KEYS[name],
            "valid_pixels": int(vals.size),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
        })

    pd.DataFrame(rows).to_csv(local_csv, index=False)

    upload(local_tif, f"{OUT}/fused_inputs/{os.path.basename(local_tif)}")
    upload(local_txt, f"{OUT}/fused_inputs/{os.path.basename(local_txt)}")
    upload(local_csv, f"{OUT}/fused_inputs/{os.path.basename(local_csv)}")


def process_model(name, feature_order, layers, mask, profile, ves_idw, ves_conf, ves_extract, td):
    log(f"{name}: hydro score")
    hydro, scores, weights = make_hydro_score(layers, feature_order, mask)

    log(f"{name}: VES calibration")
    calibrated = calibrate_with_ves(hydro, ves_idw, ves_conf, mask)

    log(f"{name}: GMM clustering")
    clusters, gmm_info = make_gmm_clusters(scores, feature_order, calibrated, ves_idw, ves_conf, mask)

    log(f"{name}: pseudo labels")
    cluster_labels, cluster_rank_df = cluster_to_pseudo_labels(clusters, calibrated, mask)
    final_labels, thresholds = five_class_reclassification(calibrated, mask)

    local = os.path.join(td, name)
    os.makedirs(local, exist_ok=True)

    paths = {
        "hydro_score_tif": os.path.join(local, f"{name}_pre_ves_hydro_score.tif"),
        "calibrated_score_tif": os.path.join(local, f"{name}_ves_calibrated_score.tif"),
        "ves_idw_score_tif": os.path.join(local, f"{name}_ves_idw_score.tif"),
        "ves_confidence_tif": os.path.join(local, f"{name}_ves_confidence.tif"),
        "clusters_tif": os.path.join(local, f"{name}_gmm_clusters.tif"),
        "cluster_labels_tif": os.path.join(local, f"{name}_cluster_ranked_pseudo_labels.tif"),
        "final_labels_tif": os.path.join(local, f"{name}_final_5class_pseudo_labels.tif"),

        "calibrated_score_png": os.path.join(local, f"{name}_ves_calibrated_score.png"),
        "clusters_png": os.path.join(local, f"{name}_gmm_clusters.png"),
        "cluster_labels_png": os.path.join(local, f"{name}_cluster_ranked_pseudo_labels.png"),
        "final_labels_png": os.path.join(local, f"{name}_final_5class_pseudo_labels.png"),

        "weights_csv": os.path.join(local, f"{name}_generated_score_weights.csv"),
        "cluster_summary_csv": os.path.join(local, f"{name}_gmm_cluster_ranking_summary.csv"),
        "class_summary_csv": os.path.join(local, f"{name}_final_5class_summary.csv"),
        "ves_extract_csv": os.path.join(local, f"{name}_ves_points_with_scores_and_rasters.csv"),
        "metadata_json": os.path.join(local, f"{name}_metadata.json"),
    }

    write_tif(paths["hydro_score_tif"], np.nan_to_num(hydro * 10000, nan=0).astype("int16"), profile, "int16", 0)
    write_tif(paths["calibrated_score_tif"], np.nan_to_num(calibrated * 10000, nan=0).astype("int16"), profile, "int16", 0)
    write_tif(paths["ves_idw_score_tif"], np.nan_to_num(ves_idw * 10000, nan=0).astype("int16"), profile, "int16", 0)
    write_tif(paths["ves_confidence_tif"], np.nan_to_num(ves_conf * 10000, nan=0).astype("int16"), profile, "int16", 0)

    write_tif(paths["clusters_tif"], clusters, profile, "uint8", 0)
    write_tif(paths["cluster_labels_tif"], cluster_labels, profile, "uint8", 0)
    write_tif(paths["final_labels_tif"], final_labels, profile, "uint8", 0)

    plot_score(calibrated, paths["calibrated_score_png"], f"{name} VES-calibrated groundwater score")
    plot_map(clusters, paths["clusters_png"], f"{name} GMM clusters", cluster=True)
    plot_map(cluster_labels, paths["cluster_labels_png"], f"{name} cluster-ranked pseudo labels", cluster=False)
    plot_map(final_labels, paths["final_labels_png"], f"{name} final 5-class pseudo labels", cluster=False)

    pd.DataFrame([
        {
            "feature": f,
            "advisor_priority_rank": i + 1,
            "generated_relative_weight": float(weights[f]),
            "direction": "positive" if f in BENEFIT else "negative" if f in COST else "categorical_data_driven",
            "s3_key": RASTER_KEYS[f],
        }
        for i, f in enumerate(feature_order)
    ]).to_csv(paths["weights_csv"], index=False)

    cluster_rank_df.to_csv(paths["cluster_summary_csv"], index=False)
    class_summary(final_labels, calibrated, mask).to_csv(paths["class_summary_csv"], index=False)

    ves_out = ves_extract.copy()
    ves_out[f"{name}_pre_ves_hydro_score"] = np.nan
    ves_out[f"{name}_ves_calibrated_score"] = np.nan
    ves_out[f"{name}_final_label_class"] = np.nan
    ves_out[f"{name}_final_label_name"] = ""

    for idx, r in ves_out.iterrows():
        if not bool(r["inside_final_aoi_mask"]):
            continue
        rr = int(r["row"])
        cc = int(r["col"])
        ves_out.loc[idx, f"{name}_pre_ves_hydro_score"] = float(hydro[rr, cc])
        ves_out.loc[idx, f"{name}_ves_calibrated_score"] = float(calibrated[rr, cc])
        cls = int(final_labels[rr, cc])
        ves_out.loc[idx, f"{name}_final_label_class"] = cls
        ves_out.loc[idx, f"{name}_final_label_name"] = CLASS_LABELS[cls - 1] if cls > 0 else ""

    ves_out.to_csv(paths["ves_extract_csv"], index=False)

    final_counts = pd.Series(final_labels[mask]).value_counts().sort_index().to_dict()
    cluster_counts = pd.Series(clusters[mask]).value_counts().sort_index().to_dict()

    meta = {
        "model": name,
        "aoi_key": AOI_KEY,
        "ves_key": VES_KEY,
        "features_in_advisor_order": feature_order,
        "method": (
            "AOI-aligned thematic layers -> hydro-score from advisor feature list -> "
            "VES score from resistivity/thickness/depths -> IDW VES calibration -> "
            "GMM clusters + final 5-class reclassification."
        ),
        "important_note": (
            "VES points are used as calibration guidance, not full spatial ground truth. "
            "GMM clusters are exported separately; final pseudo labels use calibrated score "
            "to ensure all 5 reclassification classes are represented."
        ),
        "ves_points_total_after_lonlat_cleaning": int(len(ves_extract)),
        "ves_points_inside_mask": int(ves_extract["inside_final_aoi_mask"].sum()),
        "gmm_info": gmm_info,
        "thresholds": thresholds,
        "final_class_counts": {str(k): int(v) for k, v in final_counts.items()},
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
        "weights": {k: float(v) for k, v in weights.items()},
        "valid_pixels": int(mask.sum()),
    }

    with open(paths["metadata_json"], "w") as f:
        json.dump(meta, f, indent=2)

    for p in paths.values():
        upload(p, f"{OUT}/{name}/{os.path.basename(p)}")

    gc.collect()
    log(f"{name}: done")


def main():
    with tempfile.TemporaryDirectory() as td:
        log("Downloading AOI")
        aoi_local = os.path.join(td, "aoi.tif")
        download(AOI_KEY, aoi_local)

        _, aoi_positive, profile, transform, crs, shape = read_aoi_reference(aoi_local)

        log("Downloading and aligning input rasters")
        raw_layers = {}

        for name in ALL_13:
            local = os.path.join(td, f"{name}.tif")
            download(RASTER_KEYS[name], local)
            raw_layers[name] = read_align(local, shape, transform, crs, categorical=(name in CATEGORICAL))

        mask, mask_method = build_real_aoi_mask(raw_layers, aoi_positive)
        log(f"Mask method: {mask_method}")
        log(f"Valid AOI pixels: {mask.sum()}")

        layers = {}
        for name in ALL_13:
            layers[name] = fill_missing(raw_layers[name], mask)

        log("Downloading and preparing VES data")
        ves_local = os.path.join(td, "NW_VES_DATE.csv")
        download(VES_KEY, ves_local)

        ves_df = load_ves(ves_local, profile)
        ves_extract = extract_raster_at_ves(ves_df, layers, mask)

        ves_clean_csv = os.path.join(td, "final_ves_cleaned_points_with_scores.csv")
        ves_extract.to_csv(ves_clean_csv, index=False)
        upload(ves_clean_csv, f"{OUT}/ves/{os.path.basename(ves_clean_csv)}")

        log(f"VES points loaded: {len(ves_extract)}")
        log(f"VES points inside AOI mask: {int(ves_extract['inside_final_aoi_mask'].sum())}")

        log("Creating VES IDW calibration surface")
        ves_idw, ves_conf = idw_ves_score(ves_extract, mask, profile)

        write_fused(layers, mask, profile, td)

        for model_name, feature_order in CONFIGS.items():
            process_model(
                model_name,
                feature_order,
                layers,
                mask,
                profile,
                ves_idw,
                ves_conf,
                ves_extract,
                td,
            )

    log("All final VES-calibrated outputs complete.")


if __name__ == "__main__":
    main()
