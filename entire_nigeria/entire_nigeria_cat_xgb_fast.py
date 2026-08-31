#!/usr/bin/env python3
"""
ENTIRE NIGERIA CAT/XGB GROUNDWATER PIPELINE
Version: V8_CATEGORICAL_REMAP_INTERACTIONS

What this version does
----------------------
1. Reads the 10 local Nigeria raster layers.
2. Aligns every layer to one common reduced grid.
3. Saves a correctly aligned 10-band fused GeoTIFF.
4. Applies the exact user-supplied layer ranking:
      1 Rainfall
      2 Lithology
      3 Lineament density
      4 TWI
      5 Slope
      6 Soil
      7 NDWI
      8 Drainage density
      9 NDVI
     10 LULC
5. Converts that ranking to rank-sum weights.
6. Includes lithology, soil and LULC without treating their category codes as
   ordered numeric values.
7. Creates hydrogeologically directed pseudo-labels.
8. Trains compact CatBoost and XGBoost models.
9. Predicts the complete valid Nigeria AOI.
10. Saves raw and calibrated five-class maps, confidence, suitability,
    metrics, models and all ranking/category metadata.
11. Uploads all outputs to S3.

Important scientific limitation
-------------------------------
No independent pixel-level VES, borehole-yield or groundwater target raster is
used in this script. The training labels are pseudo-labels created from ranked
hydrogeological assumptions. Internal metrics measure agreement with that
pseudo-label logic, not independent field validation.

Class encoding
--------------
1 = Very High
2 = High
3 = Moderate
4 = Low
5 = Very Low
0 = NoData
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import boto3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from catboost import CatBoostClassifier
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================

VERSION = "V8_CATEGORICAL_REMAP_INTERACTIONS"

BUCKET = "sachet-imerg-nigeria"
OUTPUT_PREFIX = "output/entire_nigeria_cat_xgb_v7_ranked"

LOCAL_INPUT_DIR = Path(
    "/home/ec2-user/datasets_2024/entire_nigeria/input_layers"
)

WORKDIR = Path("/home/ec2-user/entire_nigeria_cat_xgb_v7_ranked")
WORKDIR.mkdir(parents=True, exist_ok=True)

# Native input grid is approximately 35,508 x 43,893.
# Scale 10 gives approximately 3,551 x 4,390 (~15.6 million cells).
ANALYSIS_SCALE = 10

RANDOM_SEED = 42
MAX_TRAINING_PIXELS = 100_000
CATEGORY_MAPPING_SAMPLE_SIZE = 300_000
PREDICTION_BATCH_SIZE = 200_000
TEST_SIZE = 0.20

CAT_ITERATIONS = 200
XGB_ESTIMATORS = 220

PNG_MAX_DIMENSION = 1900
UPLOAD_OUTPUTS = True

# Use the strict common footprint. Change to 9 only if one source has genuine
# internal NoData gaps that should not remove otherwise valid Nigeria pixels.
MIN_VALID_LAYERS = 10

LAYER_FILES: Dict[str, str] = {
    "drainage": "drainage.tif",
    "lineament": "lineament.tif",
    "lithology": "lithology.tif",
    "LULC": "LULC.tif",
    "NDVI": "NDVI.tif",
    "NDWI": "NDWI.tif",
    "rainfall": "rainfall.tif",
    "slope": "slope.tif",
    "soil_texture": "soil_texture.tif",
    "TWI": "TWI.tif",
}

CATEGORICAL_LAYERS = {"lithology", "soil_texture", "LULC"}
CONTINUOUS_LAYERS = [
    "rainfall",
    "lineament",
    "TWI",
    "slope",
    "NDWI",
    "drainage",
    "NDVI",
]

# Exact ranking supplied by the user.
LAYER_RANKS: Dict[str, int] = {
    "rainfall": 1,
    "lithology": 2,
    "lineament": 3,
    "TWI": 4,
    "slope": 5,
    "soil_texture": 6,
    "NDWI": 7,
    "drainage": 8,
    "NDVI": 9,
    "LULC": 10,
}

# Rank-sum conversion for 10 criteria:
# rank 1 gets 10/55, rank 2 gets 9/55, ..., rank 10 gets 1/55.
RANK_SUM_DENOMINATOR = sum(range(1, len(LAYER_RANKS) + 1))
LAYER_WEIGHTS: Dict[str, float] = {
    layer: (len(LAYER_RANKS) - rank + 1) / RANK_SUM_DENOMINATOR
    for layer, rank in LAYER_RANKS.items()
}

# Continuous-direction assumptions:
# positive = higher values generally increase suitability
# negative = higher values generally decrease suitability
#
# Drainage density is set negative because high drainage density often reflects
# greater runoff and reduced infiltration. Change it to positive only when your
# project-specific hydrogeology supports that interpretation.
CONTINUOUS_DIRECTIONS: Dict[str, str] = {
    "rainfall": "positive",
    "lineament": "positive",
    "TWI": "positive",
    "slope": "negative",
    "NDWI": "positive",
    "drainage": "negative",
    "NDVI": "positive",
}

# Final calibrated class proportions. These limit a dominant Very High class.
# They are explicit assumptions rather than field-derived thresholds.
TARGET_CLASS_PROPORTIONS = {
    1: 0.10,  # Very High
    2: 0.20,  # High
    3: 0.30,  # Moderate
    4: 0.25,  # Low
    5: 0.15,  # Very Low
}

# The final continuous suitability combines the learned CAT/XGB score with the
# independently computed ranked hydrogeological prior. This prevents the model
# from drifting too far from the requested layer ranking.
MODEL_SUITABILITY_WEIGHT = 0.70
RANKED_PRIOR_WEIGHT = 0.30

# Optional expert overrides for categorical class codes.
#
# Leave a dictionary empty to use the automatic hydro-context mapping. Once the
# actual legends for lithology, soil texture and LULC are confirmed, enter
# expert scores from 0.0 (least favorable) to 1.0 (most favorable), for example:
#
# "lithology": {1.0: 0.25, 2.0: 0.80}
#
# Unknown categories retain the automatic inferred value.
MANUAL_CATEGORY_SUITABILITY: Dict[str, Dict[float, float]] = {
    "lithology": {},
    "soil_texture": {},
    "LULC": {},
}

# Nonlinear interaction features supplied to CAT/XGB. These do not alter the
# rank-sum weights; they allow the models to learn that favorable conditions
# often need to occur together.
INTERACTION_FEATURES = {
    "rainfall_x_lithology": ("rainfall", "lithology"),
    "lineament_x_lithology": ("lineament", "lithology"),
    "rainfall_x_TWI": ("rainfall", "TWI"),
    "low_slope_x_soil": ("slope", "soil_texture"),
    "NDWI_x_drainage": ("NDWI", "drainage"),
}

CLASS_NAMES = {
    1: "Very High",
    2: "High",
    3: "Moderate",
    4: "Low",
    5: "Very Low",
}

CLASS_COLORS = [
    "#e41a1c",
    "#fdae61",
    "#ffff66",
    "#a6d96a",
    "#1a9850",
]

FUSED_TIF = WORKDIR / "fused_10_layers_nigeria_scale10.tif"
WEIGHTED_FUSED_TIF = WORKDIR / "rank_weighted_fused_suitability_nigeria.tif"
WEIGHTED_FUSED_PNG = WORKDIR / "rank_weighted_fused_suitability_nigeria.png"
MODEL_SUITABILITY_TIF = WORKDIR / "cat_xgb_model_only_suitability_nigeria.tif"
MODEL_SUITABILITY_PNG = WORKDIR / "cat_xgb_model_only_suitability_nigeria.png"
RAW_CLASS_TIF = WORKDIR / "cat_xgb_raw_5class_nigeria.tif"
FINAL_CLASS_TIF = WORKDIR / "cat_xgb_ranked_calibrated_5class_nigeria.tif"
CONFIDENCE_TIF = WORKDIR / "cat_xgb_confidence_nigeria.tif"
SUITABILITY_TIF = WORKDIR / "cat_xgb_continuous_suitability_nigeria.tif"

RAW_CLASS_PNG = WORKDIR / "cat_xgb_raw_5class_nigeria.png"
FINAL_CLASS_PNG = WORKDIR / "cat_xgb_ranked_calibrated_5class_nigeria.png"
CONFIDENCE_PNG = WORKDIR / "cat_xgb_confidence_nigeria.png"
SUITABILITY_PNG = WORKDIR / "cat_xgb_continuous_suitability_nigeria.png"

METRICS_JSON = WORKDIR / "cat_xgb_metrics.json"
METRICS_CSV = WORKDIR / "cat_xgb_metrics.csv"
REPORT_CSV = WORKDIR / "cat_xgb_classification_report.csv"
CONFUSION_CSV = WORKDIR / "cat_xgb_confusion_matrix.csv"
AREA_RAW_CSV = WORKDIR / "area_by_class_raw.csv"
AREA_FINAL_CSV = WORKDIR / "area_by_class_calibrated.csv"
THRESHOLDS_JSON = WORKDIR / "thresholds_and_calibration.json"
FUSED_BANDS_JSON = WORKDIR / "fused_band_order.json"
RANKING_JSON = WORKDIR / "layer_ranking_and_weights.json"
CATEGORY_LOOKUPS_JSON = WORKDIR / "categorical_suitability_lookups.json"

CAT_MODEL = WORKDIR / "catboost_model.cbm"
XGB_MODEL = WORKDIR / "xgboost_model.json"
README_FILE = WORKDIR / "README.txt"
ERROR_FILE = WORKDIR / "ERROR_TRACEBACK.txt"


# =============================================================================
# INPUT AND ALIGNMENT
# =============================================================================

def verify_local_inputs() -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    missing: List[str] = []

    for layer_name, filename in LAYER_FILES.items():
        path = LOCAL_INPUT_DIR / filename

        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))
        else:
            paths[layer_name] = path

    if missing:
        raise FileNotFoundError(
            "Required local TIFF files are missing:\n"
            + "\n".join(missing)
            + "\n\nThis script intentionally does not use random /vsis3/ reads."
        )

    return paths


def valid_mask(array: np.ndarray, nodata) -> np.ndarray:
    mask = np.isfinite(array)

    if nodata is not None:
        try:
            if np.isfinite(nodata):
                mask &= ~np.isclose(array, nodata)
        except TypeError:
            pass

    return mask


def bounds_close(source_bounds, target_bounds, tolerance: float) -> bool:
    return bool(
        np.allclose(
            np.asarray(source_bounds, dtype=float),
            np.asarray(target_bounds, dtype=float),
            rtol=0,
            atol=tolerance,
        )
    )


def read_layer_to_target(
    path: Path,
    target_crs,
    target_transform,
    target_width: int,
    target_height: int,
    categorical: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    resampling = (
        Resampling.nearest
        if categorical
        else Resampling.bilinear
    )

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path.name} has no CRS.")

        target_bounds = rasterio.transform.array_bounds(
            target_height,
            target_width,
            target_transform,
        )

        tolerance = max(
            abs(float(target_transform.a)),
            abs(float(target_transform.e)),
        ) * 2.0

        fast_read = (
            src.crs == target_crs
            and bounds_close(src.bounds, target_bounds, tolerance)
        )

        if fast_read:
            destination = src.read(
                1,
                out_shape=(target_height, target_width),
                resampling=resampling,
                masked=False,
            ).astype(np.float32)
        else:
            destination = np.full(
                (target_height, target_width),
                np.nan,
                dtype=np.float32,
            )

            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_nodata=np.nan,
                resampling=resampling,
                num_threads=2,
                init_dest_nodata=True,
            )

        mask = valid_mask(destination, src.nodata)
        destination[~mask] = np.nan

        return destination, mask


def robust_stats(
    array: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    values = array[mask]

    if values.size == 0:
        raise ValueError("Layer contains no valid cells on the target grid.")

    if values.size > 250_000:
        selected = rng.choice(
            values.size,
            size=250_000,
            replace=False,
        )
        values = values[selected]

    low, median, high = np.quantile(
        values,
        [0.02, 0.50, 0.98],
    )

    if not all(np.isfinite([low, median, high])):
        raise ValueError("Non-finite robust statistics encountered.")

    if high <= low:
        high = low + 1.0

    return float(low), float(median), float(high)


def save_fused_raster(
    layer_arrays: Dict[str, np.ndarray],
    common_mask: np.ndarray,
    output_profile: dict,
) -> None:
    profile = output_profile.copy()
    profile.update(
        driver="GTiff",
        count=len(LAYER_FILES),
        dtype="float32",
        nodata=-9999.0,
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES",
    )

    with rasterio.open(FUSED_TIF, "w", **profile) as dst:
        for band_index, layer_name in enumerate(LAYER_FILES, start=1):
            array = layer_arrays[layer_name].copy()
            array[~common_mask] = -9999.0
            array[~np.isfinite(array)] = -9999.0

            dst.write(array.astype(np.float32), band_index)
            dst.set_band_description(band_index, layer_name)

    FUSED_BANDS_JSON.write_text(
        json.dumps(
            {
                "version": VERSION,
                "band_order": {
                    str(index): layer_name
                    for index, layer_name in enumerate(LAYER_FILES, start=1)
                },
                "nodata": -9999.0,
                "analysis_scale": ANALYSIS_SCALE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# RANKED SUITABILITY LOGIC
# =============================================================================

def robust_scale(
    values: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    denominator = max(high - low, 1e-12)

    return np.clip(
        (values - low) / denominator,
        0.0,
        1.0,
    )


def continuous_component(
    values: np.ndarray,
    layer_name: str,
    stats: Mapping[str, Tuple[float, float, float]],
) -> np.ndarray:
    low, _, high = stats[layer_name]

    component = robust_scale(
        np.asarray(values, dtype=np.float64),
        low,
        high,
    )

    if CONTINUOUS_DIRECTIONS[layer_name] == "negative":
        component = 1.0 - component

    return component.astype(np.float32)


def build_preliminary_continuous_score(
    sample_df: pd.DataFrame,
    stats: Mapping[str, Tuple[float, float, float]],
) -> np.ndarray:
    """
    Build a preliminary score using continuous layers only.

    It is used solely to infer a suitability value for each lithology, soil and
    LULC category without assuming that larger category codes are better.
    """
    score = np.zeros(len(sample_df), dtype=np.float64)

    continuous_weight_total = sum(
        LAYER_WEIGHTS[layer_name]
        for layer_name in CONTINUOUS_LAYERS
    )

    for layer_name in CONTINUOUS_LAYERS:
        component = continuous_component(
            sample_df[layer_name].to_numpy(),
            layer_name,
            stats,
        )

        score += LAYER_WEIGHTS[layer_name] * component

    score /= max(continuous_weight_total, 1e-12)

    return score.astype(np.float32)


def infer_category_lookup(
    category_values: np.ndarray,
    reference_score: np.ndarray,
) -> Dict[str, float]:
    """
    Infer category suitability from the median continuous hydro-score.

    Category codes are treated only as labels. They are never interpreted as
    ordinal numbers. Small categories are shrunk toward the global median.
    """
    category_values = np.asarray(category_values)
    reference_score = np.asarray(reference_score, dtype=np.float64)

    finite = np.isfinite(category_values) & np.isfinite(reference_score)

    if not finite.any():
        return {}

    category_values = category_values[finite]
    reference_score = reference_score[finite]

    global_median = float(np.median(reference_score))
    rows = []

    for category in np.unique(category_values):
        category_mask = category_values == category
        count = int(category_mask.sum())
        median_score = float(np.median(reference_score[category_mask]))

        # Empirical-Bayes-style shrinkage for rare classes.
        shrinkage_strength = 200.0
        shrunk_score = (
            count * median_score
            + shrinkage_strength * global_median
        ) / (count + shrinkage_strength)

        rows.append((category, count, shrunk_score))

    shrunk_values = np.asarray(
        [row[2] for row in rows],
        dtype=np.float64,
    )

    if np.ptp(shrunk_values) <= 1e-12:
        normalized = np.full(
            shrunk_values.shape,
            0.5,
            dtype=np.float64,
        )
    else:
        lower, upper = np.quantile(
            shrunk_values,
            [0.02, 0.98],
        )

        if upper <= lower:
            lower = float(shrunk_values.min())
            upper = float(shrunk_values.max())

        normalized = np.clip(
            (shrunk_values - lower) / max(upper - lower, 1e-12),
            0.0,
            1.0,
        )

    lookup: Dict[str, float] = {}

    for (category, _count, _score), suitability in zip(rows, normalized):
        lookup[str(float(category))] = float(suitability)

    return lookup


def apply_category_lookup(
    values: np.ndarray,
    lookup: Mapping[str, float],
    default_value: float = 0.5,
) -> np.ndarray:
    values = np.asarray(values)
    result = np.full(
        values.shape,
        default_value,
        dtype=np.float32,
    )

    for category_text, suitability in lookup.items():
        category = float(category_text)
        result[np.isclose(values, category)] = float(suitability)

    result[~np.isfinite(values)] = default_value

    return result


def build_category_lookups(
    sample_df: pd.DataFrame,
    stats: Mapping[str, Tuple[float, float, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Build non-ordinal suitability mappings for categorical layers.

    The automatic value for a category is inferred from the median continuous
    hydrogeological context in which that category occurs. Expert overrides,
    when supplied, replace only the specified category codes.
    """
    preliminary_score = build_preliminary_continuous_score(
        sample_df,
        stats,
    )

    lookups: Dict[str, Dict[str, float]] = {}

    for layer_name in ["lithology", "soil_texture", "LULC"]:
        automatic_lookup = infer_category_lookup(
            sample_df[layer_name].to_numpy(),
            preliminary_score,
        )

        for category_code, suitability in MANUAL_CATEGORY_SUITABILITY[
            layer_name
        ].items():
            if not 0.0 <= float(suitability) <= 1.0:
                raise ValueError(
                    f"Manual suitability for {layer_name} category "
                    f"{category_code} must be between 0 and 1."
                )
            automatic_lookup[str(float(category_code))] = float(suitability)

        lookups[layer_name] = automatic_lookup

    return lookups


def transform_features(
    features: pd.DataFrame,
    stats: Mapping[str, Tuple[float, float, float]],
    category_lookups: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """
    Transform raw predictors into physically meaningful 0–1 components.

    Continuous layers are robustly normalized and direction-adjusted.
    Lithology, soil and LULC category IDs are mapped to suitability values;
    the numeric category IDs themselves are never treated as magnitudes.
    Interaction features are then added for CAT/XGB.
    """
    transformed: Dict[str, np.ndarray] = {}

    for layer_name in LAYER_FILES:
        if layer_name in CATEGORICAL_LAYERS:
            transformed[layer_name] = apply_category_lookup(
                features[layer_name].to_numpy(),
                category_lookups[layer_name],
            )
        else:
            transformed[layer_name] = continuous_component(
                features[layer_name].to_numpy(),
                layer_name,
                stats,
            )

    # Because slope has already been direction-adjusted, the transformed
    # "slope" column means low-slope suitability, not raw steepness.
    for interaction_name, (first, second) in INTERACTION_FEATURES.items():
        transformed[interaction_name] = (
            transformed[first] * transformed[second]
        ).astype(np.float32)

    ordered_columns = list(LAYER_FILES.keys()) + list(
        INTERACTION_FEATURES.keys()
    )

    return pd.DataFrame(
        transformed,
        columns=ordered_columns,
    )


def weighted_ranked_score(
    transformed_features: pd.DataFrame,
) -> np.ndarray:
    score = np.zeros(
        len(transformed_features),
        dtype=np.float64,
    )

    for layer_name in LAYER_FILES:
        score += (
            LAYER_WEIGHTS[layer_name]
            * transformed_features[layer_name].to_numpy(dtype=np.float64)
        )

    return np.clip(score, 0.0, 1.0).astype(np.float32)


def pseudo_label_thresholds(
    score: np.ndarray,
) -> List[float]:
    # Ascending score:
    # 15% Very Low, 25% Low, 30% Moderate, 20% High, 10% Very High.
    return [
        float(value)
        for value in np.quantile(
            score,
            [0.15, 0.40, 0.70, 0.90],
        )
    ]


def classify_score(
    score: np.ndarray,
    thresholds: List[float],
) -> np.ndarray:
    q15, q40, q70, q90 = thresholds

    classes = np.full(
        score.shape,
        3,
        dtype=np.uint8,
    )

    classes[score > q90] = 1
    classes[(score > q70) & (score <= q90)] = 2
    classes[(score > q15) & (score <= q40)] = 4
    classes[score <= q15] = 5

    return classes


def ensemble_probabilities(
    cat_model: CatBoostClassifier,
    xgb_model: XGBClassifier,
    x: np.ndarray,
) -> np.ndarray:
    return (
        cat_model.predict_proba(x)
        + xgb_model.predict_proba(x)
    ) / 2.0


def probabilities_to_suitability(probabilities: np.ndarray) -> np.ndarray:
    # Model class order:
    # 0 Very High, 1 High, 2 Moderate, 3 Low, 4 Very Low.
    class_values = np.asarray(
        [1.00, 0.75, 0.50, 0.25, 0.00],
        dtype=np.float32,
    )

    return probabilities @ class_values


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def write_single_band(
    path: Path,
    array: np.ndarray,
    output_profile: dict,
    dtype: str,
    nodata,
    description: str,
    predictor: int,
) -> None:
    profile = output_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="DEFLATE",
        predictor=predictor,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        dst.set_band_description(1, description)


def save_class_png(
    tif_path: Path,
    png_path: Path,
    title: str,
) -> None:
    with rasterio.open(tif_path) as src:
        scale = min(
            1.0,
            PNG_MAX_DIMENSION / max(src.width, src.height),
        )

        width = max(1, int(src.width * scale))
        height = max(1, int(src.height * scale))

        array = src.read(
            1,
            out_shape=(1, height, width),
            resampling=Resampling.nearest,
        )

    masked = np.ma.masked_where(array == 0, array)

    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_bad("white")

    norm = BoundaryNorm(
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        cmap.N,
    )

    fig, ax = plt.subplots(figsize=(13, 10))

    image = ax.imshow(
        masked,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=18)
    ax.axis("off")

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.045,
        pad=0.035,
        ticks=[1, 2, 3, 4, 5],
    )

    colorbar.ax.set_yticklabels(
        [CLASS_NAMES[i] for i in range(1, 6)]
    )
    colorbar.set_label("Groundwater Potential Class")

    fig.tight_layout()
    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_continuous_png(
    tif_path: Path,
    png_path: Path,
    title: str,
    label: str,
) -> None:
    with rasterio.open(tif_path) as src:
        scale = min(
            1.0,
            PNG_MAX_DIMENSION / max(src.width, src.height),
        )

        width = max(1, int(src.width * scale))
        height = max(1, int(src.height * scale))

        array = src.read(
            1,
            out_shape=(1, height, width),
            resampling=Resampling.bilinear,
        )

    masked = np.ma.masked_where(array < 0, array)

    fig, ax = plt.subplots(figsize=(13, 10))

    image = ax.imshow(
        masked,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=18)
    ax.axis("off")

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.045,
        pad=0.035,
    )
    colorbar.set_label(label)

    fig.tight_layout()
    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def class_area_table(
    class_array: np.ndarray,
    common_mask: np.ndarray,
) -> pd.DataFrame:
    values = class_array[common_mask]
    rows = []

    for class_id in range(1, 6):
        count = int(np.sum(values == class_id))

        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "pixel_count": count,
                "percentage": 100.0 * count / max(values.size, 1),
            }
        )

    return pd.DataFrame(rows)


def upload_outputs(s3_client) -> None:
    output_files = [
        FUSED_TIF,
        WEIGHTED_FUSED_TIF,
        WEIGHTED_FUSED_PNG,
        MODEL_SUITABILITY_TIF,
        MODEL_SUITABILITY_PNG,
        FUSED_BANDS_JSON,
        RANKING_JSON,
        CATEGORY_LOOKUPS_JSON,
        RAW_CLASS_TIF,
        FINAL_CLASS_TIF,
        CONFIDENCE_TIF,
        SUITABILITY_TIF,
        RAW_CLASS_PNG,
        FINAL_CLASS_PNG,
        CONFIDENCE_PNG,
        SUITABILITY_PNG,
        METRICS_JSON,
        METRICS_CSV,
        REPORT_CSV,
        CONFUSION_CSV,
        AREA_RAW_CSV,
        AREA_FINAL_CSV,
        THRESHOLDS_JSON,
        CAT_MODEL,
        XGB_MODEL,
        README_FILE,
    ]

    for local_path in output_files:
        key = f"{OUTPUT_PREFIX}/{local_path.name}"

        print(
            f"Uploading {local_path.name} -> s3://{BUCKET}/{key}",
            flush=True,
        )

        s3_client.upload_file(
            str(local_path),
            BUCKET,
            key,
        )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    s3_client = boto3.client("s3")

    print("=" * 82, flush=True)
    print(
        f"ENTIRE NIGERIA CAT/XGB PIPELINE — {VERSION}",
        flush=True,
    )
    print("=" * 82, flush=True)
    print(f"Analysis scale: {ANALYSIS_SCALE}", flush=True)
    print("Raster source: LOCAL FILES ONLY", flush=True)
    print("Fused raster output: ENABLED", flush=True)
    print("Ranking source: USER-SUPPLIED MODEL_2", flush=True)
    print(
        "Categorical treatment: NON-ORDINAL REMAPPING + OPTIONAL OVERRIDES",
        flush=True,
    )
    print(
        f"Final suitability blend: {MODEL_SUITABILITY_WEIGHT:.0%} model + "
        f"{RANKED_PRIOR_WEIGHT:.0%} ranked prior",
        flush=True,
    )

    print("\nLayer ranking and rank-sum weights:", flush=True)

    for layer_name, rank in sorted(
        LAYER_RANKS.items(),
        key=lambda item: item[1],
    ):
        print(
            f"  {rank:2d}. {layer_name:14s} "
            f"weight={LAYER_WEIGHTS[layer_name]:.6f}",
            flush=True,
        )

    RANKING_JSON.write_text(
        json.dumps(
            {
                "version": VERSION,
                "ranking": LAYER_RANKS,
                "rank_sum_weights": LAYER_WEIGHTS,
                "continuous_directions": CONTINUOUS_DIRECTIONS,
                "categorical_layers": sorted(CATEGORICAL_LAYERS),
                "manual_category_overrides": MANUAL_CATEGORY_SUITABILITY,
                "interaction_features": INTERACTION_FEATURES,
                "model_suitability_weight": MODEL_SUITABILITY_WEIGHT,
                "ranked_prior_weight": RANKED_PRIOR_WEIGHT,
                "weight_sum": float(sum(LAYER_WEIGHTS.values())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nStep 1/9: Verify local TIFF inputs", flush=True)

    local_paths = verify_local_inputs()

    for layer_name, path in local_paths.items():
        print(
            f"  OK: {layer_name:14s} "
            f"{path} ({path.stat().st_size / (1024**3):.2f} GiB)",
            flush=True,
        )

    free_gib = shutil.disk_usage(WORKDIR).free / (1024**3)
    print(f"\nFree disk space: {free_gib:.1f} GiB", flush=True)

    print("\nStep 2/9: Build common reduced grid", flush=True)

    with rasterio.open(local_paths["NDVI"]) as reference:
        if reference.crs is None:
            raise ValueError("NDVI reference raster has no CRS.")

        target_width = math.ceil(
            reference.width / ANALYSIS_SCALE
        )
        target_height = math.ceil(
            reference.height / ANALYSIS_SCALE
        )

        target_transform = reference.transform * Affine.scale(
            reference.width / target_width,
            reference.height / target_height,
        )

        target_crs = reference.crs
        native_height = reference.height
        native_width = reference.width

        output_profile = reference.profile.copy()
        output_profile.update(
            width=target_width,
            height=target_height,
            transform=target_transform,
            crs=target_crs,
        )

    print(
        f"  Native shape:   {native_height:,} x {native_width:,}",
        flush=True,
    )
    print(
        f"  Analysis shape: {target_height:,} x {target_width:,}",
        flush=True,
    )

    print("\nStep 3/9: Align all 10 layers", flush=True)

    layer_arrays: Dict[str, np.ndarray] = {}
    layer_masks: Dict[str, np.ndarray] = {}
    stats: Dict[str, Tuple[float, float, float]] = {}

    for layer_name, path in tqdm(
        list(local_paths.items()),
        desc="Align layers",
        unit="layer",
        mininterval=1.0,
    ):
        array, mask = read_layer_to_target(
            path=path,
            target_crs=target_crs,
            target_transform=target_transform,
            target_width=target_width,
            target_height=target_height,
            categorical=layer_name in CATEGORICAL_LAYERS,
        )

        low, median, high = robust_stats(
            array,
            mask,
            rng,
        )

        layer_arrays[layer_name] = array
        layer_masks[layer_name] = mask
        stats[layer_name] = (low, median, high)

        print(
            f"\n  {layer_name:14s}: "
            f"valid={int(mask.sum()):,}, "
            f"p02={low:.6g}, median={median:.6g}, p98={high:.6g}",
            flush=True,
        )

    valid_count = np.sum(
        np.stack(list(layer_masks.values()), axis=0),
        axis=0,
    )

    common_mask = valid_count >= MIN_VALID_LAYERS
    common_flat = np.flatnonzero(common_mask.ravel())

    if common_flat.size == 0:
        raise RuntimeError(
            "No common valid AOI pixels found. "
            "Set MIN_VALID_LAYERS = 9 only if one source has legitimate gaps."
        )

    print(
        f"\n  Common valid AOI pixels: {common_flat.size:,}",
        flush=True,
    )

    print("\nStep 4/9: Save correctly fused 10-band dataset", flush=True)

    save_fused_raster(
        layer_arrays=layer_arrays,
        common_mask=common_mask,
        output_profile=output_profile,
    )

    print(f"  Saved locally: {FUSED_TIF}", flush=True)

    print(
        "\nStep 5/9: Infer categorical suitability for "
        "lithology, soil and LULC",
        flush=True,
    )

    mapping_count = min(
        CATEGORY_MAPPING_SAMPLE_SIZE,
        common_flat.size,
    )

    mapping_indices = rng.choice(
        common_flat,
        size=mapping_count,
        replace=False,
    )

    raw_mapping_matrix = np.empty(
        (mapping_count, len(LAYER_FILES)),
        dtype=np.float32,
    )

    feature_names = list(LAYER_FILES.keys())

    for column_index, layer_name in enumerate(feature_names):
        values = layer_arrays[layer_name].ravel()[mapping_indices]
        _, median, _ = stats[layer_name]

        invalid = ~np.isfinite(values)

        if invalid.any():
            values = values.copy()
            values[invalid] = median

        raw_mapping_matrix[:, column_index] = values

    mapping_df = pd.DataFrame(
        raw_mapping_matrix,
        columns=feature_names,
    )

    category_lookups = build_category_lookups(
        mapping_df,
        stats,
    )

    CATEGORY_LOOKUPS_JSON.write_text(
        json.dumps(
            {
                "version": VERSION,
                "method": (
                    "Category suitability = shrunk median of the preliminary "
                    "continuous hydro-score; category codes are labels only."
                ),
                "lookups": category_lookups,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for layer_name, lookup in category_lookups.items():
        print(
            f"  {layer_name}: {len(lookup)} categories mapped",
            flush=True,
        )

    del raw_mapping_matrix
    del mapping_df
    gc.collect()

    print("\nStep 6/9: Build ranked training sample and pseudo-labels", flush=True)

    training_count = min(
        MAX_TRAINING_PIXELS,
        common_flat.size,
    )

    training_indices = rng.choice(
        common_flat,
        size=training_count,
        replace=False,
    )

    raw_training_matrix = np.empty(
        (training_count, len(feature_names)),
        dtype=np.float32,
    )

    for column_index, layer_name in enumerate(feature_names):
        values = layer_arrays[layer_name].ravel()[training_indices]
        _, median, _ = stats[layer_name]

        invalid = ~np.isfinite(values)

        if invalid.any():
            values = values.copy()
            values[invalid] = median

        raw_training_matrix[:, column_index] = values

    raw_training_df = pd.DataFrame(
        raw_training_matrix,
        columns=feature_names,
    )

    training_df = transform_features(
        raw_training_df,
        stats,
        category_lookups,
    )

    ranked_score = weighted_ranked_score(training_df)
    training_thresholds = pseudo_label_thresholds(ranked_score)
    labels = classify_score(
        ranked_score,
        training_thresholds,
    )

    print(
        pd.Series(labels).value_counts().sort_index(),
        flush=True,
    )

    print("\nStep 7/9: Train and evaluate CAT/XGB ensemble", flush=True)

    y_zero = labels.astype(np.int32) - 1

    x_train, x_test, y_train, y_test = train_test_split(
        training_df,
        y_zero,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_zero,
    )

    cat_model = CatBoostClassifier(
        iterations=CAT_ITERATIONS,
        depth=6,
        learning_rate=0.08,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=RANDOM_SEED,
        l2_leaf_reg=5.0,
        thread_count=-1,
        verbose=25,
        allow_writing_files=False,
    )

    cat_model.fit(
        x_train,
        y_train,
        eval_set=(x_test, y_test),
        use_best_model=True,
        early_stopping_rounds=30,
    )

    cat_model.save_model(CAT_MODEL)

    xgb_model = XGBClassifier(
        n_estimators=XGB_ESTIMATORS,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.05,
        reg_lambda=1.3,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        max_bin=128,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    xgb_model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        verbose=25,
    )

    xgb_model.save_model(XGB_MODEL)

    test_probabilities = ensemble_probabilities(
        cat_model,
        xgb_model,
        x_test.to_numpy(dtype=np.float32),
    )

    test_prediction = np.argmax(
        test_probabilities,
        axis=1,
    ).astype(np.int32)

    true_classes = np.asarray(
        y_test,
        dtype=np.int32,
    ) + 1

    predicted_classes = test_prediction + 1

    metrics = {
        "version": VERSION,
        "analysis_scale": ANALYSIS_SCALE,
        "accuracy": float(
            accuracy_score(
                true_classes,
                predicted_classes,
            )
        ),
        "f1_macro": float(
            f1_score(
                true_classes,
                predicted_classes,
                average="macro",
            )
        ),
        "precision_macro": float(
            precision_score(
                true_classes,
                predicted_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                true_classes,
                predicted_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "kappa": float(
            cohen_kappa_score(
                true_classes,
                predicted_classes,
            )
        ),
        "training_pixels": int(len(x_train)),
        "testing_pixels": int(len(x_test)),
        "common_aoi_pixels": int(common_flat.size),
        "label_type": "rank-weighted hydrogeological pseudo-labels",
        "lithology_included": True,
        "independent_field_validation": False,
    }

    METRICS_JSON.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([metrics]).to_csv(
        METRICS_CSV,
        index=False,
    )

    report = classification_report(
        true_classes,
        predicted_classes,
        labels=[1, 2, 3, 4, 5],
        target_names=[
            CLASS_NAMES[i]
            for i in range(1, 6)
        ],
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(REPORT_CSV)

    cm = confusion_matrix(
        true_classes,
        predicted_classes,
        labels=[1, 2, 3, 4, 5],
    )

    pd.DataFrame(
        cm,
        index=[
            f"true_{CLASS_NAMES[i]}"
            for i in range(1, 6)
        ],
        columns=[
            f"pred_{CLASS_NAMES[i]}"
            for i in range(1, 6)
        ],
    ).to_csv(CONFUSION_CSV)

    print(json.dumps(metrics, indent=2), flush=True)

    print("\nStep 8/9: Predict full AOI and calibrate final classes", flush=True)

    raw_class_flat = np.zeros(
        target_height * target_width,
        dtype=np.uint8,
    )

    confidence_flat = np.full(
        target_height * target_width,
        -9999.0,
        dtype=np.float32,
    )

    suitability_flat = np.full(
        target_height * target_width,
        -9999.0,
        dtype=np.float32,
    )

    model_suitability_flat = np.full(
        target_height * target_width,
        -9999.0,
        dtype=np.float32,
    )

    ranked_prior_flat = np.full(
        target_height * target_width,
        -9999.0,
        dtype=np.float32,
    )

    for start in tqdm(
        range(
            0,
            common_flat.size,
            PREDICTION_BATCH_SIZE,
        ),
        desc="Predict batches",
        unit="batch",
        mininterval=1.0,
    ):
        indices = common_flat[
            start : start + PREDICTION_BATCH_SIZE
        ]

        raw_batch_matrix = np.empty(
            (indices.size, len(feature_names)),
            dtype=np.float32,
        )

        for column_index, layer_name in enumerate(feature_names):
            values = layer_arrays[layer_name].ravel()[indices]
            _, median, _ = stats[layer_name]

            invalid = ~np.isfinite(values)

            if invalid.any():
                values = values.copy()
                values[invalid] = median

            raw_batch_matrix[:, column_index] = values

        raw_batch_df = pd.DataFrame(
            raw_batch_matrix,
            columns=feature_names,
        )

        transformed_batch_df = transform_features(
            raw_batch_df,
            stats,
            category_lookups,
        )

        probabilities = ensemble_probabilities(
            cat_model,
            xgb_model,
            transformed_batch_df.to_numpy(dtype=np.float32),
        )

        raw_classes = (
            np.argmax(
                probabilities,
                axis=1,
            ).astype(np.uint8)
            + 1
        )

        confidence = np.max(
            probabilities,
            axis=1,
        ).astype(np.float32)

        model_suitability = probabilities_to_suitability(
            probabilities,
        ).astype(np.float32)

        ranked_prior = weighted_ranked_score(
            transformed_batch_df,
        ).astype(np.float32)

        suitability = np.clip(
            MODEL_SUITABILITY_WEIGHT * model_suitability
            + RANKED_PRIOR_WEIGHT * ranked_prior,
            0.0,
            1.0,
        ).astype(np.float32)

        raw_class_flat[indices] = raw_classes
        confidence_flat[indices] = confidence
        model_suitability_flat[indices] = model_suitability
        ranked_prior_flat[indices] = ranked_prior
        suitability_flat[indices] = suitability

        del raw_batch_matrix
        del raw_batch_df
        del transformed_batch_df
        del probabilities
        del raw_classes
        del confidence
        del model_suitability
        del ranked_prior
        del suitability

    valid_suitability = suitability_flat[common_flat]

    final_thresholds = [
        float(value)
        for value in np.quantile(
            valid_suitability,
            [0.15, 0.40, 0.70, 0.90],
        )
    ]

    final_class_flat = np.zeros_like(raw_class_flat)

    final_class_flat[common_flat] = classify_score(
        valid_suitability,
        final_thresholds,
    )

    raw_class_array = raw_class_flat.reshape(
        target_height,
        target_width,
    )

    final_class_array = final_class_flat.reshape(
        target_height,
        target_width,
    )

    confidence_array = confidence_flat.reshape(
        target_height,
        target_width,
    )

    suitability_array = suitability_flat.reshape(
        target_height,
        target_width,
    )

    model_suitability_array = model_suitability_flat.reshape(
        target_height,
        target_width,
    )

    ranked_prior_array = ranked_prior_flat.reshape(
        target_height,
        target_width,
    )

    write_single_band(
        RAW_CLASS_TIF,
        raw_class_array,
        output_profile,
        dtype="uint8",
        nodata=0,
        description="Raw CAT/XGB groundwater potential class",
        predictor=2,
    )

    write_single_band(
        FINAL_CLASS_TIF,
        final_class_array,
        output_profile,
        dtype="uint8",
        nodata=0,
        description=(
            "Ranked and calibrated CAT/XGB groundwater potential class"
        ),
        predictor=2,
    )

    write_single_band(
        CONFIDENCE_TIF,
        confidence_array,
        output_profile,
        dtype="float32",
        nodata=-9999.0,
        description="CAT/XGB maximum ensemble probability",
        predictor=3,
    )

    write_single_band(
        SUITABILITY_TIF,
        suitability_array,
        output_profile,
        dtype="float32",
        nodata=-9999.0,
        description=(
            "Blended CAT/XGB and rank-weighted groundwater suitability score"
        ),
        predictor=3,
    )

    write_single_band(
        MODEL_SUITABILITY_TIF,
        model_suitability_array,
        output_profile,
        dtype="float32",
        nodata=-9999.0,
        description="CAT/XGB model-only groundwater suitability score",
        predictor=3,
    )

    write_single_band(
        WEIGHTED_FUSED_TIF,
        ranked_prior_array,
        output_profile,
        dtype="float32",
        nodata=-9999.0,
        description=(
            "Rank-weighted fused hydrogeological suitability prior"
        ),
        predictor=3,
    )

    class_area_table(
        raw_class_array,
        common_mask,
    ).to_csv(
        AREA_RAW_CSV,
        index=False,
    )

    class_area_table(
        final_class_array,
        common_mask,
    ).to_csv(
        AREA_FINAL_CSV,
        index=False,
    )

    THRESHOLDS_JSON.write_text(
        json.dumps(
            {
                "version": VERSION,
                "training_ranked_score_thresholds": training_thresholds,
                "final_model_suitability_thresholds": final_thresholds,
                "target_class_proportions": TARGET_CLASS_PROPORTIONS,
                "ranking": LAYER_RANKS,
                "rank_sum_weights": LAYER_WEIGHTS,
                "continuous_directions": CONTINUOUS_DIRECTIONS,
                "manual_category_overrides": MANUAL_CATEGORY_SUITABILITY,
                "interaction_features": INTERACTION_FEATURES,
                "model_suitability_weight": MODEL_SUITABILITY_WEIGHT,
                "ranked_prior_weight": RANKED_PRIOR_WEIGHT,
                "lithology_included": True,
                "note": (
                    "Final thresholds are post-model quantile calibration. "
                    "They control class dominance but are not field-validated "
                    "groundwater thresholds."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    save_class_png(
        RAW_CLASS_TIF,
        RAW_CLASS_PNG,
        "Raw CAT/XGB 5-Class Groundwater Potential — Entire Nigeria",
    )

    save_class_png(
        FINAL_CLASS_TIF,
        FINAL_CLASS_PNG,
        (
            "Ranked CAT/XGB 5-Class Groundwater Potential "
            "— Entire Nigeria"
        ),
    )

    save_continuous_png(
        CONFIDENCE_TIF,
        CONFIDENCE_PNG,
        "CAT/XGB Ensemble Confidence — Entire Nigeria",
        "Maximum Ensemble Probability",
    )

    save_continuous_png(
        SUITABILITY_TIF,
        SUITABILITY_PNG,
        (
            "Blended CAT/XGB + Ranked Groundwater Suitability "
            "— Entire Nigeria"
        ),
        "Suitability Score (0 = Very Low, 1 = Very High)",
    )

    save_continuous_png(
        MODEL_SUITABILITY_TIF,
        MODEL_SUITABILITY_PNG,
        "CAT/XGB Model-Only Groundwater Suitability — Entire Nigeria",
        "Model Suitability Score",
    )

    save_continuous_png(
        WEIGHTED_FUSED_TIF,
        WEIGHTED_FUSED_PNG,
        "Rank-Weighted Fused Groundwater Suitability — Entire Nigeria",
        "Rank-Weighted Hydrogeological Score",
    )

    README_FILE.write_text(
        f"""Entire Nigeria CAT/XGB Groundwater Pipeline

Version:
  {VERSION}

Input folder:
  {LOCAL_INPUT_DIR}

S3 output folder:
  s3://{BUCKET}/{OUTPUT_PREFIX}/

Fused 10-band predictor stack:
  s3://{BUCKET}/{OUTPUT_PREFIX}/{FUSED_TIF.name}

Rank-weighted fused suitability raster:
  s3://{BUCKET}/{OUTPUT_PREFIX}/{WEIGHTED_FUSED_TIF.name}

User-supplied ranking and rank-sum weights:
  1. Rainfall            {LAYER_WEIGHTS["rainfall"]:.6f}
  2. Lithology           {LAYER_WEIGHTS["lithology"]:.6f}
  3. Lineament density   {LAYER_WEIGHTS["lineament"]:.6f}
  4. TWI                 {LAYER_WEIGHTS["TWI"]:.6f}
  5. Slope               {LAYER_WEIGHTS["slope"]:.6f}
  6. Soil                {LAYER_WEIGHTS["soil_texture"]:.6f}
  7. NDWI                {LAYER_WEIGHTS["NDWI"]:.6f}
  8. Drainage density    {LAYER_WEIGHTS["drainage"]:.6f}
  9. NDVI                {LAYER_WEIGHTS["NDVI"]:.6f}
 10. LULC                {LAYER_WEIGHTS["LULC"]:.6f}

How categorical layers are handled:
  Lithology, soil texture and LULC codes are treated strictly as category
  labels. Their suitability values are inferred from the median continuous
  hydrogeological context for each category, with shrinkage for rare classes.
  Optional expert overrides can replace individual inferred scores.
  Numeric category IDs are never normalized or treated as ordered magnitudes.

Nonlinear model interactions:
  rainfall x lithology
  lineament x lithology
  rainfall x TWI
  low-slope suitability x soil suitability
  NDWI x drainage suitability

Final continuous suitability:
  70% CAT/XGB model suitability + 30% rank-weighted fused prior.

Fused band order:
  1 drainage
  2 lineament
  3 lithology
  4 LULC
  5 NDVI
  6 NDWI
  7 rainfall
  8 slope
  9 soil_texture
 10 TWI

Grid:
  Native reference: {native_height} x {native_width}
  Analysis output:  {target_height} x {target_width}
  Analysis scale:   {ANALYSIS_SCALE}

Main final map:
  {FINAL_CLASS_TIF.name}

Scientific limitation:
  No independent pixel-level VES or borehole target was used.
  Internal metrics measure reproduction of ranked pseudo-labels.
  Final class proportions are calibrated assumptions.
""",
        encoding="utf-8",
    )

    print("\nStep 9/9: Upload fused dataset and all outputs", flush=True)

    if UPLOAD_OUTPUTS:
        upload_outputs(s3_client)

    print("\nDONE", flush=True)

    print(
        f"Fused 10-band dataset: "
        f"s3://{BUCKET}/{OUTPUT_PREFIX}/{FUSED_TIF.name}",
        flush=True,
    )

    print(
        f"Rank-weighted fused suitability: "
        f"s3://{BUCKET}/{OUTPUT_PREFIX}/{WEIGHTED_FUSED_TIF.name}",
        flush=True,
    )

    print(
        f"Final map: "
        f"s3://{BUCKET}/{OUTPUT_PREFIX}/{FINAL_CLASS_TIF.name}",
        flush=True,
    )

    del layer_arrays
    del layer_masks
    gc.collect()


if __name__ == "__main__":
    try:
        main()

    except Exception:
        traceback_text = traceback.format_exc()

        ERROR_FILE.write_text(
            traceback_text,
            encoding="utf-8",
        )

        print(traceback_text, flush=True)

        try:
            boto3.client("s3").upload_file(
                str(ERROR_FILE),
                BUCKET,
                f"{OUTPUT_PREFIX}/{ERROR_FILE.name}",
            )
        except Exception:
            pass

        sys.exit(1)
