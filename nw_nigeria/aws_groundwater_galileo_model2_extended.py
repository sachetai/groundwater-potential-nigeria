#!/usr/bin/env python3
from aws_groundwater_galileo_common import run_galileo_model_job

BUCKET = "sachet-imerg-nigeria"
FUSED_KEY = "output/josef_3model/fused/fused_extended.nc4"
EMB_KEY = "output/josef_3model/galileo/extended/galileo_embeddings.npy"
META_KEY = "output/josef_3model/galileo/extended/galileo_metadata.csv"
OUT_PREFIX = "output/josef_3model/galileo_models/extended"

FEATURE_ORDER = [
    "Lithology",
    "Lineament_Density",
    "Drainage_Density",
    "Slope",
    "Soil",
    "rainfall",
    "LULC",
    "NDVI",
    "NDWI",
    "TWI",
]

WEIGHTS = {
    "Lithology": 1.30,
    "Lineament_Density": 1.15,
    "Drainage_Density": -1.00,
    "Slope": -1.10,
    "Soil": 0.25,
    "rainfall": 0.95,
    "LULC": 0.20,
    "NDVI": 0.45,
    "NDWI": 0.85,
    "TWI": 1.05,
}

if __name__ == "__main__":
    run_galileo_model_job(
        bucket=BUCKET,
        fused_key=FUSED_KEY,
        emb_key=EMB_KEY,
        meta_key=META_KEY,
        out_prefix=OUT_PREFIX,
        feature_order=FEATURE_ORDER,
        weights=WEIGHTS,
        subset_name="extended",
        max_train_samples=120000,
        rf_trees=300,
        xgb_estimators=400,
        cat_iters=500,
    )
