#!/usr/bin/env python3
from aws_galileo_embed_common import run_embedding_job

BUCKET = "sachet-imerg-nigeria"
FUSED_KEY = "output/josef_3model/fused/fused_extended.nc4"
OUT_PREFIX = "output/josef_3model/galileo/extended"

FEATURES = [
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

if __name__ == "__main__":
    run_embedding_job(
        bucket=BUCKET,
        fused_key=FUSED_KEY,
        out_prefix=OUT_PREFIX,
        feature_names=FEATURES,
        galileo_repo="/home/ec2-user/galileo",
        galileo_model_dir="/home/ec2-user/galileo/data/models/nano",
        workdir="/tmp/galileo_extended",
        batch_size=1024,
        months_value=6,
        device="cpu",
        save_pca_png=False,
    )
