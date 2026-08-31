#!/usr/bin/env python3
"""
aws_fuse_nw_2024_layers.py

Fuse NW Nigeria thematic layers (already NW-cropped) into a single NetCDF4 (.nc4),
and (optionally) export each layer as a GeoTIFF — then upload outputs to S3.

 Key fixes vs your failing run:
- NO s3fs.put() (removes: AttributeError: 'dict' object has no attribute 'upper')
- Uses boto3 upload_file() for uploads (stable)
- Uses /vsis3/ for reading rasters directly from S3 with rioxarray
- Adds a CF-style "spatial_ref" coordinate WITH:
    - crs_wkt
    - GeoTransform (6 numbers)
  so your downstream scripts can read CRS/transform reliably.

S3 layout assumed:
  s3://<bucket>/datasets_2024/<layer>.tif
  s3://<bucket>/output/<writes outputs here>

Run:
  conda activate imerg
  python aws_fuse_nw_2024_layers.py \
    --bucket sachet-imerg-nigeria \
    --datasets_prefix datasets_2024/ \
    --output_prefix output/ \
    --out_nc_name fused_nw_2024.nc4 \
    --export_tifs 0
"""

import os
import shutil
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
import rasterio
from rasterio.enums import Resampling

import boto3
from botocore.exceptions import ClientError

from tqdm import tqdm


# -------------------------
# Default config (override via CLI)
# -------------------------
DEFAULT_REFERENCE_LAYER_NAME = "NDVI"

# Each key is the canonical output variable name in NetCDF.
# Each value is a list of candidate filenames under datasets_prefix.
THEMATIC_LAYERS: Dict[str, List[str]] = {
    "NDVI": ["NDVI.tif", "ndvi.tif"],
    "NDWI": ["NDWI.tif", "ndwi.tif"],
    "LST": ["LST.tif", "lst.tif"],
    "LULC": ["LULC.tif", "lulc.tif", "Landcover.tif", "landcover.tif"],
    "Slope": ["Slope.tif", "slope.tif"],
    "SPI": ["SPI.tif", "spi.tif"],
    "TWI": ["TWI.tif", "twi.tif"],
    "TPI": ["TPI.tif", "tpi.tif"],
    "Lineament_Density": ["Lineament_Density.tif", "lineament_density.tif", "LineamentDensity.tif"],
    "Drainage_Density": ["Drainage_Density.tif", "drainage_density.tif", "DrainageDensity.tif"],
    "Soil": ["Soil.tif", "soil.tif"],
    # IMPORTANT: per your advisor -> use Lithology, not Geology
    "Lithology": ["Lithology.tif", "lithology.tif"],

    # rainfall (your NW 2024 total raster)
    "rainfall": ["NW_Rainfall_2024_total.tif", "NW_Rainfall_2024_total.tif", "rainfall.tif", "Rainfall.tif"],
}

# Categorical/discrete rasters -> nearest neighbor
CATEGORICAL = {"LULC", "Soil", "Lithology"}

# Compression for NetCDF
NETCDF_COMPLEVEL = 4


# -------------------------
# S3 helpers (boto3)
# -------------------------
def s3_client():
    return boto3.client("s3")


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def vsis3_path(bucket: str, key: str) -> str:
    # GDAL /vsis3/ path (works with AWS creds on EC2/IAM role)
    return f"/vsis3/{bucket}/{key}"


def s3_list_keys(bucket: str, prefix: str) -> List[str]:
    """List all object keys under prefix."""
    cli = s3_client()
    keys: List[str] = []
    paginator = cli.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            keys.append(obj["Key"])
    return keys


def s3_upload_file(bucket: str, local_path: str, key: str):
    """Stable upload (avoids s3fs/aiobotocore checksum bugs)."""
    cli = s3_client()
    cli.upload_file(local_path, bucket, key)


# -------------------------
# Raster helpers
# -------------------------
def open_raster_vsis3(bucket: str, key: str) -> xr.DataArray:
    """
    Open GeoTIFF from S3 using /vsis3/ path into a 2D (y,x) DataArray with CRS.
    """
    path = vsis3_path(bucket, key)
    da = rioxarray.open_rasterio(path, masked=True)  # typically (band, y, x)

    # Force 2D
    if "band" in da.dims:
        da2d = da.isel(band=0, drop=True)
    else:
        da2d = da

    # Ensure CRS exists
    if da2d.rio.crs is None:
        raise RuntimeError(f"CRS missing for {s3_uri(bucket, key)}")

    # Ensure spatial dims named y/x
    da2d = da2d.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    return da2d


def reproject_to_match(layer: xr.DataArray, template: xr.DataArray, name: str) -> xr.DataArray:
    """
    Reproject/resample layer to match template grid.
    """
    resampling = Resampling.nearest if name in CATEGORICAL else Resampling.bilinear
    out = layer.rio.reproject_match(template, resampling=resampling)
    out = out.astype(np.float32)
    out.name = name
    return out


def print_layer_stats(name: str, da: xr.DataArray):
    arr = da.values
    valid = np.isfinite(arr)
    n_valid = int(valid.sum())
    n_total = int(arr.size)
    if n_valid == 0:
        print(f"✗ {name}: ALL NaN (valid=0/{n_total})")
        return
    v = arr[valid]
    print(
        f"✓ {name}: valid={n_valid:,}/{n_total:,} ({100.0*n_valid/n_total:.2f}%) | "
        f"min={float(np.nanmin(v)):.4f} mean={float(np.nanmean(v)):.4f} max={float(np.nanmax(v)):.4f}"
    )


def add_spatial_ref(ds: xr.Dataset, ref: xr.DataArray) -> xr.Dataset:
    """
    Add a 'spatial_ref' coordinate compatible with downstream code:
      - attrs['crs_wkt']
      - attrs['GeoTransform'] = "a b c d e f" (6 numbers)
    """
    crs = ref.rio.crs
    transform = ref.rio.transform()  # affine

    # GDAL GeoTransform order: (c, a, b, f, d, e)
    # rasterio Affine is:
    #   | a  b  c |
    #   | d  e  f |
    #   | 0  0  1 |
    gt = (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)
    gt_str = " ".join([f"{x:.16g}" for x in gt])

    spatial_ref = xr.DataArray(
        0,
        dims=(),
        name="spatial_ref",
        attrs={
            "crs_wkt": crs.to_wkt(),
            "GeoTransform": gt_str,
        },
    )

    ds = ds.assign_coords(spatial_ref=spatial_ref)
    return ds


# -------------------------
# Resolve layer keys from S3
# -------------------------
def resolve_layer_keys(bucket: str, datasets_prefix: str, layers: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Resolve each thematic layer to an existing S3 key under datasets_prefix.
    We list once, then match filenames.
    """
    all_keys = s3_list_keys(bucket, datasets_prefix)
    if not all_keys:
        raise RuntimeError(f"No objects found under {s3_uri(bucket, datasets_prefix)}")

    # Build a set for O(1) lookups
    key_set = set(all_keys)

    resolved: Dict[str, str] = {}
    missing: List[str] = []

    for name, candidates in layers.items():
        found = None
        for fn in candidates:
            k = f"{datasets_prefix}{fn}"
            if k in key_set:
                found = k
                break
        if found is None:
            missing.append(name)
        else:
            resolved[name] = found

    if missing:
        # show what actually exists (a little)
        sample = sorted(list(key_set))[:30]
        raise RuntimeError(
            "These layers were not found in S3 under "
            f"{s3_uri(bucket, datasets_prefix)}: {', '.join(missing)}\n"
            f"Sample keys under prefix:\n  - " + "\n  - ".join(sample)
        )

    return resolved


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--datasets_prefix", default="datasets_2024/")
    ap.add_argument("--output_prefix", default="output/")
    ap.add_argument("--out_nc_name", default="fused_nw_2024.nc4")
    ap.add_argument("--reference_layer", default=DEFAULT_REFERENCE_LAYER_NAME)
    ap.add_argument("--export_tifs", type=int, default=0, help="1 to export each layer GeoTIFF, else 0")
    ap.add_argument("--workdir", default="/tmp/nw_fuse_work")
    args = ap.parse_args()

    bucket = args.bucket
    datasets_prefix = args.datasets_prefix if args.datasets_prefix.endswith("/") else (args.datasets_prefix + "/")
    output_prefix = args.output_prefix if args.output_prefix.endswith("/") else (args.output_prefix + "/")

    # Fresh workdir
    shutil.rmtree(args.workdir, ignore_errors=True)
    os.makedirs(args.workdir, exist_ok=True)
    outdir = os.path.join(args.workdir, "out")
    os.makedirs(outdir, exist_ok=True)

    # 1) Resolve keys
    resolved = resolve_layer_keys(bucket, datasets_prefix, THEMATIC_LAYERS)

    if args.reference_layer not in resolved:
        raise RuntimeError(
            f"Reference layer '{args.reference_layer}' not resolved. "
            f"Available: {list(resolved.keys())}"
        )

    ref_key = resolved[args.reference_layer]
    print(f"Loading reference raster grid: {s3_uri(bucket, ref_key)}")

    # 2) Load reference
    ref = open_raster_vsis3(bucket, ref_key)
    print(f"Reference CRS: {ref.rio.crs}")
    print(f"Reference shape: {tuple(ref.shape)} (y,x)")
    print(f"Reference bounds: {ref.rio.bounds()}")

    # 3) Load + match each layer
    fused_vars: Dict[str, xr.DataArray] = {}

    layer_names = list(resolved.keys())
    for name in tqdm(layer_names, desc="Loading+matching layers", unit="layer"):
        key = resolved[name]
        da = open_raster_vsis3(bucket, key)

        # fast path if already same grid
        try:
            same_grid = (
                da.rio.crs == ref.rio.crs
                and da.shape == ref.shape
                and np.allclose(np.array(da.rio.transform()), np.array(ref.rio.transform()))
            )
        except Exception:
            same_grid = False

        if not same_grid:
            da_m = reproject_to_match(da, ref, name)
        else:
            da_m = da.astype(np.float32)
            da_m.name = name

        fused_vars[name] = da_m

    # 4) Build dataset + spatial_ref
    ds = xr.Dataset(fused_vars)

    # Ensure coords named x/y from reference
    ds = ds.assign_coords(x=ref["x"], y=ref["y"])

    # Add spatial_ref w/ CRS + GeoTransform for downstream scripts
    ds = add_spatial_ref(ds, ref)

    # 5) Write NetCDF locally
    out_nc_local = os.path.join(outdir, args.out_nc_name)
    out_nc_key = f"{output_prefix}{args.out_nc_name}"

    print(f"\nWriting NetCDF: {out_nc_local}")
    encoding = {v: {"zlib": True, "complevel": NETCDF_COMPLEVEL} for v in ds.data_vars}
    ds.to_netcdf(out_nc_local, format="NETCDF4", encoding=encoding)

    # Upload NetCDF using boto3 (FIXED)
    print(f"Uploading NetCDF to: {s3_uri(bucket, out_nc_key)}")
    s3_upload_file(bucket, out_nc_local, out_nc_key)

    # 6) Optional GeoTIFF export
    tif_keys: List[str] = []
    if int(args.export_tifs) == 1:
        print("\nExporting GeoTIFFs...")
        for name in tqdm(list(ds.data_vars), desc="GeoTIFF export", unit="tif"):
            tif_local = os.path.join(outdir, f"{name}.tif")
            tif_key = f"{output_prefix}{name}.tif"

            # Use rioxarray export
            ds[name].rio.to_raster(tif_local, compress="DEFLATE", tiled=True)

            print(f"Uploading GeoTIFF: {s3_uri(bucket, tif_key)}")
            s3_upload_file(bucket, tif_local, tif_key)
            tif_keys.append(tif_key)

            # delete local to save disk
            try:
                os.remove(tif_local)
            except Exception:
                pass

    # 7) Quick sanity stats
    print("\n--- Quick sanity stats (after matching to reference grid) ---")
    for name in ds.data_vars:
        print_layer_stats(str(name), ds[name])

    # 8) Cleanup (keep minimal)
    try:
        os.remove(out_nc_local)
    except Exception:
        pass
    shutil.rmtree(args.workdir, ignore_errors=True)

    print("\n✅ DONE")
    print("NetCDF:", s3_uri(bucket, out_nc_key))
    if tif_keys:
        print("GeoTIFFs:")
        for k in tif_keys:
            print(" -", s3_uri(bucket, k))


if __name__ == "__main__":
    main()
