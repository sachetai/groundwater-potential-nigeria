# Groundwater Potential Mapping in Nigeria

This repository contains the Python workflows used for groundwater potential mapping in Northwestern Nigeria and the subsequent whole-Nigeria extension.

## Northwestern Nigeria

The NW Nigeria workflow includes:

- thematic-layer fusion and preprocessing
- Galileo embedding generation
- Galileo-enhanced groundwater modeling
- CatBoost / XGBoost machine-learning workflows
- VES-based field validation
- AHP vs. ML spatial comparison
- raw vs. Galileo model comparison
- NW Nigeria AOI clipping and output preparation

The principal thematic inputs include:

1. Rainfall
2. Lithology
3. Lineament density
4. Topographic Wetness Index (TWI)
5. Slope
6. Soil
7. NDWI
8. Drainage density
9. NDVI
10. Land Use / Land Cover (LULC)

## Entire Nigeria

`entire_nigeria/entire_nigeria_cat_xgb_fast.py` contains the whole-Nigeria CAT/XGB groundwater-potential workflow. The pipeline aligns the ten thematic layers, produces a fused dataset, applies the ranked hydrogeological importance of the input variables, trains CatBoost and XGBoost models, and generates five groundwater-potential classes:

- Very High
- High
- Moderate
- Low
- Very Low

## Data

Large raster datasets, model outputs, VES datasets, embeddings, and other research data are not stored in this repository.

## Repository Structure

```text
groundwater-potential-nigeria/
├── nw_nigeria/
├── entire_nigeria/
├── README.md
└── .gitignore

