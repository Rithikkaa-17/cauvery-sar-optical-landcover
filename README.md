# SAR-optical land-cover classification, Cauvery Delta (Tamil Nadu, India), June 2024 - May 2025

Reference data, Sentinel-1/Sentinel-2 features and code accompanying the manuscript
"Sentinel-1 SAR and Sentinel-2 Optical Land-Cover Classification with a Prototype
Decision-Support Dashboard for the Monsoon-Driven Cauvery Delta".

Study area: 10.85-11.15 N, 79.05-79.45 E. Classes: Rice, Vegetation, Built-up (Other excluded).

## data/
| File | Content |
|---|---|
| `sample_points_1000_seed42.csv` | Simple random sample (numpy default_rng seed 42, minimum spacing 200 m). Points P0001-P0500 were labelled. |
| `labels_interpreter1_points1-500.csv` | Visual-interpretation labels by interpreter 1 (Rice / Vegetation / Built-up / Other) with labelling timestamps. |
| `labels_interpreter2_points1-100.csv` | Independent labels by interpreter 2 for points 1-100. |
| `features_labels_449.csv` | The 449 analysed points: label, coordinates, annual NDVI, VH, VV, VV_VH_ratio (= VV_dB / VH_dB) and VV_minus_VH_dB; the 29-date Sentinel-1 VV and VH series (columns VV_YYYYMMDD, VH_YYYYMMDD); and 12 time-series summary features. |

## gee/ (Google Earth Engine Code Editor scripts)
- `labelling_tool.js` - point-labelling tool used by the interpreters.
- `extract_annual_features.js` - annual NDVI and Sentinel-1 features at the labelled points.
- `extract_s1_timeseries.js` - per-date Sentinel-1 VV/VH at the labelled points.
Asset paths inside the scripts refer to the authors' Earth Engine project; replace them with your own copies of the CSV files.

## analysis/ (Python 3: numpy, pandas, scipy, scikit-learn, tensorflow)
- `make_sample.py` - regenerates `sample_points_1000_seed42.csv` exactly.
- `r5_analysis.py` - model definitions (logistic regression, Random Forest, the fully connected network) and statistics helpers.
- `oof.py`, `ts_eval.py` - repeated stratified 5-fold cross-validation with random and 1 km block folds (Tables 3-4, Figure 13).
- `v2_extras.py` - permutation importance (Table 5) and training curves (Figure 12).
The scripts read working copies named `v2_analysis.csv` (annual features) and `ts2.csv` (time series); both are subsets of the columns in `features_labels_449.csv`.

## Licence
Data: CC BY 4.0. Code: MIT.
