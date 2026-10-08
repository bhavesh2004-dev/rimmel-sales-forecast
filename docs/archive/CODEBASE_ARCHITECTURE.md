# Codebase Architecture: Rimmel Demand Forecasting & Inventory Planning System

**System Version**: Certified Production Model v1 (Exp6 Architecture)  
**Core Engine**: ZERO Treatment + LightGBM Regressor + Combined Calibration ($\alpha=0.10, \beta=0.10$)  
**Target Catalog**: 674 Canonical SKUs across Amazon, eBay, Website, and Other  
**Primary Horizon**: 10 Calendar Days (Validation: Sep 1–10, 2026 | Forward Forecast: Sep 11–20, 2026)  
**Author**: ML Production Engineering Team  
**Date**: September 23, 2026  

---

## 1. High-Level System Architecture

The system operates as a strictly governed, unidirectional data and inference pipeline designed to forecast physical demand for 674 cosmetics products and deliver inventory procurement recommendations.

```
====================================================================================================
                        END-TO-END PRODUCTION PIPELINE ARCHITECTURE
====================================================================================================

[1. RAW TRANSACTION INGESTION]
    │ Source: data/Rimmel Brand Sales Data - 1 Jan 2025 to 10 Sep 2026.xlsx (101,085 rows)
    │ Lineage: 1-indexed source_row_id + SHA-256 backup in data/raw/ORIGINAL_COPY.xlsx
    ▼
[2. DATA NORMALIZATION & MASTER CATALOG RESOLUTION]
    │ Modules: src/normalization.py, src/sku_mapping.py, src/platform_mapping.py
    │ Resolution: 901 raw SKUs unified to 674 Canonical SKUs using item_name/barcode
    │ Platform mapping: 8 commercial channels mapped to Amazon, eBay, Website, Other
    │ Constraint: child_asin strictly Amazon-only; NULL on other channels
    │ Shared Inventory: Stock tracked per SKU; NEVER summed across platforms
    ▼
[3. DAILY OBSERVATION GRID & ZERO TREATMENT]
    │ Table: ml_features_zero in data/rimmel_clean.db (573,678 rows, 406 days)
    │ Grain: DATE x PLATFORM x CANONICAL_SKU (1,413 active series)
    │ States: OBSERVED_SALE, STOCKOUT_DEMAND_CENSORED, PRE_LAUNCH, POST_DISCONTINUATION, etc.
    │ ZERO Treatment: Unobserved sales days modeled as true 0-demand (prevents 12x over-forecasting)
    ▼
[4. FEATURE ENGINEERING (74 CAUSAL FEATURES)]
    │ Module: src/phase2_feature_engineering.py
    │ Strict Causality: All rolling features use .shift(1).rolling() -> 0% future data leakage
    │ 9 Clusters: Recent Demand (85.96% gain), Amazon Signals, Momentum, Metadata, Inventory, etc.
    ▼
[5. MACHINE LEARNING ENGINE (LIGHTGBM REGRESSOR)]
    │ Hyperparameters: n_estimators=150, max_depth=6, num_leaves=31, lr=0.05, seed=42
    │ Serialized Artifacts: models/production_lgbm_model.pkl (SHA-256 verified)
    │ Feature Schema: models/production_features.json
    ▼
[6. POST-HOC MODEL CALIBRATION (EXP6)]
    │ Rule 1 (Zero-Demand): If recent 7-day velocity v7 == 0 -> y_hat = y_hat * 0.10
    │ Rule 2 (Stockout Dampening): If shared warehouse stock == 0 -> y_hat = y_hat * 0.10
    │ Non-Negative Floor: y_hat = max(0.0, y_hat)
    ▼
[7. 10-DAY PHYSICAL AGGREGATION & REPORTING LAYER]
    │ Modules: src/generate_client_reports.py, src/final_production_system.py
    │ Mathematical Rule: Continuous daily predictions are SUMMED across all 10 days BEFORE rounding
    │ Client Grain: Strictly ONE ROW = ONE CANONICAL SKU = ONE 10-DAY PERIOD (674 rows)
    │ Deliverables:
    │   • reports/Rimmel_Validation_Sep01_Sep10_2026.xlsx (Actuals: 2,069 vs Pred: 2,079 [+0.48%])
    │   • reports/Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx (Forecast: 1,934 units)
    ▼
[8. STREAMLIT INTERACTIVE USER INTERFACE]
    │ Entrypoint: app.py (streamlit run app.py)
    │ Security: Read-only access to cached deliverables; zero database or model mutation risks
    │ 5 Tabs: Product Inspector, Data View, Forecast Overview, Validation, Inventory / Planning
====================================================================================================
```

---

## 2. Directory & Component Inventory

### Active Production Modules (`src/`)
These modules form the active, certified production system:

| Module | Primary Function | Key Dependencies |
| :--- | :--- | :--- |
| [`src/generate_client_reports.py`](file:///c:/Users/bhave/Desktop/ml_project/src/generate_client_reports.py) | **Production reporting engine**. Generates official SKU-level 10-day summary Excel reports and dashboard caches. | `lightgbm`, `openpyxl`, `sqlite3` |
| [`src/final_production_system.py`](file:///c:/Users/bhave/Desktop/ml_project/src/final_production_system.py) | **End-to-end retraining & certification pipeline**. Trains LightGBM, evaluates validation holdout, serializes model. | `lightgbm`, `openpyxl`, `sqlite3` |
| [`src/data_ingestion.py`](file:///c:/Users/bhave/Desktop/ml_project/src/data_ingestion.py) | Ingests raw client Excel data with SHA-256 hash audit and lineage tracking. | `pandas`, `hashlib` |
| [`src/platform_mapping.py`](file:///c:/Users/bhave/Desktop/ml_project/src/platform_mapping.py) | Normalizes 8 retail channels into 4 platform groups (Amazon, eBay, Website, Other). | `pandas` |
| [`src/sku_mapping.py`](file:///c:/Users/bhave/Desktop/ml_project/src/sku_mapping.py) | Resolves raw SKUs to 674 Canonical SKUs, fixes child ASIN to Amazon-only. | `pandas`, `numpy` |
| [`src/normalization.py`](file:///c:/Users/bhave/Desktop/ml_project/src/normalization.py) | Builds the 573,678-row daily observation grid with 4-way operational state tagging. | `pandas`, `numpy` |
| [`src/phase2_feature_engineering.py`](file:///c:/Users/bhave/Desktop/ml_project/src/phase2_feature_engineering.py) | Computes all 74 strictly causal features with zero data leakage. | `sqlite3`, `pandas` |
| [`src/build_sqlite_database.py`](file:///c:/Users/bhave/Desktop/ml_project/src/build_sqlite_database.py) | Full database builder converting raw Excel to `data/rimmel_clean.db`. | `sqlite3`, `pandas` |

### Configuration & Testing
- [`config/settings.py`](file:///c:/Users/bhave/Desktop/ml_project/config/settings.py): Central configuration file defining catalog counts, temporal windows, paths, calibration parameters, and LightGBM settings.
- [`tests/test_production_system.py`](file:///c:/Users/bhave/Desktop/ml_project/tests/test_production_system.py): Validates model hyperparameters, feature counts, Excel report schemas, unit reconciliation sums, and database integrity.
- [`tests/test_feature_leakage.py`](file:///c:/Users/bhave/Desktop/ml_project/tests/test_feature_leakage.py): Validates point-in-time causality, $t < T$ rolling bounds, and platform feature isolation.

### Preserved Historical Experiments (`src/` & `archive/`)
These scripts are preserved for audit lineage and research reproducibility:
- `src/phase3_experiment_runner.py`: The controlled ZERO vs. AVERAGE treatment experiment (proved ZERO reduces WAPE from 1,178.77% to 98.22%).
- `src/phase3_1_runner.py`: The Exp1–Exp6 calibration optimization runner (certified Exp6 with 90.54% WAPE and +2.52% bias).
- `src/phase4_walk_forward_runner.py`: 4-window rolling walk-forward temporal cross-validation (100% win rate over baseline).
- `src/phase5_burst_experiment_runner.py`: Quantile / Pinball loss burst experiment (rigorously rejected due to over-forecasting bias).
- `src/final_production_runner.py` & `src/final_production_delivery.py`: Preceding development milestones of the production pipeline.
- `archive/previous_versions/`: Deprecated Phase 1 heuristics (588 SKUs, August 2026).

---

## 3. Database Schema (`data/rimmel_clean.db`)

The SQLite database contains 8 core tables:

1. `raw_transactions` (101,085 rows): Untouched raw transactions with 1-indexed `source_row_id`.
2. `training_window_v5` (61,511 rows): Cleaned, normalized transactions during the ML-eligible period (`2025-08-01` to `2026-09-10`).
3. `daily_sku_platform_grid` (573,678 rows): Full Cartesian product of active (Platform x Canonical SKU) pairs across 406 calendar days.
4. `ml_features_zero` (573,678 rows, 89 columns): The certified production feature table with ZERO treatment and 74 causal features.
5. `ml_features_average` (573,678 rows, 89 columns): The experimental AVERAGE treatment table used as a scientific control.
6. `sku_master` (674 rows): Canonical catalog master with category, pack multiplier, launch date, and lifetime sales.
7. `feature_dictionary` (74 rows): Metadata and leakage audit for each feature.
8. `daily_grid_features` (124,320 rows): Legacy SKU-daily aggregation table.

---

## 4. Security & Model Artifact Verification

### Serialized Model Integrity
The production LightGBM model is stored as a serialized artifact in `models/production_lgbm_model.pkl`. To prevent untrusted binary substitution, cryptographic verification is enforced:

- **Model SHA-256**: `821ba6acbea6f2a7cc81527810411389a64034c2f0c93a9636fab4d7f4605508`
- **Features SHA-256**: `824411ebcf4659c42ecca89cc114ff5eb636462125c8eeb84864b3d61390e63e`
- **Configuration Record**: [`models/production_model_config.json`](file:///c:/Users/bhave/Desktop/ml_project/models/production_model_config.json)
- **Automated Verification**: `tests/test_production_system.py` automatically asserts this SHA-256 checksum during test execution.

### Security Boundaries
- **No Credentials / API Keys**: The project requires zero cloud credentials, API tokens, or external network access.
- **Local SQLite Storage**: Data is contained in a local file-based database (`data/rimmel_clean.db`).
- **Read-Only UI**: The Streamlit interface (`app.py`) only reads pre-computed CSV/Parquet caches and cannot execute database mutations or trigger unauthorized model retraining.

---

## 5. Developer Onboarding Quick-Start

To clone and run this repository on a fresh machine:

```bash
# 1. Clone repository
git clone https://github.com/bhavesh2004-dev/rimmel-sales-forecast.git
cd rimmel-sales-forecast

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite (verifies model, features, and reports in ~4 seconds)
python -m unittest discover tests

# 5. Launch the Streamlit interactive dashboard
streamlit run app.py
```
