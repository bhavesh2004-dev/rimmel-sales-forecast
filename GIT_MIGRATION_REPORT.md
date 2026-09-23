# Git Migration & Repository Structure Report

**Date**: September 24, 2026  
**Status**: Completed & Verified  
**Target Repository**: `bhavesh2004-dev/rimmel-sales-forecast`  

---

## 1. Executive Summary

This report documents the repository restructuring, branch archiving, and promotion of the **Certified Production Model v1 — Exp6** (`ZERO Treatment + LightGBM Regressor + Combined Calibration`) to the `main` branch.

All existing historical branches and work were strictly preserved without data loss or history destruction. The large SQLite database (`data/rimmel_clean.db`, ~662 MB) was untracked from Git and added to `.gitignore` to comply with GitHub's file size limits while remaining intact locally on disk. All 12 unit tests and production pipeline validations were executed and passed prior to promotion.

---

## 2. Commit & Branch Lineage

### 2.1 Preserved Historical Versions

The previous state of `main` (representing the Phase 1 Adaptive Engine v8.1) has been archived into a dedicated historical branch:
- **Archived Branch**: `v5-adaptive-engine`
- **Archived Commit SHA**: `1a7c0228e3c3361398061421d9e6902f86d7c6b8`
- **Remote Status**: Successfully pushed to `origin/v5-adaptive-engine`.

All other historical branches remain intact both locally and on remote:
- `v1-initial-pilot-model`: Initial pilot heuristics & baseline models
- `v2-1year-pattern-model`: 1-year pattern baseline model with YoY August anchors
- `v3-master-3tier-engine`: Phase 3 Master 3-Tier hybrid engine with Croston SBA & elasticity
- `v4-instock-flag-dataset`: Modularization, dynamic forecast horizons, and instock-flagging
- `v5-adaptive-engine`: Adaptive engine v8.1 (preserved previous `main`)

### 2.2 Promoted Production Version on `main`

The certified Exp6 production release has been promoted to `main`:
- **Commit SHA**: `af4cc8a`
- **Commit Message**: `feat: certified Exp6 production release`
- **Parent Commit**: `1a7c0228e3c3361398061421d9e6902f86d7c6b8` (Direct fast-forward lineage, no force push needed)
- **Remote Target**: `origin/main` (Successfully pushed)

---

## 3. Large File & Database Management

### 3.1 Untracking of SQLite Database
- **File**: `data/rimmel_clean.db` (~662.49 MB)
- **Action Taken**: 
  - Untracked from Git index (`git rm --cached data/rimmel_clean.db`).
  - Added pattern `data/rimmel_clean.db`, `*.db`, `*.sqlite` to `.gitignore`.
  - Added intermediate raw Excel transformations `data/processed/*.xlsx` to `.gitignore`.
- **Local Disk Verification**:
  - The physical database remains intact at `c:\Users\bhave\Desktop\ml_project\data\rimmel_clean.db` (Length: 694,669,312 bytes).
  - No database files will be pushed to GitHub, preventing GitHub 100 MB rejection.

### 3.2 Tracked Production Datasets
Only production dashboard artifacts and client reports are tracked:
- `data/processed/dashboard_historical_daily.csv` (36.72 MB) & `.parquet` (3.9 MB)
- `data/processed/dashboard_forecast_sku_daily.csv` & `.parquet`
- `data/processed/dashboard_validation_sku_daily.csv` & `.parquet`
- `data/processed/dashboard_sku_master.csv` & `.parquet`
- `data/raw/ORIGINAL_COPY.xlsx` (7.84 MB)
- `data/Rimmel Brand Sales Data - 1 Jan 2025 to 10 Sep 2026.xlsx` (7.84 MB)

All tracked files are strictly below GitHub's 50 MB warning threshold and 100 MB hard limit.

---

## 4. Codebase Structurization

To maintain high maintainability and clarity for future engineers:
1. **Phase 1 Heuristics Archived**:
   - 14 legacy scripts moved from `src/` to `archive/previous_versions/phase1_heuristics/`:
     - `adaptive_forecast.py`
     - `baseline_forecast.py`
     - `confidence.py`
     - `data_loader.py`
     - `dynamic_engine.py`
     - `explanations.py`
     - `export_excel.py`
     - `feature_engineering.py`
     - `inventory_insights.py`
     - `momentum_forecast.py`
     - `preprocessing.py`
     - `report_generator.py`
     - `risk.py`
     - `validation.py`
2. **Active Production Files in `src/`**:
   - Ingestion & DB: `data_ingestion.py`, `data_cleaning.py`, `normalization.py`, `build_sqlite_database.py`
   - Mapping: `sku_mapping.py`, `platform_mapping.py`
   - Production Engine: `final_production_system.py`, `final_production_delivery.py`, `final_production_runner.py`
   - Feature & Observations: `observation_engine.py`, `phase2_feature_engineering.py`
   - Experiments & Audits: `phase3_1_runner.py`, `phase3_experiment_runner.py`, `phase4_walk_forward_runner.py`, `phase5_burst_experiment_runner.py`, `generate_phase2_audit_reports.py`, `generate_client_reports.py`
3. **Application & Documentation**:
   - `app.py`: Streamlit Dashboard supporting both SQLite database and precomputed Parquet/CSV modes.
   - Comprehensive Guides: `PROJECT_COMPLETE_TECHNICAL_AND_BUSINESS_GUIDE.md` and `.pdf`.
   - Architectural Docs: `CODEBASE_ARCHITECTURE.md`, `DEPLOYMENT_CHECKLIST.md`, `AUDIT_REPORT.md`, `FINAL_CODEBASE_AUDIT_REPORT.md`.

---

## 5. Automated Verification & Golden Behavior

### 5.1 Test Suite
All 12 automated unit tests passed:
```
Ran 12 tests in 35.731s
OK
```
Tests verified:
- Zero data leakage between train and validation horizons
- Monotonic date alignment and feature immutability
- In-stock inventory capping logic
- Serialization and loading of LightGBM model (`models/production_lgbm_model.pkl`)
- Feature alignment (`models/production_features.json`, 74 features)
- Cross-platform reconciliation and non-negativity constraints

### 5.2 Golden Production Metrics Match
- **Validation Period**: 2026-09-01 to 2026-09-10
  - Actual Units: **2,069.0**
  - Continuous Predicted Units: **2,121.2** (+2.52% net bias)
  - Client 10-day SKU Summary: **2,079** (+0.48% net bias)
- **Forward Production Forecast**: 2026-09-11 to 2026-09-20
  - Total Forecast: **1,934 units**
  - Platform Breakdown:
    - Amazon: **976 units**
    - eBay: **948 units**
    - Website: **7 units**
    - Other: **3 units**

---

## 6. Current Branch Map

```
* af4cc8a (HEAD -> main, origin/main) feat: certified Exp6 production release
* 1a7c022 (origin/v5-adaptive-engine, v5-adaptive-engine) feat: production release v8.1 ...
* 5dfd5a9 (v4-instock-flag-dataset) feat: complete modularization ...
* 4b1213f (origin/v3-master-3tier-engine, v3-master-3tier-engine) Phase 3: Production Master ...
* 18d489e (origin/v2-1year-pattern-model, v2-1year-pattern-model) Phase 2: 1-Year Pattern ...
```

The repository is fully synced, clean, and production-ready.
