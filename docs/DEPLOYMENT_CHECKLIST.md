# Deployment Checklist: Rimmel Demand Forecasting System

**Target Deployment**: Production Pre-Commit & Server Hosting Gate  
**Model Architecture**: Certified Production Model v1 — Exp6  
**Date**: September 23, 2026  

Use this checklist to verify that the repository is completely production-ready before pushing commits to GitHub or deploying the application to an internal server.

---

## 1. Environment & Dependencies

- [x] **Python Version Compatibility**: Verified on Python 3.10 – 3.12.
- [x] **Locked Dependencies in `requirements.txt`**:
  - `pandas>=2.0.0,<3.0.0`
  - `numpy>=1.24.0,<3.0.0`
  - `scikit-learn>=1.3.0,<2.0.0`
  - `lightgbm>=4.0.0,<5.0.0` *(Certified model engine)*
  - `openpyxl>=3.1.0,<4.0.0`
  - `xlsxwriter>=3.1.0,<4.0.0`
  - `pyarrow>=14.0.0,<25.0.0`
  - `streamlit>=1.28.0,<2.0.0`
  - `plotly>=5.18.0,<7.0.0`
  - `reportlab>=4.0.0,<6.0.0`
  - `pypdf>=4.0.0,<7.0.0`
- [x] **Clean Installation Test**: Running `pip install -r requirements.txt` succeeds with zero dependency conflicts.

---

## 2. Model Artifacts & Cryptographic Verification

- [x] **Model File Exists**: `models/production_lgbm_model.pkl` (422 KB).
- [x] **Features Metadata Exists**: `models/production_features.json` (74 causal features).
- [x] **Config File Exists**: `models/production_model_config.json`.
- [x] **SHA-256 Checksum Verified**:
  - Model binary: `821ba6acbea6f2a7cc81527810411389a64034c2f0c93a9636fab4d7f4605508`
  - Features JSON: `824411ebcf4659c42ecca89cc114ff5eb636462125c8eeb84864b3d61390e63e`
- [x] **Exact Hyperparameters Enforced**:
  `n_estimators=150`, `max_depth=6`, `num_leaves=31`, `learning_rate=0.05`, `random_state=42`.

---

## 3. Database & Data Integrity

- [x] **Database File Exists**: `data/rimmel_clean.db` (577 MB).
- [x] **Raw Table Preservation**: `raw_transactions` contains 101,085 untouched rows with 1-indexed `source_row_id`.
- [x] **Feature Table Verification**: `ml_features_zero` contains 573,678 rows across 89 columns.
- [x] **Canonical Catalog Size**: Exactly 674 Canonical SKUs resolved in `sku_master`.
- [x] **Shared Warehouse Inventory Policy**: Stock is tracked in a single central warehouse pool per SKU and is **never summed across platforms**.

---

## 4. Automated Regression & Unit Test Gate

Execute the complete test suite:
```bash
python -m unittest discover tests
```

- [x] **12/12 Tests Passing** (execution time ~4.4s).
- [x] **Model & Config Parameters Verified**: `test_model_and_config_parameters` PASS.
- [x] **74 Causal Features Verified**: `test_feature_metadata_and_long_term_features` PASS.
- [x] **Validation Report Structure & Sums Verified**: `test_report_1_validation_excel_structure` PASS (2,069 actual units).
- [x] **Forward Forecast Structure & Bounds Verified**: `test_report_2_production_forecast_excel_structure` PASS (all predictions non-negative).
- [x] **Backward-Compatible Copies Verified**: `test_backward_compatibility_copies` PASS.
- [x] **Settings Configuration Constants Verified**: `test_config_settings_parameters` PASS.
- [x] **Database Row Counts & Tables Verified**: `test_database_integrity_and_row_counts` PASS.
- [x] **Zero Future Data Leakage Verified**:
  - `test_zero_current_day_target_leakage` PASS
  - `test_lag_1_exact_match_previous_day` PASS
  - `test_v7_window_bounds` PASS
  - `test_validation_partition_isolation` PASS
  - `test_platform_feature_isolation` PASS

---

## 5. Client Reporting Deliverables

Verify that the four primary Excel files exist in `reports/` and are populated:
1. `reports/Rimmel_Validation_Sep01_Sep10_2026.xlsx`
2. `reports/Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx`
3. `reports/validation_report_sep_01_to_10_2026.xlsx`
4. `reports/production_forecast_sep_11_to_20_2026.xlsx`

- [x] **Validation Excel**: Exactly 674 SKU rows (1 row per SKU) + 1 header = 675 rows.
- [x] **Forward Forecast Excel**: Exactly 674 SKU rows + 1 header = 675 rows.
- [x] **Validation Actual Units**: Exactly 2,069 physical units.
- [x] **Validation Predicted Units**: Exactly 2,079 physical units (+0.48% net variance).
- [x] **Forward Forecast Demand**: Exactly 1,934 physical units (Amazon: 976, eBay: 948, Website: 7, Other: 3).

To regenerate reports from scratch at any time:
```bash
python -m src.generate_client_reports
```

---

## 6. Streamlit User Interface

- [x] **App Launches Cleanly**: `streamlit run app.py` starts without syntax or import errors.
- [x] **Read-Only Operation**: App has zero endpoints for writing to `data/rimmel_clean.db` or triggering unauthorized model retraining.
- [x] **All 5 Tabs Render**:
  - Tab 1: Product Inspector (renders Plotly historical + validation + forecast chart for all 674 SKUs).
  - Tab 2: Data View (filters forward forecast and validation tables).
  - Tab 3: Forecast Overview (portfolio rollup, platform breakdown, top 10 SKUs).
  - Tab 4: Validation (retrospective benchmark metrics, actual vs predicted chart).
  - Tab 5: Inventory / Planning (shared warehouse stock, Days of Cover, risk filters).
- [x] **Download Buttons Work**: Direct downloads for Validation Excel and Forward Forecast Excel work as expected.

---

## 7. Git Hygiene & Security Audit

- [x] **No Hardcoded User Paths**: Zero occurrences of `C:\Users\` in code or tests.
- [x] **No Secrets / Tokens / Passwords**: Verified zero credentials, API keys, or private tokens committed.
- [x] **Clean `.gitignore`**:
  - Scratch pads (`scratch/`, `scratch_prompt.txt`, `*.log`, `*.tmp`) properly ignored.
  - Python caches (`__pycache__/`, `*.pyc`) ignored.
  - Virtual environments (`venv/`, `.venv/`) ignored.
  - Official client Excel deliverables (`reports/Rimmel_*.xlsx`) preserved.
- [x] **Documentation Updated**:
  - `README.md` reflects Exp6 production architecture.
  - `CODEBASE_ARCHITECTURE.md` available for developers.
  - `PROJECT_COMPLETE_TECHNICAL_AND_BUSINESS_GUIDE.md` (129 KB, 34 sections) and `.pdf` (57 pages) available.

---

## 8. Deployment Sign-Off

| Check | Responsible Engineer | Status | Date |
| :--- | :--- | :---: | :--- |
| **Model Verification** | ML Production Team | **PASS** | Sep 23, 2026 |
| **Pipeline & Data Audit** | Data Engineering Team | **PASS** | Sep 23, 2026 |
| **Reporting & Reconciliation** | Inventory Planning Team | **PASS** | Sep 23, 2026 |
| **Repository Hygiene & Security**| DevOps / QA Team | **PASS** | Sep 23, 2026 |

**FINAL STATUS: 100% PRODUCTION READY FOR GITHUB COMMIT AND DEPLOYMENT**
