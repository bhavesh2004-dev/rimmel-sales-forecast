# Codebase Audit Report: Rimmel Demand Forecasting & Inventory Planning System

**System Version**: Certified Production Model v1 — Exp6  
**Auditor**: ML Production Engineering & Quality Assurance  
**Date**: September 23, 2026  
**Status**: Pre-Deployment Comprehensive Repository & Pipeline Audit  
**Golden Architecture**: ZERO Treatment + LightGBM Regressor (`n_estimators=150, max_depth=6, num_leaves=31, lr=0.05, seed=42`) + Combined Calibration ($\alpha=0.10, \beta=0.10$)  

---

## Executive Summary

Before committing this codebase to GitHub and preparing it for production deployment, a comprehensive audit of the entire repository was conducted across 25 distinct dimensions—including data ingestion, catalog normalization, observation grid creation, feature engineering, model inference, calibration, reporting, Streamlit UI, dependencies, security, and repository hygiene.

The core machine learning engine (Exp6) and its mathematical integrity are certified, empirical validation holds (+0.48% bias on 10-day client summary; +2.52% bias on continuous daily series; 100% win rate across walk-forward windows; 0% future data leakage across all 74 causal features). However, several significant infrastructure, configuration, dependency, and maintainability issues were uncovered that must be resolved to ensure the repository is robust, reproducible, and ready for deployment.

---

## Audit Findings Matrix

| ID | Category | Severity | File / Component | Summary | Fix Required | Model Impact |
| :---: | :---: | :---: | :--- | :--- | :---: | :---: |
| **SEC-01** | Dependencies | **CRITICAL** | `requirements.txt` | `lightgbm` missing from declared dependencies | Add `lightgbm>=4.0.0` | None |
| **SEC-02** | Git Hygiene | **CRITICAL** | `.gitignore` | `reports/` wildcard ignores client deliverables | Whitelist client Excel deliverables | None |
| **CFG-01** | Configuration | **HIGH** | `config/settings.py` | Configuration out of sync with Exp6 production | Update to 674 SKUs, Sep 2026 dates, Exp6 params | None |
| **DOC-01** | Documentation | **HIGH** | `README.md` | Entire README describes legacy Phase 1 (588 SKUs, Aug 2026) | Overhaul README to describe Exp6 production | None |
| **ARC-01** | Maintainability | **HIGH** | `src/` path handling | Fragile cwd-relative paths in ingestion and runners | Anchor paths to `BASE_DIR` | None |
| **DEP-01** | Dependencies | **MEDIUM** | `requirements.txt` | Missing `reportlab`, `pypdf`, and `pyarrow` | Add missing runtime dependencies | None |
| **PKG-01** | Architecture | **MEDIUM** | `src/__init__.py` | Exports deprecated Phase 1 heuristic functions | Update package exports to production routines | None |
| **SEC-03** | Security | **MEDIUM** | `models/production_lgbm_model.pkl` | Model loaded via pickle without integrity verification | Add SHA-256 verification and security docs | None |
| **HYG-01** | Cleanliness | **MEDIUM** | Root directory | `scratch_prompt.txt` untracked in repository root | Move to `scratch/` or ignore | None |
| **LEG-01** | Legacy Code | **MEDIUM** | `src/` (13 legacy scripts) | Deprecated Phase 1 scripts remain in active `src/` | Clearly classify legacy vs active production | None |
| **LOG-01** | Logging | **LOW** | `src/` scripts | Print statements used instead of standard logging | Introduce structured logging facade | None |
| **TST-01** | Test Coverage | **LOW** | `tests/` | Test suite missing direct checks on db tables and schema | Expand test suite to 12+ comprehensive tests | None |

---

## Detailed Findings & Remediations

### 1. CRITICAL FINDINGS

#### [CRIT-01] Missing `lightgbm` in `requirements.txt`
- **File**: [`requirements.txt`](file:///c:/Users/bhave/Desktop/ml_project/requirements.txt#L1-L8)
- **Problem**: `requirements.txt` declares `pandas`, `numpy`, `streamlit`, `plotly`, `xlsxwriter`, `openpyxl`, and `scikit-learn`, but completely omits `lightgbm`.
- **Why it is a problem**: LightGBM is the certified machine learning engine at the heart of this entire project. Any new engineer or deployment container running `pip install -r requirements.txt` will immediately fail with `ModuleNotFoundError: No module named 'lightgbm'`.
- **Business Impact**: Production deployment failure; zero automated forecast capability in fresh environments.
- **Technical Impact**: Broken dependency chain; inability to execute `generate_client_reports.py` or `final_production_system.py`.
- **Recommended Fix**: Add `lightgbm>=4.0.0` to `requirements.txt`.
- **Model Behavior Changed**: No.
- **Regression Testing Required**: Yes (verify import and inference).

#### [CRIT-02] `.gitignore` Wildcard Suppresses Client Deliverables in `reports/`
- **File**: [`.gitignore`](file:///c:/Users/bhave/Desktop/ml_project/.gitignore#L23-L25)
- **Problem**: Line 24 specifies `reports/`, which causes git to completely ignore the `reports` directory.
- **Why it is a problem**: The primary deliverables required by stakeholders—[`reports/Rimmel_Validation_Sep01_Sep10_2026.xlsx`](file:///c:/Users/bhave/Desktop/ml_project/reports/Rimmel_Validation_Sep01_Sep10_2026.xlsx) and [`reports/Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx`](file:///c:/Users/bhave/Desktop/ml_project/reports/Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx)—will not be tracked by git. A fresh `git clone` will lack these reports, causing `tests/test_production_system.py` to fail on checkout.
- **Business Impact**: Client deliverables missing from repository; client reports invisible to remote team members.
- **Technical Impact**: Test suite failure on clean checkout (`test_backward_compatibility_copies` and `test_report_*`).
- **Recommended Fix**: Update `.gitignore` to allow official production Excel workbooks and CSV summaries in `reports/` while ignoring temporary scratch files:
  ```gitignore
  # Ignore temporary reports, but keep official production deliverables
  reports/*
  !reports/Rimmel_Validation_Sep01_Sep10_2026.xlsx
  !reports/Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx
  !reports/validation_report_sep_01_to_10_2026.xlsx
  !reports/production_forecast_sep_11_to_20_2026.xlsx
  !reports/*.csv
  ```
- **Model Behavior Changed**: No.
- **Regression Testing Required**: Yes (verify git status and test suite execution).

---

### 2. HIGH FINDINGS

#### [HIGH-01] `config/settings.py` Out of Sync with Production Exp6 Reality
- **File**: [`config/settings.py`](file:///c:/Users/bhave/Desktop/ml_project/config/settings.py#L18-L43)
- **Problem**: `config/settings.py` hardcodes deprecated Phase 1 parameters:
  - `CATALOG_SKU_COUNT = 588` (Actual certified catalog is 674 Canonical SKUs)
  - `DEFAULT_PROD_CUTOFF = '2026-07-31'` (Actual production cutoff is `2026-09-10`)
  - `HOLDOUT_TRAIN_END = '2026-07-20'` (Actual validation holdout train end is `2026-08-31`)
  - `HOLDOUT_EVAL_START = '2026-07-21'` (Actual validation start is `2026-09-01`)
  - `HOLDOUT_EVAL_END = '2026-07-31'` (Actual validation end is `2026-09-10`)
  - `DEFAULT_FORECAST_HORIZON_DAYS = 11` (Actual forward forecast horizon is 10 days, Sep 11–20)
- **Why it is a problem**: Any developer or pipeline importing constants from `config.settings` will execute against the wrong temporal boundaries and outdated SKU catalog count.
- **Business Impact**: Risk of running pipeline on old dates; confusion regarding product count (588 vs 674).
- **Technical Impact**: Inconsistent configuration layer across modules.
- **Recommended Fix**: Update `config/settings.py` to define the certified production constants:
  - `CATALOG_SKU_COUNT = 674`
  - `PRODUCTION_TRAIN_START = '2025-08-01'`
  - `PRODUCTION_TRAIN_END = '2026-09-10'`
  - `VALIDATION_START = '2026-09-01'`
  - `VALIDATION_END = '2026-09-10'`
  - `FORWARD_FORECAST_START = '2026-09-11'`
  - `FORWARD_FORECAST_END = '2026-09-20'`
  - `FORECAST_HORIZON_DAYS = 10`
  - `CALIBRATION_ALPHA = 0.10`
  - `CALIBRATION_BETA = 0.10`
  - `LGBM_PARAMS = {...}`
- **Model Behavior Changed**: No.
- **Regression Testing Required**: Yes (verify all tests pass).

#### [HIGH-02] Outdated `README.md` Documenting Obsolete Phase 1 System
- **File**: [`README.md`](file:///c:/Users/bhave/Desktop/ml_project/README.md#L1-L153)
- **Problem**: `README.md` documents the old August 2026, 588-SKU heuristic blend (Baseline + Momentum + Adaptive Blend), claiming the system forecasts July 21–31 and August 2026. It mentions zero details about Exp6, ZERO Treatment, LightGBM, the 74 causal features, or September 2026 operations.
- **Why it is a problem**: A developer, client, or technical interviewer inspecting the repo will read documentation that completely contradicts the actual production codebase.
- **Business Impact**: Client confusion, lack of technical transparency, damaged credibility.
- **Technical Impact**: Complete misalignment between user documentation and actual implementation.
- **Recommended Fix**: Overhaul `README.md` to reflect the certified Exp6 production architecture, 674 Canonical SKUs, LightGBM model, zero data leakage, and reproduction instructions.
- **Model Behavior Changed**: No.
- **Regression Testing Required**: No.

#### [HIGH-03] Fragile Cwd-Relative Paths in `src/` Modules
- **Files**:
  - `src/final_production_system.py`: `db_path = os.path.abspath('data/rimmel_clean.db')`
  - `src/final_production_runner.py`: `db_path = os.path.abspath('data/rimmel_clean.db')`
  - `src/final_production_delivery.py`: `db_path = os.path.abspath('data/rimmel_clean.db')`
  - `src/data_ingestion.py`: `RAW_SOURCE_FILE = os.path.join('data', '...')`
- **Problem**: Using `os.path.abspath('data/...')` without anchoring to `__file__` assumes the Python process was launched with the project root as `os.getcwd()`. If a script is run from a subfolder, an IDE test runner, or an external caller, it will fail with `FileNotFoundError`.
- **Why it is a problem**: Fragile execution; causes failures in automated CI/CD runners and external scripts.
- **Business Impact**: Pipeline failures in non-standard execution contexts.
- **Technical Impact**: Path resolution errors across scripts.
- **Recommended Fix**: Anchor all module paths using:
  ```python
  BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  DB_PATH = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')
  ```
- **Model Behavior Changed**: No.
- **Regression Testing Required**: Yes.

---

### 3. MEDIUM FINDINGS

#### [MED-01] Missing Runtime & Documentation Dependencies in `requirements.txt`
- **File**: [`requirements.txt`](file:///c:/Users/bhave/Desktop/ml_project/requirements.txt)
- **Problem**: Missing `reportlab` (used to compile `PROJECT_COMPLETE_TECHNICAL_AND_BUSINESS_GUIDE.pdf`), `pypdf` (used to verify PDF pages and integrity), and `pyarrow` (used for `.parquet` serialization in `data/processed/`).
- **Why it is a problem**: Anyone attempting to regenerate documentation or use parquet caching will encounter missing module errors.
- **Recommended Fix**: Update `requirements.txt` with locked/tested version bounds for `lightgbm`, `reportlab`, `pypdf`, and `pyarrow`.
- **Model Behavior Changed**: No.
- **Regression Testing Required**: Yes.

#### [MED-02] `src/__init__.py` Exposes Legacy Phase 1 Heuristics
- **File**: [`src/__init__.py`](file:///c:/Users/bhave/Desktop/ml_project/src/__init__.py#L1-L17)
- **Problem**: The package `__init__.py` exports functions from `baseline_forecast`, `momentum_forecast`, `adaptive_forecast`, `dynamic_engine`, etc., which are deprecated Phase 1 heuristic scripts.
- **Why it is a problem**: Misleads developers into believing `from src import ...` runs the production model.
- **Recommended Fix**: Update `src/__init__.py` to provide a clean package namespace highlighting the certified production system and data pipeline routines.
- **Model Behavior Changed**: No.
- **Regression Testing Required**: Yes.

#### [MED-03] Serialized Model Security (Pickle Deserialization)
- **File**: [`models/production_lgbm_model.pkl`](file:///c:/Users/bhave/Desktop/ml_project/models/production_lgbm_model.pkl)
- **Problem**: The trained model artifact is serialized with `pickle`. Unpickling arbitrary files in Python poses security risks if an untrusted binary is substituted.
- **Why it is a problem**: Standard security audit requirement for machine learning repositories before public hosting.
- **Recommended Fix**:
  1. Record the SHA-256 hash of `production_lgbm_model.pkl` in `production_model_config.json`.
  2. Implement an optional integrity check on model load in production runners.
  3. Document the security model in `README.md` and `CODEBASE_ARCHITECTURE.md`.
- **Model Behavior Changed**: No.
- **Regression Testing Required**: Yes.

#### [MED-04] Repository Root Hygiene: Untracked Scratch Files
- **File**: `scratch_prompt.txt`
- **Problem**: A 17.8 KB scratch prompt file resides in the root directory.
- **Why it is a problem**: Clutters the root of the GitHub repository with developer notes.
- **Recommended Fix**: Move to `scratch/` or add to `.gitignore`.
- **Model Behavior Changed**: No.
- **Regression Testing Required**: No.

#### [MED-05] Redundant Production Pipeline Scripts
- **Files**:
  - `src/final_production_runner.py` (Prompt 2 iteration)
  - `src/final_production_delivery.py` (Prompt 3 iteration)
  - `src/final_production_system.py` (Prompt 4 certified pipeline)
  - `src/generate_client_reports.py` (Prompt 8 client-facing 10-day SKU summary generator)
- **Problem**: Having multiple runners with similar names can confuse a new developer.
- **Recommended Fix**: Document the exact role of each script in `CODEBASE_ARCHITECTURE.md`, designating `src/final_production_system.py` as the full model retraining/certification pipeline and `src/generate_client_reports.py` as the daily/operational reporting engine.
- **Model Behavior Changed**: No.
- **Regression Testing Required**: No.

---

### 4. LOW & INFORMATIONAL FINDINGS

#### [LOW-01] Streamlit UI Redundant Data Loading Error Handling
- **File**: [`app.py`](file:///c:/Users/bhave/Desktop/ml_project/app.py#L120-L163)
- **Finding**: When cache files in `data/processed/` are missing, `app.py` falls back to empty DataFrames and displays helpful instructions. This is great behavior, but could be enhanced with an explicit check prompting the user to run `python src/generate_client_reports.py`.
- **Recommended Fix**: Add a friendly informational alert in `app.py` when caches are missing.

#### [LOW-02] Test Suite Enhancement
- **File**: [`tests/`](file:///c:/Users/bhave/Desktop/ml_project/tests/)
- **Finding**: The existing test suite (`test_production_system.py` and `test_feature_leakage.py`) passes 10/10 tests in ~7 seconds, covering hyperparameters, feature lists, Excel structures, actual unit sums, and feature leakage. Adding tests for configuration constants and database schema will make the test harness even more thorough.
- **Recommended Fix**: Add regression assertions for `config.settings` and database table row counts.

---

## Golden Behavior Baseline (Pre-Refactoring Snapshot)

To ensure **0% model drift** and **100% mathematical preservation**, the following empirical metrics represent the immutable source of truth:

| Dimension | Golden Metric Value | Source of Truth |
| :--- | :---: | :--- |
| **Model Engine** | `LGBMRegressor` | `models/production_lgbm_model.pkl` |
| **Model Hyperparameters** | `n_estimators=150, max_depth=6, num_leaves=31, lr=0.05, seed=42` | `models/production_model_config.json` |
| **Causal Feature Count** | Exactly **74 features** | `models/production_features.json` |
| **Catalog Canonical SKUs** | Exactly **674 SKUs** | `sku_master` in `data/rimmel_clean.db` |
| **Validation Period** | September 1–10, 2026 (10 Calendar Days) | `ml_features_zero` holdout partition |
| **Validation Actual Units** | **2,069.0 physical units** | `raw_transactions` ground truth |
| **Continuous Model Predicted**| **2,121.2 units** (+2.52% net bias, 90.54% WAPE) | Holdout benchmark evaluation |
| **Client 10-Day SKU Summary** | **2,079 units** (+10 units / +0.48% net variance) | `Rimmel_Validation_Sep01_Sep10_2026.xlsx` |
| **Forward Forecast (Sep 11–20)**| **1,934 units** (Amazon: 976, eBay: 948, Website: 7, Other: 3) | `Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx` |
| **Shared Warehouse Pool** | Single shared central warehouse pool per SKU; never summed across channels | Central warehouse physical constraint |
| **Feature Leakage** | Strictly $t < T$; 0% future data leakage | `tests/test_feature_leakage.py` |
| **Automated Unit Tests** | 10/10 passing tests in 7.18s | `python -m unittest discover tests` |

---

## Action Plan for Remediation

1. **Fix `requirements.txt`**: Add `lightgbm`, `reportlab`, `pypdf`, `pyarrow`.
2. **Fix `.gitignore`**: Whitelist official client reports, ignore scratch files and temporary test artifacts.
3. **Update `config/settings.py`**: Align all constants with certified Exp6 parameters.
4. **Anchor Module Paths**: Ensure all modules use `BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))`.
5. **Clean Root Directory**: Move `scratch_prompt.txt` to `scratch/` or ignore it.
6. **Update `src/__init__.py`**: Modernize package exports.
7. **Create `CODEBASE_ARCHITECTURE.md`**: Provide an architectural blueprint for onboarding engineers.
8. **Rewrite `README.md`**: Provide a complete, modern, GitHub-ready project manual.
9. **Create `DEPLOYMENT_CHECKLIST.md`**: Provide a pre-deployment operational checklist.
10. **Expand & Verify Test Suite**: Ensure 100% of unit tests pass and golden metrics are preserved with zero deviation.
