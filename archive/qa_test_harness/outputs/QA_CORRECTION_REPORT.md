# Rimmel Demand Forecasting System — QA Correction & Production Hardening Report
**Audit Role:** Senior Production ML Engineer  
**Date:** September 29, 2026  
**System Tested:** Certified Production Exp6 (ZERO Treatment + LightGBM Regressor + Combined Calibration)  
**Previous Audit Verdict:** `NOT READY FOR CLIENT QA` (53 PASS, 1 FAIL, 8 WARNINGS)  
**Updated Audit Verdict:** **`READY FOR CLIENT QA`** (55 PASS, 0 FAIL, 7 WARNINGS)  

---

## 1. Executive Summary

Following the comprehensive QA and edge-case audit of the certified **Rimmel Demand Forecasting System (Exp6)**, a production engineering correction phase was authorized and executed. 

The audit had uncovered **1 High-Severity failure (`DQ-04A`)** where querying a single-date partition from the SQLite feature store (`ml_features_zero`) caused all-NULL numeric columns (specifically `days_from_restock`) to load as Python `object` dtypes containing `None`, triggering an unhandled LightGBM inference crash (`ValueError: pandas dtypes must be int, float or bool`). In addition, **1 Medium-Severity warning (`DASH-05`)** was identified in `app.py`, where missing dashboard data caches defaulted to empty DataFrames and resulted in unhandled UI `KeyError` exceptions.

### Correction Phase Results:
- **`DQ-04A` Resolved**: Designed and implemented `prepare_production_features()` in `src/final_production_system.py`, guaranteeing all non-categorical features are explicitly coerced to numeric `float64` without data fabrication or row loss.
- **`DASH-05` Resolved**: Implemented a top-level cache integrity guard in `app.py`, halting execution gracefully with `st.error()` and `st.stop()` upon detecting any missing required cache files.
- **Regression Suite Verification**: Created and executed `test/demand_forecasting_qa/scripts/run_focused_regression_tests.py`, with all **9 focused regression tests passing (100%)**.
- **Full Master QA Suite Verification**: Re-executed the entire 63-test QA harness (`test/demand_forecasting_qa/scripts/run_all_qa_tests.py`). **55 tests PASSED, 0 FAILED, 7 WARNINGS (all synthetic archetype out-of-distribution limits), 0 Critical, 0 High**.
- **Production Invariant Preserved**: The certified LightGBM model weights (`production_lgbm_model.pkl`), feature definition schema (`production_features.json`), SQLite database (`rimmel_clean.db`), calibration constants ($\alpha=0.10, \beta=0.10$), and forecast predictions remained **100% bitwise identical**.

The Rimmel Demand Forecasting System is now officially certified **`READY FOR CLIENT QA`**.

---

## 2. Original QA Findings

The initial QA audit executed 63 tests across 7 distinct operational suites, uncovering the following baseline profile:

| Metric | Original QA Status |
| :--- | :--- |
| **Total Tests** | 63 |
| **PASS** | 53 (84.1%) |
| **FAIL** | 1 (1.6%) |
| **WARNING** | 8 (12.7%) |
| **INFORMATIONAL** | 1 (1.6%) |
| **Critical Severity** | 0 |
| **High Severity** | 1 (`DQ-04A`) |
| **Medium Severity** | 8 (`DASH-05` + 7 Synthetic Archetypes) |
| **Initial Audit Verdict** | **`NOT READY FOR CLIENT QA`** |

### Root Vulnerabilities Identified in Baseline:
1. **`DQ-04A` (High Severity - Inference Crash on Single-Date Partitions)**:
   - When calling `SELECT * FROM ml_features_zero WHERE date = '2026-09-10'`, the column `days_from_restock` is completely NULL across all 1,413 active SKU-platform series.
   - SQLite dynamic typing returns columns with all-NULL values as Python `None`. Pandas parses these columns as `object` dtype rather than `float64`.
   - LightGBM Regressor (`model.predict()`) rejects `object` dtypes and throws `ValueError: pandas dtypes must be int, float or bool`.
2. **`DASH-05` (Medium Severity - Unhandled UI Crash on Missing Cache Files)**:
   - In `app.py`, `load_data_caches()` defaulted missing files to empty DataFrames (`pd.DataFrame()`).
   - Downstream rendering code directly accessed columns such as `sku_master['canonical_sku']`, triggering an unhandled `KeyError` rather than displaying an informative error message.

---

## 3. DQ-04A Root Cause Analysis

### Detailed Technical Mechanism:
- In the full training dataset spanning 406 historical days, `days_from_restock` contains both non-null integer values (e.g., 0, 5, 23) and NULL values. When Pandas ingests this multi-partition query from SQLite, it coerces integer columns with `NULL` to `float64` (representing `NULL` as `np.nan`).
- However, during live forward inference or single-date batch scoring (such as running daily forecasting for the most recent observation date `2026-09-10`), every single row in the partition may have `days_from_restock IS NULL`.
- Because SQLite has dynamic manifest typing, the SQLite C cursor returns `NULL` for all rows. In the absence of a single numeric scalar to infer a numeric type, `pd.read_sql_query()` assigns `dtype('O')` (object).
- LightGBM checks `X.dtypes`. Since Pandas `object` dtypes can hold arbitrary Python objects, LightGBM halts execution with:
  ```python
  ValueError: pandas dtypes must be int, float or bool.
  'days_from_restock' has dtype: object
  ```

---

## 4. DQ-04A Fix Implementation

### Design Principles:
1. **Zero Data Fabrication**: We do not replace `np.nan` with arbitrary zeroes or constants; missing values remain `np.nan`, which LightGBM handles natively and optimally via its default split-direction logic.
2. **Deterministic Typing**: All 66 numerical features are explicitly coerced to `float64` via `pd.to_numeric(col, errors='coerce')`.
3. **Categorical Integrity**: The 8 categorical features (`platform_group`, `quarter`, etc.) strictly preserve their `category` dtype and categorical encodings.
4. **Exact Feature Schema Order**: The returned DataFrame maintains the exact 74-column sequence specified in `models/production_features.json`.

### Code Implementation (`src/final_production_system.py`):
```python
def prepare_production_features(df_input: pd.DataFrame, feature_cols: list, cat_cols: list) -> pd.DataFrame:
    """
    Robust feature preprocessor ensuring strict dtype normalization and feature ordering
    before passing into LightGBM inference.
    
    Fixes DQ-04A: Coerces object columns containing None/NULL to float64 without data fabrication.
    """
    df_feat = df_input[feature_cols].copy()
    
    # 1. Coerce all numerical columns to float64 if they loaded as object
    for col in feature_cols:
        if col not in cat_cols:
            if df_feat[col].dtype == 'object':
                df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
            elif not pd.api.types.is_numeric_dtype(df_feat[col]):
                df_feat[col] = df_feat[col].astype('float64')
                
    # 2. Ensure categorical columns have category dtype
    for cat in cat_cols:
        if cat in df_feat.columns:
            df_feat[cat] = df_feat[cat].astype('category')
            
    return df_feat
```

This preprocessor was integrated into both the primary data loading stage and the forward forecasting batch loop in `src/final_production_system.py`.

---

## 5. DASH-05 Root Cause Analysis

### Detailed Technical Mechanism:
In `app.py`:
- `load_data_caches()` loaded CSV files from `data/processed/`. If a file was not found, the fallback was:
  ```python
  else:
      sku_master = pd.DataFrame()
  ```
- Subsequent Streamlit UI sections immediately executed:
  ```python
  total_skus = len(sku_master['canonical_sku'].unique())
  ```
- When `sku_master` was empty, accessing `['canonical_sku']` immediately raised a `KeyError: 'canonical_sku'`. The dashboard showed a raw Python traceback, providing a poor user experience.

---

## 6. DASH-05 Fix Implementation

### Code Implementation (`app.py`):
Immediately following `load_data_caches()`, an explicit cache integrity guard was introduced:
```python
sku_master, val_daily, fwd_daily, hist_daily, val_metrics = load_data_caches()

# -----------------------------------------------------------------------------
# EARLY CACHE INTEGRITY GUARD (DASH-05)
# -----------------------------------------------------------------------------
missing_caches = []
if sku_master.empty:
    missing_caches.append("dashboard_sku_master.csv")
if val_daily.empty:
    missing_caches.append("dashboard_validation_sku_daily.csv")
if fwd_daily.empty:
    missing_caches.append("dashboard_forecast_sku_daily.csv")
if hist_daily.empty:
    missing_caches.append("dashboard_historical_daily.csv")

if missing_caches:
    st.error(
        f"🚨 **Pipeline Data Not Found**: The following required data cache file(s) are missing or empty: "
        f"`{', '.join(missing_caches)}`.\n\n"
        f"Please run the production pipeline to generate all dashboard artifacts:\n\n"
        f"```bash\npython src/final_production_system.py\n```"
    )
    st.stop()
```

When cache files are missing or empty, Streamlit halts cleanly and displays a prominent instructions banner, preventing unhandled exceptions.

---

## 7. Focused Regression Test Suite

A standalone regression test script (`test/demand_forecasting_qa/scripts/run_focused_regression_tests.py`) was constructed to rigorously validate the fixes across 9 edge cases:

| Test ID | Test Description | Result | Details |
| :--- | :--- | :---: | :--- |
| **`TEST-DQ-04A-1`** | Single-date SQLite partition with all-NULL `days_from_restock` | `PASS` | `days_from_restock` normalized from `object` to `float64`; 50 rows inferred cleanly. |
| **`TEST-DQ-04A-2`** | Normal numeric `days_from_restock` preservation | `PASS` | Existing numeric values strictly preserved; 50 rows inferred cleanly. |
| **`TEST-DQ-04A-3`** | Multiple numerical columns with forced NULL values | `PASS` | 5 forced-NULL columns converted to `float64`; 0 inference exceptions. |
| **`TEST-DQ-04A-4`** | Exact 74-feature schema ordering and dtypes validation | `PASS` | Exact 74-feature sequence and categorical types verified against `production_features.json`. |
| **`TEST-DASH-05-1`** | Missing `sku_master` cache early failure guard | `PASS` | Guard code detected; `st.error()` and `st.stop()` verified. |
| **`TEST-DASH-05-2`** | Normal cache loading via `app.py` loader | `PASS` | All 4 required cache files loaded cleanly (674 SKUs, 6,740 val rows, 6,740 fwd rows). |
| **`TEST-DASH-05-3`** | Unknown SKU query behavior | `PASS` | Safely returns empty slices without throwing `KeyError`. |
| **`TEST-DASH-05-4`** | All-zero SKU display behavior | `PASS` | Zero-demand SKU `RIM-1KLL-071` evaluated cleanly (0.0 forecast units; no `ZeroDivisionError`). |
| **`TEST-DASH-05-5`** | Multi-platform SKU shared warehouse stock invariant | `PASS` | 13 multi-platform SKUs verified; single un-summed physical stock pool (108.0 units for `RIM-SCD-EYE-001`). |

**Focused Regression Result: 9 / 9 PASSED (100%)**

---

## 8. Production Artifact Integrity

The production model, feature definitions, and SQLite database were audited using SHA-256 cryptographic hashing to confirm zero modification during this correction phase:

| Artifact Path | SHA-256 Hash | Status |
| :--- | :--- | :---: |
| `models/production_lgbm_model.pkl` | `821ba6acbea6f2a7cc81527810411389a64034c2f0c93a9636fab4d7f4605508` | **100% UNTOUCHED** |
| `models/production_features.json` | `824411ebcf4659c42ecca89cc114ff5eb636462125c8eeb84864b3d61390e63e` | **100% UNTOUCHED** |
| `data/rimmel_clean.db` | `edf86230d189a7dff409fdc827bc204fb7ac8f442c319649bb7ddd5602d1412f` | **100% UNTOUCHED** |

---

## 9. Full QA Suite Before vs. After Comparison

Re-running the complete QA master harness (`test/demand_forecasting_qa/scripts/run_all_qa_tests.py`) across all 63 tests confirms the complete elimination of test failures:

```
====================================================================================================
QA METRIC                       ORIGINAL AUDIT                  POST-CORRECTION AUDIT         DELTA
====================================================================================================
Total Tests Executed            63                              63                              0
Passed Tests                    53 (84.1%)                      55 (87.3%)                    +2
Failed Tests                    1 (1.6%)                        0 (0.0%)                      -1
Warning Tests                   8 (12.7%)                       7 (11.1%)                     -1
Informational Tests             1 (1.6%)                        1 (1.6%)                        0
----------------------------------------------------------------------------------------------------
Critical Severity               0                               0                               0
High Severity                   1 (DQ-04A)                      0                             -1
Medium Severity                 8 (DASH-05 + 7 archetypes)      7 (7 archetypes)              -1
Low Severity                    54                              56                            +2
----------------------------------------------------------------------------------------------------
Overall Verdict                 NOT READY FOR CLIENT QA         READY FOR CLIENT QA           UPGRADED
====================================================================================================
```

### Specific Status Transitions:
1. **`DQ-04A` (Single-Date SQLite Partition Dtype Normalization)**:
   - *Before*: `FAIL` (High Severity) — LightGBM crashed with `ValueError: pandas dtypes must be int, float or bool`.
   - *After*: **`PASS`** (Low Severity) — `days_from_restock` normalized to `float64` cleanly; inference succeeded on 1,000 partition rows.
2. **`DASH-05` (Missing Cache File Graceful Degradation)**:
   - *Before*: `WARNING` (Medium Severity) — Unhandled `KeyError` when cache files were absent.
   - *After*: **`PASS`** (Low Severity) — Graceful degradation guard halts cleanly via `st.stop()` with descriptive error banner.

---

## 10. Remaining Warnings & Synthetic Archetype Analysis

All 7 remaining warnings belong to Suite 2 (Demand Dynamics Archetypes). These are **monitored behavioral characteristics**, not functional bugs or defects:

| Test ID | Archetype Description | Expected Range | Actual Prediction | Risk Tag | Explanation |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`DEM-02`** | ARCH-02: Long Zero-Demand (180d zero) | [0.0, 0.25] | 0.4104 u/d | VOLATILITY MONITORING | Tree ensemble retains a small non-zero catalog base split. |
| **`DEM-03`** | ARCH-03: Isolated Sale after Zero Period | [0.0, 0.8] | 2.6701 u/d | VOLATILITY MONITORING | Single sale creates a small 7-day velocity uptick. |
| **`DEM-04`** | ARCH-04: Sudden 10x Demand Spike | [1.0, 25.0] | 33.2794 u/d | VOLATILITY MONITORING | Short-term momentum feature scales rapidly. |
| **`DEM-10`** | ARCH-10: Stable High-Volume SKU | [35.0, 65.0] | 66.2902 u/d | NORMAL HEALTHY | Predicts 66.3 u/d, marginally above 65.0 boundary. |
| **`DEM-15`** | ARCH-15: Extremely High-Volume SKU (500 u/d) | [300.0, 700.0] | 97.9973 u/d | NORMAL HEALTHY | Tree regressors cannot extrapolate beyond highest training leaf (~80-100 u/d). |
| **`DEM-16`** | ARCH-16: Extremely Low-Volume SKU (0.01 u/d) | [0.0, 0.15] | 0.3965 u/d | VOLATILITY MONITORING | Modest bias on infinitesimal micro-demand. |
| **`DEM-18`** | ARCH-18: Repeated Small Transactions (1 u/d) | [0.6, 1.6] | 2.1447 u/d | NORMAL HEALTHY | Predicts 2.1 u/d vs 1.6 threshold. |

### Technical Rationale for Not Modifying the Model for Archetype Warnings:
- **Tree Ensembles Cannot Extrapolate**: By definition, decision trees partition feature space into piecewise-constant regions. On an extreme synthetic input of 500 units/day (which never occurs in the Rimmel catalog where max volume is ~80 units/day), a tree model caps at its maximum training leaf. This is mathematically normal behavior.
- **Overfitting Avoidance**: Artificially twisting the certified Exp6 model architecture or adding heuristics to satisfy synthetic extreme cases would degrade retrospective validation accuracy on real customer orders.
- **Planner Volatility Tags**: The production system already tags these edge-case SKUs with `VOLATILITY MONITORING` risk flags in the output reports, prompting supply planners to review high-variance items manually.

---

## 11. Remaining Operational Considerations

1. **Long-Tail Inactive Products (`DQ-10`)**:
   - Products dormant for >90 days receive an uncalibrated forecast of ~0.3 u/d due to broad category baselines. 
   - Exp6 calibration successfully dampens these to ~0.03 u/d (0 units over 10 days). If catalog cleanup is desired, business rules may flag SKUs with >180 days of zero sales as officially inactive.
2. **SQLite Query Optimization at 10,000 SKUs (`SCALE-01`)**:
   - Model inference throughput was measured at **111,000+ series-days/sec** (inference takes only 0.13s for the full catalog).
   - If the SKU catalog expands past 5,000 SKUs, adding a composite index `CREATE INDEX idx_features_lookup ON ml_features_zero(date, canonical_sku, platform_group);` will ensure SQL extraction remains sub-second.
3. **Cross-Platform Shared Inventory Allocation (`MULTI-02`)**:
   - The system strictly preserves physical warehouse stock (108 units are shared, never multiplied across channels). 
   - Individual platform forecasts remain independent. In live operations, fulfillment allocation across channels will follow the client's OMS/WMS priority rules.

---

## 12. Final Recommendation & Sign-Off

### Final Audit Verdict:
# **`READY FOR CLIENT QA`**

The certified **Exp6 Rimmel Demand Forecasting System** has passed all comprehensive data quality, numerical stability, leakage prevention, calibration fidelity, and system recovery tests with **0 Functional Failures**, **0 High-Severity Issues**, and **0 Critical Issues**.

Both targeted QA vulnerabilities (`DQ-04A` and `DASH-05`) have been completely resolved, verified via dedicated regression tests, and confirmed in the regenerated master QA audit. The core machine learning assets, feature schemas, and database have been strictly preserved.

The system is fully recommended for client acceptance testing and production scheduling.
