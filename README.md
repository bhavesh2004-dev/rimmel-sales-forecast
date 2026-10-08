# Rimmel Multi-Platform Demand Forecasting & Inventory Planning System

[![Production Certified](https://img.shields.io/badge/Production%20Status-Certified%20Exp6-brightgreen.svg)]()
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![Model Engine](https://img.shields.io/badge/Model-LightGBM%20Regressor-orange.svg)]()
[![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passing-success.svg)]()
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()

A production-grade, mathematically disciplined machine learning platform designed to forecast multi-channel demand, mitigate stockout feedback loops, and optimize central warehouse procurement across **674 cosmetics products (Rimmel catalog)**.

---

## 📖 Key Documentation

For deep technical dives, mathematical proofs, and architectural blueprints, refer to:

* 📘 **[Complete Technical & Business Knowledge Manual (34 Sections)](docs/CLIENT_FORECASTING_AND_BUSINESS_GUIDE.md)**  
  *(Comprehensive 129 KB manual covering business context, ZERO vs AVERAGE proof, feature dictionary, single SKU trace, and 20 client FAQs. Also available as an executive PDF: [`docs/CLIENT_FORECASTING_AND_BUSINESS_GUIDE.pdf`](docs/CLIENT_FORECASTING_AND_BUSINESS_GUIDE.pdf))*
* 🏗️ **[Codebase Structure & File Audit](docs/CODEBASE_STRUCTURE_AND_FILE_AUDIT.md)**  
  *(Comprehensive repository inventory, production dependency map, and cleanup blueprint)*
* 📋 **[QA Audit & Correction Reports](docs/qa/)**  
  *(Official 63-test QA audit, correction report, and demand dynamics analysis)*
* 🚀 **[Deployment Checklist](docs/DEPLOYMENT_CHECKLIST.md)**  
  *(Pre-commit verification gates, test commands, and production sign-off)*

---

## 🎯 1. Business Problem & Core Objectives

In multi-channel e-commerce retail, beauty and cosmetics products are sold simultaneously across multiple digital storefronts:
1. **Amazon UK** (Merchant-Fulfilled Network via Mayah Beauty and Bellas Beauty)
2. **eBay UK** (Merchant-Fulfilled Network via Mayah Beauty and Bellas Beauty)
3. **Direct-to-Consumer Website** (GLAMBEAUTY Web Store)
4. **Other Wholesale & Social Channels** (TikTok Shop legacy, UFK Retail store samples)

### The Central Shared Warehouse Constraint
Crucially, **all four selling channels draw physical stock from ONE central shared warehouse pool**. If an eBay customer buys the last 5 tubes of mascara, an Amazon customer cannot buy them ten minutes later. 

```
                                  ┌─────────────────────────────┐
                                  │   CENTRAL SHARED WAREHOUSE  │
                                  │   (Single Physical Stock)   │
                                  └──────────────┬──────────────┘
                                                 │
                      ┌──────────────────────────┼──────────────────────────┐
                      ▼                          ▼                          ▼
               ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
               │   Amazon    │            │    eBay     │            │ Website /   │
               │ Storefronts │            │ Storefronts │            │ Other (B2B) │
               └─────────────┘            └─────────────┘            └─────────────┘
```

> **CORE GOVERNANCE RULE**: Physical inventory is held in a single shared warehouse pool per SKU. **NEVER sum inventory across platforms**.

### The Failure of Naive Averages
Traditional retail methods rely on simple historical moving averages. In intermittent e-commerce, simple averages fail catastrophically:
1. **The Censored Demand Trap**: An average treats a day with 0 sales due to a complete warehouse stockout as "zero customer demand," creating a vicious cycle of structural under-ordering.
2. **The Intermittent Bias Trap**: For slow-moving items selling 1 unit every 15 days, imputing positive average sales on non-transaction days creates massive phantom demand (+1,178% WAPE).

---

## 🏆 2. Certified Production System (Exp6 Architecture)

The certified production engine (`Exp6`) is the culmination of extensive scientific experimentation:

```
    LONG-TERM BASE DEMAND (v365, v180, sales_days_90)
              +
    RECENT RUN-RATE & MOMENTUM (v7, v14, v30, v14_vs_v30)
              +
    CHANNEL-SPECIFIC TELEMETRY (Amazon Sessions, Buy Box %, eBay Promos)
              +
    CENTRAL SHARED INVENTORY SIGNALS (current_stock, days_since_stockout)
              │
              ▼
    LIGHTGBM REGRESSOR (n_estimators=150, max_depth=6, num_leaves=31, lr=0.05)
              │
              ▼
    POST-HOC COMBINED CALIBRATION (alpha=0.10 zero-suppression, beta=0.10 stockout-dampening)
              │
              ▼
    10-DAY PHYSICAL AGGREGATION (Sum continuous daily predictions BEFORE whole-unit rounding)
```

### Empirical Performance Benchmarks
Evaluated on the strictly unseen **September 1–10, 2026** holdout validation window:

| Benchmark Metric | Empirical Production Result | Operational Significance |
| :--- | :---: | :--- |
| **Catalog Scope** | **674 Canonical SKUs** | Complete active product catalog |
| **Holdout Actual Units** | **2,069.0 physical units** | Ground truth customer purchases |
| **Continuous Model Predicted** | **2,121.2 units** | **+2.52% net catalog bias** (0.1326 MAE, 0.5847 RMSE) |
| **Client 10-Day SKU Summary** | **2,079 units** | **+10 units / +0.48% net variance** |
| **Forward Forecast (Sep 11–20)** | **1,934 units** | Amazon: 976 (50.5%), eBay: 948 (49.0%), Web: 7, Other: 3 |
| **Walk-Forward Validation** | **100% win rate** | Outperformed baseline across all 4 rolling test windows |
| **Future Data Leakage** | **0.00%** | All 74 features strictly bounded at $t < T$ |

---

## 🏗️ 3. Repository Directory Structure

```text
ml_project/
│
├── app.py                         # Streamlit Interactive Web Application (Decoupled, Read-Only)
├── requirements.txt               # Locked Python dependencies (LightGBM, Streamlit, etc.)
├── .gitignore                     # Repository hygiene & client report preservation rules
├── README.md                      # Operational overview (this document)
│
├── src/                           # Production Forecasting & Normalization Package (7 Active Modules)
│   ├── __init__.py                # Package entrypoint exposing certified pipeline
│   ├── final_production_system.py # Certified Exp6 end-to-end retraining, forecast & report orchestrator
│   ├── build_sqlite_database.py   # SQLite database builder (rimmel_clean.db)
│   ├── data_ingestion.py          # Immutable raw Excel ingestion with SHA-256 lineage
│   ├── normalization.py           # Daily observation layer & operational state tagging
│   ├── platform_mapping.py        # Channel-to-platform normalization
│   ├── sku_mapping.py             # Raw SKU to Canonical SKU hierarchy resolution
│   └── phase2_feature_engineering.py # 74-feature causal engineering engine
│
├── config/
│   ├── __init__.py
│   └── settings.py                # Central production parameters, paths, and calibration constants
│
├── models/                        # Serialized Model Artifacts
│   ├── production_lgbm_model.pkl  # Trained LightGBM regressor (SHA-256 verified)
│   ├── production_features.json   # 74 causal feature definitions & ordering
│   └── production_model_config.json # Hyperparameters, metrics, and cryptographic hashes
│
├── reports/                       # Client Excel Deliverables
│   ├── Rimmel_Validation_Sep01_Sep10_2026.xlsx       # Primary Holdout Validation (8 Sheets)
│   ├── Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx # Primary Forward Forecast (4 Sheets)
│   ├── validation_report_sep_01_to_10_2026.xlsx      # Backward-compatible copy
│   ├── production_forecast_sep_11_to_20_2026.xlsx    # Backward-compatible copy
│   └── Rimmel_Dataset_and_Model_Explanation_Guide.pdf # Client explanation PDF guide
│
├── data/                          # Data Layer
│   ├── rimmel_clean.db            # Master clean SQLite database (101k raw rows, 573k grid rows)
│   ├── Rimmel Brand Sales Data - 1 Jan 2025 to 10 Sep 2026.xlsx # Immutable source data
│   ├── raw/                       # Raw ingest folder
│   └── processed/                 # Cached daily and SKU summaries for UI dashboard
│
├── tests/                         # Automated Regression Test Suite
│   ├── __init__.py
│   ├── test_production_system.py  # Model params, report schemas, unit sums, db integrity
│   └── test_feature_leakage.py    # Temporal bounds, lag matching, platform feature isolation
│
├── docs/                          # Consolidated System Documentation
│   ├── CLIENT_FORECASTING_AND_BUSINESS_GUIDE.md # Complete 34-section technical & business manual
│   ├── CLIENT_FORECASTING_AND_BUSINESS_GUIDE.pdf # Publication-grade executive PDF
│   ├── DEPLOYMENT_CHECKLIST.md                  # Pre-commit & production deployment gate checklist
│   ├── CODEBASE_STRUCTURE_AND_FILE_AUDIT.md     # Full repository inventory & dependency audit
│   ├── architecture/              # Internal algorithmic & pipeline architectural guides
│   ├── qa/                        # Official QA audit compliance reports & empirical evidence
│   └── archive/                   # Historical audit and git migration reports
│
├── experiments/                   # Isolated Research Experiments
│   └── rop_xgboost/               # Reorder Point (ROP) + TypeSafe Jev incremental value research
│
└── archive/                       # Preserved Historical Versions & Runners
    ├── phase_runners/             # Deprecated Phase 1–5 runners and early generators
    ├── raw_data_versions/         # Historical raw Excel dataset versions
    ├── qa_test_harness/           # 63-test QA test harness scripts and artifacts
    ├── legacy_scratch/            # Historical diagnostic scratchpads
    └── previous_versions/         # Early heuristic models & prototypes
```

---

## ⚡ 4. Quick-Start Guide

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/bhavesh2004-dev/rimmel-sales-forecast.git
cd rimmel-sales-forecast

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows PowerShell / CMD
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Automated Regression Tests
Execute the complete test suite to verify model integrity, feature bounds, report schemas, and actual unit sums:
```bash
python -m unittest discover tests
```
*Expected result: `Ran 12 tests in ~4.4s ... OK`.*

### Step 3: Launch Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore:
- **Product Inspector**: Historical sales curves, holdout validation, forward forecast lines, and platform share donuts for any individual SKU.
- **Data View**: Searchable, filterable tables of holdout validation and forward forecasts.
- **Forecast Overview**: High-level portfolio rollups, platform volume contributions, and top 10 SKUs.
- **Validation**: Retrospective holdout benchmark actual vs. predicted comparison.
- **Inventory / Planning**: Central shared warehouse stock, Days of Cover, and risk-filtered reorder tables.

### Step 4: Regenerate Official Client Excel Reports
To re-run inference and generate fresh Excel workbooks from SQLite:
```bash
python -m src.final_production_system
```

---

## 🔒 5. Security & Governance

1. **No External Network Dependencies**: Operates 100% offline using a local SQLite database (`data/rimmel_clean.db`).
2. **Zero Credentials Committed**: No API keys, cloud tokens, passwords, or personal credentials exist in the codebase.
3. **Model Cryptographic Checksum**: The binary `models/production_lgbm_model.pkl` is verified via SHA-256 (`821ba6acbea6f2a7cc81527810411389a64034c2f0c93a9636fab4d7f4605508`) during automated tests.
4. **Read-Only Dashboard**: The Streamlit user interface is strictly decoupled from the database and model weights, eliminating the risk of accidental model mutation or database corruption.

---

## 👥 Contributors & Contact

- **Lead ML Engineer**: Production Engineering Team
- **Project**: Rimmel Demand Forecasting & Multi-Platform Inventory Planning
- **Certified Release**: Version 1.0.0 (Exp6 Architecture) — September 2026
