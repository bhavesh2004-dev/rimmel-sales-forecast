# Multi-Brand Demand Forecasting & Replenishment Platform

[![Production Certified](https://img.shields.io/badge/Production%20Status-Certified%20Production%20v2.0-brightgreen.svg)]()
[![Model Engine](https://img.shields.io/badge/Model-Shared%20LightGBM%20Regressor-orange.svg)]()
[![Database](https://img.shields.io/badge/Database-MySQL%208.0%2B-blue.svg)]()
[![Tests](https://img.shields.io/badge/Tests-52%2F52%20Passing-success.svg)]()
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()

A production-grade, mathematically disciplined machine learning platform designed to forecast multi-channel daily product demand, mitigate stockout feedback loops, and optimize central warehouse procurement across **1,433 product SKUs** spanning seven retail beauty and cosmetics brands:

- **Rimmel** (767 SKUs)
- **Max Factor** (475 SKUs)
- **Kifra** (68 SKUs)
- **Weleda** (61 SKUs)
- **Delilah** (29 SKUs)
- **Geek & Gorgeous** (20 SKUs)
- **Frank Body** (13 SKUs)

---

## 🏗️ Repository Structure

```text
rimmel-sales-forecast/
├── .env.example              # Template environment credentials file (safe for git)
├── .gitignore                # Git exclusions (strictly ignores .env, raw dumps, large archives)
├── README.md                 # Primary project overview and QA handover guide
├── requirements.txt          # Python production and testing dependencies
├── schema_init.sql           # MySQL DDL initialization script
├── app.py                    # Multi-brand interactive operational Streamlit UI
└── multibrand_pipeline/      # Certified 7-Brand Machine Learning Pipeline
    ├── README.md             # Detailed pipeline technical manual
    ├── run_pipeline.py       # Master end-to-end production runner (Stages 1-5)
    ├── run_dynamic_pipeline.py # Dynamic cutoff discovery & holdout validation runner
    ├── config/               # Configuration files & feature schemas
    │   ├── dynamic_pipeline_config.yaml
    │   ├── feature_schema.json
    │   └── pipeline_config.yaml
    ├── models/               # Certified LightGBM model weights & metadata
    │   ├── global_lgbm_model.pkl
    │   └── model_metadata.json
    ├── reports/              # Certified deliverable workbooks & reports
    │   ├── MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx
    │   └── ...
    ├── src/                  # Production pipeline source code
    │   ├── db_manager.py     # MySQL connection and pooling manager
    │   ├── data_discovery.py # Dynamic temporal cutoff discovery
    │   ├── observation_grid.py # Cartesian daily grid generation
    │   ├── features.py       # 60 causal feature calculation (zero leakage)
    │   ├── forecast.py       # LightGBM forward inference
    │   ├── inventory.py      # Shared warehouse stock pool logic
    │   ├── replenishment.py  # ROP replenishment calculations
    │   ├── generate_combined_operational_excel.py # 21-column Excel builder
    │   └── simplify_and_import_operational_forecast.py # MySQL persistence
    └── tests/                # Automated QA test suite (52 test cases)
        ├── test_causal_leakage.py
        ├── test_dynamic_date_discovery.py
        ├── test_feature_availability.py
        ├── test_forecast_horizon.py
        ├── test_forecast_recursion.py
        ├── test_inventory_calculations.py
        ├── test_mysql_output_schema.py
        ├── test_observation_logic.py
        ├── test_replenishment.py
        ├── test_report_schema.py
        ├── test_schema_parity.py
        ├── test_source_ingestion_reconciliation.py
        └── test_validation_leakage.py
```

---

## 🚀 Quality Testing (QA) Handover Guide

This repository has been prepared specifically for rigorous evaluation by the **Quality Testing (QA) Team**. Follow these steps to clone, configure, test, and run the pipeline:

### 1. Environment Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/bhavesh2004-dev/rimmel-sales-forecast.git
cd rimmel-sales-forecast
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Database Environment Credentials
Create a `.env` file in the repository root by copying `.env.example`:
```bash
cp .env.example .env
```
Provide your MySQL connection credentials in `.env`:
```ini
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=multibrand_forecasting_dev
MYSQL_USER=forecast_app
MYSQL_PASSWORD=your_secure_password
```
> **Security Guarantee**: The `.env` file and all credentials are automatically protected by `.gitignore` and are never committed or pushed to GitHub.

### 3. Initialize Database Schema (For Fresh Installations)
To create the canonical database tables, execute `schema_init.sql`:
```bash
mysql -u forecast_app -p multibrand_forecasting_dev < schema_init.sql
```

### 4. Execute the Automated Test Suite (52 Tests)
Run pytest across all 13 test suites:
```bash
python -m pytest multibrand_pipeline/tests -v
```
All **52 tests** should pass in ~35 seconds with 0 failures and 0 errors:
- **Causal Leakage**: Validates zero future lookahead (lag-1, shift-1).
- **Dynamic Discovery**: Validates automatic discovery of cutoff dates per brand.
- **Feature Parity**: Validates all 60 schema features across history tiers.
- **Inventory & Replenishment**: Validates single shared warehouse stock rules, Lead Time Demand, and Minimum Stock Level invariants.
- **MySQL Output Schema**: Validates single canonical output table `operational_forecast_rop` with 21 columns and unique constraints.

### 5. Execute the Master Pipeline End-to-End
Execute the full master pipeline:
```bash
python multibrand_pipeline/run_pipeline.py
```

Execution produces:
1. Console progress logs detailing each of the 5 pipeline stages.
2. The certified 21-column operational Excel deliverable at:
   `multibrand_pipeline/reports/MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx`
3. Idempotent persistence of 1,433 SKU records into MySQL table `operational_forecast_rop`.
4. Automated post-import SQL assertion verification.

---

## 🎯 Core Business Logic & Architectural Principles

### 1. Central Shared Warehouse Constraint
Physical inventory is held in **ONE single central warehouse pool per SKU** that fulfills orders across Amazon, eBay, Website, and Wholesale.
> **Rule**: Never sum inventory across platforms. `current_stock` is a SKU-level attribute.

### 2. Additive Channel Law
Multi-platform sales are forecasted individually and aggregate cleanly to the SKU total:
$$\text{Amazon Predicted} + \text{eBay Predicted} + \text{Website Predicted} + \text{Other Predicted} = \text{10-Day Total Forecast}$$

### 3. Strict Validation Temporal Isolation
Features are calculated strictly using data up to the forecast origin date $t < T$. No actual sales data from the forecast window leaks into lag or rolling windows.

### 4. Reorder Point (ROP) Replenishment Invariant
Replenishment calculations enforce:
$$\text{Lead Time Demand (LTD)} = \text{Forecasted 10-day demand}$$
$$\text{Target Stock} = \text{Lead Time Demand} + \text{Minimum Stock Level (MSL} \ge 6\text{)}$$
$$\text{Replenishment Quantity} = \max(0, \text{Target Stock} - \text{Current Stock})$$
$$\text{Projected Stock Buffer} = \text{Current Stock} + \text{Replenishment Quantity} - \text{LTD} \ge 6.0$$

---

## 📊 Canonical Database Architecture

The runtime database `multibrand_forecasting_dev` uses:
- **Input Table**: `normalized_sales` (134,901 rows) — Unified, normalized historical daily sales across all 7 brands.
- **Canonical Output Table**: `operational_forecast_rop` (1,433 rows) — Single authoritative table holding 21 business columns per SKU.
- All redundant legacy output tables (`forecast_results`, `report_forecast_inventory`, `report_rop`, etc.) have been permanently retired.

---

## 🔒 Confidentiality & Secret Hygiene

- All credentials, API keys, and database passwords are kept strictly local in `.env`.
- Database dumps (`database_backups/`), raw order files (`data/*.csv`, `data/*.xlsx`), and scratchpad files are excluded via `.gitignore`.
- This repository is ready for external QA review on the `main` branch.
