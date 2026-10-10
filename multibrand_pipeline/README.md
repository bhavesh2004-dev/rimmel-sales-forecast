# Multi-Brand Demand Forecasting & Replenishment Pipeline (Production v2.0)

[![Production Certified](https://img.shields.io/badge/Production%20Status-Certified-brightgreen.svg)]()
[![Model Engine](https://img.shields.io/badge/Model-Shared%20LightGBM%20Regressor-orange.svg)]()
[![Database](https://img.shields.io/badge/Database-MySQL%208.0%2B-blue.svg)]()
[![Tests](https://img.shields.io/badge/Tests-52%2F52%20Passing-success.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()

Production-grade, mathematically disciplined multi-brand retail demand forecasting and inventory replenishment engine covering seven major beauty and personal care brands:
- **Rimmel** (767 SKUs)
- **Max Factor** (475 SKUs)
- **Kifra** (68 SKUs)
- **Weleda** (61 SKUs)
- **Delilah** (29 SKUs)
- **Geek & Gorgeous** (20 SKUs)
- **Frank Body** (13 SKUs)

**Total Scope:** **1,433 canonical product SKUs** sold across Amazon, eBay, Website, and Other retail platforms.

---

## 📁 Pipeline Architecture & Directory Structure

```text
multibrand_pipeline/
├── run_pipeline.py                 # Master end-to-end production pipeline entrypoint
├── run_dynamic_pipeline.py         # Dynamic cutoff discovery & holdout validation runner
├── config/                         # Configuration specifications
│   ├── pipeline_config.yaml        # Database connection & execution parameters
│   ├── dynamic_pipeline_config.yaml# Brand metadata, platform mappings, horizon settings
│   └── feature_schema.json         # Approved 60-feature definition schema
├── models/                         # Serialized ML artifacts
│   ├── global_lgbm_model.pkl       # Certified shared LightGBM regressor
│   └── model_metadata.json         # Model MD5 hash, feature list, and training metrics
├── reports/                        # Deliverables & scientific audit reports
│   ├── MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx # Official certified deliverable
│   ├── mysql_simplification_and_operational_import_report.md
│   └── ... (historical audit reports)
├── src/                            # Modular production source code
│   ├── db_manager.py               # MySQL connection & session management
│   ├── data_discovery.py           # Dynamic date cutoffs and SKU catalog discovery
│   ├── observation_grid.py         # Cartesian daily grid expansion & zero imputation
│   ├── features.py                 # 60 causal features calculation (zero future leakage)
│   ├── feature_profiles.py         # Feature tier router (short vs. long history)
│   ├── model_router.py             # Archetype classification & model selector
│   ├── forecast.py                 # LightGBM forward simulation engine
│   ├── inventory.py                # Shared warehouse pool & days of cover logic
│   ├── replenishment.py            # ROP replenishment calculations (LTD, MSL, PO logic)
│   ├── generate_combined_operational_excel.py # 21-column certified workbook generator
│   ├── simplify_and_import_operational_forecast.py # Table consolidation & MySQL importer
│   └── ...
└── tests/                          # Automated QA test suite (52 test cases)
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

## 🚀 Quality Testing (QA) Quick Start Guide

### Step 1: Environment Setup
Ensure Python 3.10+ is installed. Install all required dependencies from the project root:
```bash
pip install -r requirements.txt
```

### Step 2: Database Credentials Configuration
Copy the template `.env.example` to `.env` in the project root and fill in your MySQL credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=multibrand_forecasting_dev
MYSQL_USER=forecast_app
MYSQL_PASSWORD=your_mysql_password
```
> **Security Note**: `.env` is strictly excluded from version control via `.gitignore`. Never commit `.env` or credentials.

### Step 3: Database Schema Initialization (If Setting Up Fresh)
If initializing a fresh MySQL instance, execute the DDL script:
```bash
mysql -u forecast_app -p multibrand_forecasting_dev < schema_init.sql
```

### Step 4: Run the Automated QA Test Suite
Verify that all 52 unit, integration, and mathematical invariant tests pass:
```bash
python -m pytest multibrand_pipeline/tests -v
```
**Expected Result:** `52 passed in ~35s, 0 failures, 0 errors`.

### Step 5: Execute the Master Forecasting Pipeline
Run the unified 5-stage production pipeline:
```bash
python multibrand_pipeline/run_pipeline.py
```

---

## ⚙️ Master Pipeline Execution Stages (`run_pipeline.py`)

When you run `run_pipeline.py`, the following 5 stages execute sequentially:

1. **[STAGE 1/5] Database Preflight Check:**
   - Connects to MySQL `multibrand_forecasting_dev`.
   - Confirms `normalized_sales` row count (134,901 rows) and historical dates across all 7 brands.

2. **[STAGE 2/5] Forward Operational Forecasting & ROP:**
   - Discovers historical cutoff dates dynamically per brand.
   - Builds continuous Cartesian daily grids with zero-imputation.
   - Computes all 60 causal features in RAM (lags, rolling velocities, CV, in-stock velocities).
   - Simulates forward 10-day demand using the certified LightGBM model (`global_lgbm_model.pkl`).
   - Applies approved Exp6 calibration ($\alpha=0.10, \beta=0.10$).
   - Distributes demand across retail platforms (Amazon, eBay, Website, Other) adhering to the **Additive Channel Law**.
   - Executes retail Reorder Point (ROP) logic (Lead Time = 10 Days, Minimum Stock Level $\ge$ 6 units).

3. **[STAGE 3/5] Excel Deliverable Generation & Certification:**
   - Creates the official 21-column workbook: `reports/MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx`.
   - Audits all invariants (no nulls, no negative quantities, buffer invariant: `Stock + Replenish - LTD >= 6.0`).

4. **[STAGE 4/5] Canonical MySQL Database Persistence:**
   - Ensures canonical output table `operational_forecast_rop` exists.
   - Persists all 1,433 operational SKU records idempotently using `ON DUPLICATE KEY UPDATE`.

5. **[STAGE 5/5] Post-Import SQL Assertions:**
   - Runs automated SQL verification checking brand distributions, quantity totals, and constraints.
   - Confirms zero errors and displays execution summary.

---

## 📊 Output Schema Specifications (`operational_forecast_rop`)

The canonical output table `operational_forecast_rop` contains exactly 21 business columns plus metadata:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `brand` | `VARCHAR(50)` | Brand name (e.g., Rimmel, Max Factor, Weleda, etc.) |
| `sku` | `VARCHAR(100)` | Canonical unique Stock Keeping Unit identifier |
| `product_name` | `VARCHAR(255)` | Catalog product description |
| `current_stock` | `INT` | Current physical shared warehouse stock |
| `forecast_period` | `VARCHAR(50)` | Date range for 10-day forecast (e.g., `2026-09-11 to 2026-09-20`) |
| `amazon_predicted` | `INT` | Forecasted 10-day Amazon unit sales |
| `ebay_predicted` | `INT` | Forecasted 10-day eBay unit sales |
| `website_predicted` | `INT` | Forecasted 10-day Website unit sales |
| `other_predicted` | `INT` | Forecasted 10-day Other channels unit sales |
| `predicted_units` | `INT` | **Total 10-Day Forecast** (Amazon + eBay + Website + Other) |
| `avg_daily_usage` | `DECIMAL(10,2)` | Average daily usage rate (`predicted_units / 10`) |
| `lead_time_days` | `INT` | Vendor replenishment lead time (Standard = 10 days) |
| `lead_time_demand` | `INT` | Forecasted sales over lead time window |
| `target_stock` | `INT` | Inventory target (`lead_time_demand + safety_stock`) |
| `min_stock_level` | `INT` | Minimum Safety Level policy threshold ($\ge 6$ units) |
| `replenishment_quantity` | `INT` | Suggested Purchase Order quantity |
| `days_of_cover` | `DECIMAL(10,2)` | Days of stock remaining before stockout |
| `rop_status` | `VARCHAR(50)` | Action status: `REORDER NOW`, `MONITOR`, `HEALTHY`, etc. |
| `risk_level` | `VARCHAR(50)` | Stockout risk classification: `CRITICAL`, `MEDIUM`, `LOW` |
| `confidence` | `VARCHAR(50)` | Model confidence level: `HIGH`, `MEDIUM`, `LOW` |
| `reason` | `VARCHAR(255)` | Actionable operational explanation |
| `run_id` | `VARCHAR(100)` | Unique execution run identifier |
| `model_id` | `VARCHAR(100)` | Model artifact identifier |

---

## 🛡️ Core Architectural Rules & Governance Invariants

1. **Single Shared Warehouse Pool:** All selling channels draw inventory from ONE central physical warehouse per SKU. **Never sum inventory across platforms**.
2. **Additive Channel Law:** $\text{Amazon} + \text{eBay} + \text{Website} + \text{Other} = \text{10-Day Total Forecast}$.
3. **Strict Validation Temporal Isolation:** Features are strictly computed using data prior to forecast origin $t < T$. Zero future target leakage into lag or rolling windows.
4. **Idempotent Persistence:** Database persistence enforces `UNIQUE KEY uk_run_brand_sku (run_id, brand, sku)`. Re-running does not produce duplicate rows.
