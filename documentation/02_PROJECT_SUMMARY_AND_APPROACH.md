# Project Summary & Strategic ML Approach
## Rimmel Sales Forecasting Platform

### 1. Business Objectives
- Develop a multi-channel daily sales forecasting system for **588 Rimmel SKUs**.
- Generate 11-day forecasts (1 August to 11 August 2026) with 95% confidence intervals (Lower & Upper Bounds).
- Train exclusively on confirmed clean historical sales data: **1 August 2025 to 31 July 2026** (365 days).
- Deliver reports in client's requested format: `order sku`, `date`, `predicted unit`, `lower bound`, `upper bound`.

---

### 2. Multi-Channel Data Engineering Pipeline
- Multi-channel sales (Amazon AFN, Amazon MFN, eBay, Website) aggregated into a single clean `total_sales` metric per SKU per day.
- Median daily `selling_price` tracked with forward/backward fill handling missing records.
- High-performance SQLite database built at `data/rimmel_clean.db` for fast model training and SQL querying.

---

### 3. Production 3-Tier Architecture

#### **Tier A: Top 20 Bestsellers (58% of Total Volume)**
* **Dominant Features**:
  * `price_elasticity_ratio` ($\text{Selling Price} / \text{30d Avg Price}$): Normalizes July discount spikes back to ~1,046 target when price normalizes in August.
  * `lag_1`, `lag_3`, `lag_7`, `lag_14`, `roll_7_mean`, `roll_14_mean`, `roll_30_mean`: Bi-weekly velocity.
  * `aug2025_11d_baseline`: Same-period last year YoY seasonal anchor.
  * `annual_365d_mean`: 12-month annual run-rate baseline.
* **Engine**: Dedicated LightGBM Tweedie Engine with Price Normalization.

#### **Tier B: Next 100 Mid-Movers (35% of Total Volume)**
* **Dominant Features**:
  * `annual_365d_mean` & `aug2025_11d_baseline`: 12-month annual run-rate baseline anchor (boosted Tier B accuracy from 27.3% to 45.7%–54.5%).
  * `roll_90_mean`: 3-month quarterly seasonal trend.
  * `lag_7`, `lag_14`, `lag_28`: Multi-lag weekly re-order window.
  * `cat_code`: Category-level shared demand learning across cosmetics lines.
* **Engine**: Global Tweedie 90-Day Trend Engine with 1-Year Baseline.

#### **Tier C: Tail 394 Slow-Movers (7% of Total Volume)**
* **Dominant Features**:
  * Inter-arrival time $p_t$ + Order size $z_t$ + SBA Correction Factor $(1 - \alpha/2)$.
  * Zero threshold filter for dead/discontinued products ($<5$ sales/year).
* **Engine**: Croston Syntetos-Boylan Approximation (SBA) Intermittent Demand Engine.

---

### 4. Backtest Accuracy Summary

* **Monthly Volume Tracking Precision**: **99.9%**  
  *(June 2026: 32,843 predicted units vs 32,805 actual sales — off by only 38 units across the entire company!)*
* **11-Day Volume Precision**: **91.0% Average** (May: 94.4%, July: 92.9%).
* **Overall Catalog Accuracy**: **46.6% Average** across walk-forward backtests.
* **Tier A Bestseller Accuracy**: **54.0% Average** (58.6% Peak in June).
* **Tier B Mid-Mover Accuracy**: **45.7% Average** (54.5% Peak in July).

---

### 5. Client Deliverables & File Map

1. **Production Master Excel Report**: [`reports/Rimmel_3Tier_Production_Aug1_to_11_2026_Forecast.xlsx`](file:///c:/Users/bhave/Desktop/ml_project/reports/Rimmel_3Tier_Production_Aug1_to_11_2026_Forecast.xlsx)
2. **Production Forecast Excel Report**: [`reports/Rimmel_Forecast_Aug2026_FINAL.xlsx`](file:///c:/Users/bhave/Desktop/ml_project/reports/Rimmel_Forecast_Aug2026_FINAL.xlsx)
3. **Interactive HTML 7-Chart Dashboard**: [`reports/Rimmel_Dashboard.html`](file:///c:/Users/bhave/Desktop/ml_project/reports/Rimmel_Dashboard.html)
4. **Streamlit Web Dashboard App**: `new_approach/app.py`
5. **Clean Codebase**: `src/process_data.py`, `src/build_sql_database.py`, `src/train_master_3tier_pipeline.py`, `src/generate_dashboard.py`
