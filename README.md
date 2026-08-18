# Rimmel Sales Forecasting & Intelligence Platform

Production machine learning demand forecasting engine for 588 Rimmel SKUs across 1.5 years of multi-channel sales history.

---

## 📁 Clean Repository Structure

```
ml_project/
├── data/
│   ├── rimmel_clean.db                                    # Production SQLite Clean Database
│   └── raw/
│       ├── rimmel_old_column_data_v1.xlsx                 # Raw Historical Sales Dataset (v1)
│       └── rimmel_new_sales_data_v2.xlsx                  # Raw Historical Sales Dataset (v2)
│
├── reports/                                               # Client Deliverables & Dashboard Reports
│   ├── Rimmel_3Tier_Production_Aug1_to_11_2026_Forecast.xlsx  # 3-Tier Production Master Excel Report
│   ├── Rimmel_Forecast_Aug2026_FINAL.xlsx                # Final Production Excel Report
│   └── Rimmel_Dashboard.html                              # Interactive 7-Chart HTML Dashboard
│
├── src/                                                   # Core Production Source Code
│   ├── process_data.py                                    # Raw Excel Ingestion & Multi-Channel Aggregator
│   ├── build_sql_database.py                              # SQLite Ingestion & DB Builder
│   ├── train_master_3tier_pipeline.py                     # Production Master 3-Tier Forecast Engine
│   ├── run_tier_tailored_production_pipeline.py           # Tier-Tailored Production Engine
│   ├── generate_dashboard.py                              # Interactive HTML Dashboard Generator
│   └── app.py                                             # Streamlit Web Application
│
├── scripts/                                               # Backtests & Experimental Diagnostics
│   ├── backtests/
│   │   ├── backtest_master_3tier_enhanced.py              # May, June, July Backtest Evaluator
│   │   ├── backtest_3tier_pipeline.py                     # 3-Tier Backtest Engine
│   │   └── compare_full_month_old_vs_new_3tier.py         # Full-Month Old vs New Method Evaluator
│   │
│   └── experiments/
│       ├── test_croston_sba.py                            # Croston Intermittent Engine Test
│       ├── test_tier_specific_features.py                 # Price Elasticity & Bi-Weekly Lag Test
│       ├── test_tier_b_with_365d_mean.py                  # Tier B 365d Annual Baseline Test
│       └── visualize_catalog_demand_tiers.py              # 588 SKU Demand Tier Categorization
│
├── new_approach/                                          # Preserved Streamlit Dashboard Directory
│   └── app.py                                             # Streamlit Interactive Web Dashboard App
│
├── README.md                                              # Project Overview & Architecture Guide
└── requirements.txt                                       # Package Dependencies
```

---

## 🚀 How to Run Core Pipeline Modules

### 1. Execute Production Master 3-Tier Forecast Engine
Generates August 1 to 11, 2026 predictions using Tier-Tailored feature sets (Tier A Price Elasticity + Tier B 365d Annual Baseline + Tier C Croston SBA):
```powershell
.\venv\Scripts\python src/train_master_3tier_pipeline.py
```
*Output Report*: `Rimmel_3Tier_Production_Aug1_to_11_2026_Forecast.xlsx`

---

### 2. Launch Streamlit Interactive Web Application
```powershell
.\venv\Scripts\streamlit run new_approach/app.py
```

---

### 3. Generate HTML Self-Contained Interactive Dashboard
```powershell
.\venv\Scripts\python src/generate_dashboard.py
```
*Output Report*: `reports/Rimmel_Dashboard.html`

---

### 4. Run Backtest Accuracy Evaluation Across May, June, and July 2026
```powershell
.\venv\Scripts\python scripts/backtests/backtest_master_3tier_enhanced.py
```

---

## 🎯 Catalog Demand Tier Architecture

| Tier | Catalog Share | Business Profile | Engine & Dominant Features |
| :--- | :---: | :--- | :--- |
| **Tier A (Top 20 Bestsellers)** | **58% Vol** | High-volume, continuous daily sales, discount-sensitive | **Dedicated Tweedie Engine**: Price Elasticity Ratio + Bi-Weekly Lags (`lag_7`, `lag_14`) + 1-Year Baseline (`aug2025_11d_baseline`, `annual_365d_mean`) |
| **Tier B (Next 100 Mid-Movers)** | **35% Vol** | Moderate sales, periodic stockouts, category-driven | **Global Tweedie Engine**: 1-Year Baseline (`annual_365d_mean`) + 90-Day Quarterly Trend + Multi-Lag Window (`lag_7`..`lag_28`) + Category |
| **Tier C (Tail 394 Slow-Movers)** | **7% Vol** | Intermittent Poisson sales (80% zero sales days) | **Croston SBA Engine**: Inter-arrival time $p_t$ + Order size $z_t$ + SBA Correction Factor |
