# Rimmel Multi-Platform Demand Forecasting System
## Operational Guide: Client Reports & Interactive Dashboard

This guide provides instructions for generating the client-facing Excel reports and running the interactive Streamlit dashboard for the **certified Exp6 production forecasting system**.

---

## 1. System Architecture Overview

The forecasting system uses a single certified model architecture:
- **Model Engine**: LightGBM Regressor (`n_estimators=150, max_depth=6, num_leaves=31, learning_rate=0.05, random_state=42`)
- **Treatment**: ZERO Treatment (unobserved dates treated as true zero-sales)
- **Calibration**: Combined Exp6 Calibration ($\alpha=0.10$ for confirmed zero demand, $\beta=0.10$ for active stockouts)
- **Causal Feature Set**: 74 causal lag, rolling volume, momentum, calendar, and platform context features rolling backward from $T-1$
- **Channel Independence**: Amazon, eBay, Website, and Other demand streams are forecasted independently and aggregated to determine total physical demand.
- **Shared Warehouse Inventory**: Physical inventory is held in ONE central pool. **Inventory is NEVER summed across platforms**.

---

## 2. Generating Client Excel Reports

To generate the two official client-facing Excel workbooks and update the high-speed dashboard data caches, run:

```powershell
python -m src.generate_client_reports
```

### Generated Report 1: Retrospective Holdout Validation
- **Path**: `reports/validation_report_sep_01_to_10_2026.xlsx`
- **Period**: September 1–10, 2026 (unseen validation window before production retraining)
- **Grain**: Strictly **`Date × SKU`** (6,740 rows across 10 calendar days and 674 catalog SKUs)
- **Primary Sheet (Sheet 1)**: `Daily SKU Validation` (Exact 13 columns: `Date`, `SKU`, `Amazon Actual Units`, `Amazon Predicted Units`, `eBay Actual Units`, `eBay Predicted Units`, `Website Actual Units`, `Website Predicted Units`, `Other Actual Units`, `Other Predicted Units`, `Total Actual Units`, `Total Predicted Units`, `Reason`)
- **Executive Sheet (Sheet 2)**: `Summary` (Validation benchmark: 14,130 observations, 2,069 Actual Units, 2,121.2 Predicted Units, 90.54% WAPE, +2.52% Bias, 0.1326 MAE, 0.5847 RMSE, and platform breakdown)

### Generated Report 2: Forward Production Forecast
- **Path**: `reports/production_forecast_sep_11_to_20_2026.xlsx`
- **Period**: September 11–20, 2026 (forward 10 calendar days for inventory replenishment)
- **Grain**: Strictly **`Date × SKU`** (6,740 rows across 10 calendar days and 674 catalog SKUs)
- **Primary Sheet (Sheet 1)**: `Daily SKU Forecast` (Exact 13 columns matching validation format; all future `Actual Units` columns strictly blank / NULL)
- **Executive Sheet (Sheet 2)**: `Summary` (10-day projected physical units by channel, percentage share, model attribution, and shared inventory governance)

---

## 3. Running the Streamlit Dashboard

Launch the interactive dashboard with:

```powershell
streamlit run app.py
```

The application runs locally on `http://localhost:8501`.

### Dashboard Tabs Guide

1. **Tab 1 — Product Inspector (Core Analysis)**:
   - **SKU Search & Selector**: Inspect any of the 674 catalog SKUs.
   - **Metadata Cards**: Category, Parent ID, Shared Warehouse Stock, 10-Day Total Forecast, Risk, and Confidence.
   - **Interactive Time-Series Graph (Plotly)**: Visualizes historical daily actuals alongside platform-specific predictions. Clear visual shading separates the Historical, Validation (Sep 1–10), and Forward Forecast (Sep 11–20) periods. Toggle between 30-day, 60-day, 90-day, and Full History views.
   - **Platform Mix Donut Chart**: Dynamic percentage share of forecast demand across Amazon, eBay, Website, and Other, accompanied by automated summary insight.
   - **Channel Cards**: Immediate breakdown of expected units by channel and total demand.

2. **Tab 2 — Data View**:
   - Tabular exploration of daily data with dynamic filters for SKU, Platform, Date Range, and Observation State.

3. **Tab 3 — Forecast Overview**:
   - Executive portfolio cards: Total Forecast Units, Channel totals, and active product counts.
   - Channel breakdown bar chart, daily forward timeline, and Top 20 SKUs ranking.

4. **Tab 4 — Validation**:
   - Direct review of the official September 1–10 validation benchmark.
   - Actual vs. Predicted comparison chart and contextual guidance on intermittent demand.

5. **Tab 5 — Inventory / Planning**:
   - Decision-support view for procurement: Current Shared Stock, 10-day Forecast Demand, Days of Inventory Cover, and Action triggers.

---

## 4. Running Regression Tests

To verify that all models, Excel workbooks, schemas, and shared inventory rules meet certification requirements:

```powershell
python -m unittest discover tests
```

Expected result: 10/10 tests passing.
