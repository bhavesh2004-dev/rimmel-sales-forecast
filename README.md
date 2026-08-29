# Rimmel 588-SKU Sales Forecasting & Inventory Decision Support Platform

A modular, explainable, and production-grade machine learning system designed to forecast multi-horizon demand, identify inventory risks, and optimize procurement for **588 cosmetics products**.

---

## 🏗️ 1. Project Architecture & Directory Structure

```text
ml_project/
│
├── app.py                         # Streamlit Web Application (Interactive UI & Dynamic Forecasting)
│
├── src/                           # Core Production Forecasting Package
│   ├── __init__.py                # Package entrypoint and exports
│   ├── data_loader.py             # Loads sales logs from master SQLite database
│   ├── preprocessing.py           # Temporal window slicing with zero data leakage
│   ├── feature_engineering.py     # Multi-horizon velocities, weekly CV, and price adjustments
│   ├── baseline_forecast.py       # Organic long-term demand anchor forecast
│   ├── momentum_forecast.py       # Fast-reacting recent trend & category momentum forecast
│   ├── adaptive_forecast.py       # Evidence-based dynamic adaptive blending engine
│   ├── confidence.py              # Evidence-based confidence classification (HIGH / MEDIUM / LOW)
│   ├── risk.py                    # Business risk & inventory status classification
│   ├── inventory_insights.py      # Days of inventory coverage & dead-stock liquidation matrix
│   ├── explanations.py            # Non-technical plain-English reasoning generator
│   ├── validation.py              # Protected holdout evaluation & error metric calculators
│   ├── report_generator.py        # Multi-sheet formatted Excel deliverable generator
│   └── dynamic_engine.py          # Unified pipeline facade orchestrator
│
├── config/
│   ├── __init__.py
│   └── settings.py                # Global database paths, protected dates, and business thresholds
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py           # Automated unit/integration test suite
│
├── reports/                       # Clean official Excel deliverables
├── archive/                       # Safely preserved legacy scripts & experiments
├── data/                          # Master clean SQLite database (rimmel_clean.db) & raw files
├── requirements.txt               # Locked dependencies
└── README.md                      # Documentation & operational guide
```

---

## 🔒 2. Forecasting Governance & Date Isolation

The platform maintains a strict separation between historical training data, internal evaluation gates, and forward production forecasts:

```text
HISTORICAL TRAINING DATA (1 Aug 2025 → 20 Jul 2026)
         │
         ▼
PROTECTED INTERNAL HOLDOUT (21 Jul 2026 → 31 Jul 2026)  <-- 100% Isolated Benchmark Gate
         │
         ▼
USER-SELECTED PRODUCTION FORECAST (e.g. 1 Aug 2026 → 31 Aug 2026)
```

1. **Model Training Window:** `1 August 2025 → 20 July 2026` (Strictly ceilinged for validation).
2. **Protected Internal Holdout Gate:** `21 July 2026 → 31 July 2026` (Protected benchmark; never accessed for model training or threshold tuning).
3. **Dynamic Production Horizon:** User-selectable future window (up to 31 days). The system automatically scales daily demand rates to the exact number of days selected.

---

## 🎯 3. Three-Forecast Methodology

For every product SKU, the system computes:

```
┌───────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
│ Forecast Model                        │ Methodology & Mathematical Purpose                                                  │
├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ 1. 🛡️ Baseline Forecast               │ • Long-term organic demand anchor ($v_{instock\_365d}$, $v_{180d}$, $v_{90d}$, $v_{30d}$).│
│                                       │ • Protects procurement against temporary flash promotions and erratic spikes.       │
├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ 2. ⚡ Momentum Forecast               │ • Fast-reacting recent run-rate ($v_{7d}$, $v_{14d}$, $v_{20d}$, CatMom, RelPrice). │
│                                       │ • Captures active summer surges and seasonal acceleration for core bestsellers.     │
├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ 3. 🎯 Recommended Forecast            │ • Final production planning number combining Baseline and Momentum ($w_{mom} \in    │
│    (Adaptive Blend)                   │   [0.15, 0.90]$) based on pre-cutoff stability and spike persistence signals.       │
└───────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ 4. Decision Intelligence & Inventory Insights

### Evidence-Based Confidence
* 🟢 **HIGH:** Sufficient historical volume ($>1000\text{ u/yr}$), low volatility ($CV \le 0.88$), high shelf availability ($>60\%$), and aligned Baseline/Momentum.
* 🟡 **MEDIUM:** Usable volume, moderate volatility, or active growth trends.
* 🔴 **LOW:** Intermittent demand ($>50\%$ zero-sales days), low volume ($<300\text{ u/yr}$), or high forecast disagreement.

### Inventory Health & Days of Inventory
* **Estimated Days of Inventory:** $\text{Current Stock} / \text{Recommended Daily Demand}$.
* **Health Categories:**
  * 🔴 **DEAD / STUCK:** Stock exists but demand is zero or near-zero for sustained periods (Liquidation candidate).
  * 🟠 **SLOW MOVING:** Inventory coverage exceeds 180 days relative to sales velocity (Overstock risk).
  * 🟡 **WATCH:** Demand is declining, volatile, or stock coverage is under 14 days (Reorder monitor).
  * 🟢 **HEALTHY:** Active turnover with balanced stock coverage.

---

## 🚀 5. How to Run the System

### 1. Installation
```bash
# Clone repository and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install locked dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Web Application
```bash
streamlit run app.py
```

### 3. Run Automated Tests
```bash
python -m unittest tests/test_pipeline.py
```

---

## 📊 6. Running Modes in the UI

1. **🚀 Mode 1: Dynamic Production Forecast**
   - Select any future date range via the date pickers (e.g. `01/08/2026 to 11/08/2026` or `01/08/2026 to 31/08/2026`).
   - View the 8-column executive table and download the formatted Excel deliverable.
2. **🧪 Mode 2: Protected Internal Holdout Validation (21–31 July 2026)**
   - Evaluates the frozen model against ground-truth actuals without generating future production forecasts.
   - Inspects WAPE, MAE, head-to-head winner breakdowns, and confidence calibration.
