# Rimmel 588-SKU Sales Forecasting & Inventory Decision Support Platform
## Master Technical Documentation & Complete Project Evolution (v1.0 → v8.1)

---

# 📑 TABLE OF CONTENTS
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Complete Version Progression & Evolution History](#2-complete-version-progression--evolution-history)
3. [Master Data Architecture & Database Schema](#3-master-data-architecture--database-schema)
4. [Mathematical Formulation & Feature Engineering](#4-mathematical-formulation--feature-engineering)
5. [Forecasting Architecture (Three-Forecast Engine)](#5-forecasting-architecture-three-forecast-engine)
6. [Decision Intelligence, Confidence & Risk Classification](#6-decision-intelligence-confidence--risk-classification)
7. [Inventory Health Matrix & Working Capital Optimization](#7-inventory-health-matrix--working-capital-optimization)
8. [Data-Grounded Reasoning Engine](#8-data-grounded-reasoning-engine)
9. [Holdout Validation Protocol vs. Dynamic Production Protocol](#9-holdout-validation-protocol-vs-dynamic-production-protocol)
10. [Modular Codebase Structure & File-by-File Guide](#10-modular-codebase-structure--file-by-file-guide)
11. [Verification, Testing & Operational Deployment Guide](#11-verification-testing--operational-deployment-guide)

---

# 1. Executive Summary & Problem Statement

### 1.1 Business Context
Rimmel London manages a master cosmetics catalog of **588 active SKUs** across high-velocity beauty categories including Mascaras, Eyeliners, Powder Compacts, Foundations, Lipsticks, and Blushes. Demand across this catalog exhibits complex retail patterns:
* **Powerhouse Bestsellers (Top 5% of SKUs):** Generate over 60% of total volume; experience seasonal summer surges and breakout growth.
* **Promotional & Volatile Lines (25% of SKUs):** Exhibit sharp, temporary sales spikes followed by immediate demand deceleration.
* **Intermittent Long-Tail Lines (70% of SKUs):** Characterized by sparse sales with high zero-sales days (>50%), where standard time-series models overforecast and create expensive excess inventory.
* **Historic Out-of-Stock Distortions:** Historical stockouts created artificial zero-sales days that artificially depressed historical run rates.

### 1.2 Strategic Mandate
Build an **honest, explainable, and inventory-useful machine learning forecasting system** that:
1. Replaces black-box guessing with transparent, mathematically rigorous signals.
2. Generates three distinct forecasts for every SKU: **Baseline (Organic Anchor)**, **Momentum (Recent Run-Rate)**, and **Recommended (Adaptive Blend)**.
3. Classifies evidence-based **Confidence** (`HIGH`, `MEDIUM`, `LOW`) and operational **Risk / Status**.
4. Computes **Estimated Days of Inventory** and quantifies **Trapped Working Capital** in dead/slow-moving inventory.
5. Provides a **Visual SKU Inspector** and formatted multi-sheet Excel deliverables.
6. Enforces strict zero-leakage holdout validation while enabling dynamic future date forecasting.

---

# 2. Complete Version Progression & Evolution History

The platform evolved through 8 distinct engineering phases:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 📅 PHASE 1 (v1.0): Initial 20-SKU Global Pilot                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Approach : Global LightGBM / XGBoost Regressors trained across top 20 SKUs.                              │
│ • Finding  : Scale distortion occurred—high-volume lines distorted error metrics; zero sales from stockouts│
│              were treated as zero demand, causing severe underforecasting upon restock.                    │
│ • Outcome  : Deprecated in favor of catalog-wide pattern modeling and out-of-stock isolation.               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 📅 PHASE 2 (v2.0): 1-Year Full History & Seasonal Anchors                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Approach : Ingested 1 full year of historical daily logs (Aug 2025 – Jul 2026); explored YoY August      │
│              anchors and monthly run-rates.                                                                 │
│ • Finding  : Static yearly averages lagged rapid summer momentum; promotional spikes distorted forecasts.   │
│ • Outcome  : Established the necessity of separating long-term organic demand from short-term momentum.     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 📅 PHASE 3 (v3.0): 3-Tier Segmentation (Tier A, Tier B, Tier C with Croston SBA)                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Approach : Segmented catalog into Tier A (>10k u/yr), Tier B (1.5k–10k u/yr), and Tier C (<1.5k u/yr)    │
│              using Croston Syntetos-Boylan Approximation (SBA) for intermittent lines.                       │
│ • Finding  : Static Croston was too sluggish for accelerating mid-tier lines and overforecasted true dead   │
│              stock.                                                                                         │
│ • Outcome  : Shifted to multi-horizon velocity weighting and evidence-based adaptive routing.               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 📅 PHASE 4 (v4.0): In-Stock Flag Cleansing & SQLite Master Ingestion                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Approach : Filtered out unrecorded/partial stock data; engineered `in_stock_flag = (current_stock > 0)`;  │
│              derived `velocity_instock_365d` to measure true organic sales velocity on in-stock days.       │
│ • Finding  : Eliminated stockout penalty; recovered true demand capacity for out-of-stock bestsellers.      │
│ • Outcome  : Built clean master SQLite database (`data/rimmel_clean.db` -> `full_history_v4`).              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 📅 PHASE 5 (v5.0): Dual-Forecast Benchmark (Baseline vs. Momentum)                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Approach : Tested two independent forecasts on locked July 21–31 holdout: Baseline (365d in-stock, 180d, │
│              90d, 30d) vs. Momentum (7d, 14d, 20d, CatMom, RelPrice).                                      │
│ • Finding  : Momentum won on high-volume sustained bestsellers (32.89% WAPE); Baseline won on volatile/promo │
│              products (91.38% vs 106.77% WAPE).                                                             │
│ • Outcome  : Proved that a single static model cannot fit all SKUs; dynamic routing is required.             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 📅 PHASE 6 (v6.0): Multi-Scale Adaptive Routing & Blending Engine                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Approach : Blended Baseline and Momentum ($w_{mom} \in [0.15, 0.90]$) using pre-cutoff stability signals │
│              (weekly CV, spike persistence index SPI, trend consistency).                                   │
│ • Finding  : Catalog WAPE improved to 41.38% without data leakage.                                          │
│ • Outcome  : Established core adaptive weighting engine.                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 📅 PHASE 7 (v7.0): Clean 8-Column Client Architecture & Inventory Matrix                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Approach : Formatted output into 8 clean executive columns; added Confidence (`HIGH`/`MEDIUM`/`LOW`),     │
│              Risk / Status, Days of Inventory, and Dead/Stuck working capital matrix ($357k+ trapped).      │
│ • Finding  : Executive adoption requires non-technical readability (5–15 second decision scanning).         │
│ • Outcome  : Built production report generator and interactive Streamlit UI.                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 📅 PHASE 8 (v8.0 & v8.1): Modularization, Dynamic Horizons & Visual Product Inspector                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Approach : Decomposed monolithic engine into 12 single-responsibility modules in `src/`; decoupled        │
│              production dates to support any user-selected window (up to 31 days) with automatic daily rate │
│              scaling; built Plotly Visual Inspector with 3 distinct forecast lines; added automated tests.  │
│ • Finding  : System achieved 100% test pass rate and verified holdout WAPE of 37.73% with zero leakage.     │
│ • Outcome  : Current Production Release.                                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# 3. Master Data Architecture & Database Schema

### 3.1 Database Specifications
* **Database File:** `data/rimmel_clean.db` (SQLite, 34.68 MB)
* **Master Table:** `full_history_v4`
* **Coverage:** All 588 SKUs $\times$ 365 daily logs (`2025-08-01` to `2026-07-31` = 214,620 rows).

### 3.2 Table Schema (`full_history_v4`)
```sql
CREATE TABLE full_history_v4 (
    sku             TEXT NOT NULL,
    date            DATE NOT NULL,
    total_sales     REAL NOT NULL DEFAULT 0.0,
    current_stock   INTEGER NOT NULL DEFAULT 0,
    in_stock_flag   INTEGER NOT NULL DEFAULT 1,
    selling_price   REAL NOT NULL DEFAULT 4.99,
    category        TEXT NOT NULL,
    PRIMARY KEY (sku, date)
);
```

### 3.3 Data Hygiene & Integrity Rules
1. **Shelf Stock Integrity:** `in_stock_flag = (current_stock > 0).astype(int)`.
2. **Missing Price Imputation:** Missing prices default to the 90-day SKU moving average or catalog default (\$4.99).
3. **No Synthetic Data:** No guessed restock dates or artificial promotion markers are injected.

---

# 4. Mathematical Formulation & Feature Engineering

All features are calculated strictly using data prior to the training cutoff ($T \le \text{train\_end}$):

### 4.1 Multi-Horizon Sales Velocities (Units / Day)
$$\begin{aligned}
v_{7d} &= \frac{1}{7} \sum_{t=T-6}^{T} \text{sales}_t \\
v_{14d} &= \frac{1}{14} \sum_{t=T-13}^{T} \text{sales}_t \\
v_{20d} &= \frac{1}{20} \sum_{t=T-19}^{T} \text{sales}_t \\
v_{30d} &= \frac{1}{30} \sum_{t=T-29}^{T} \text{sales}_t \\
v_{90d} &= \frac{1}{90} \sum_{t=T-89}^{T} \text{sales}_t \\
v_{180d} &= \frac{1}{180} \sum_{t=T-179}^{T} \text{sales}_t \\
v_{365d} &= \frac{\text{Total Annual Sales}}{\text{Total Observed Days}} \\
v_{instock\_365d} &= \frac{\sum_{t \in \text{InStock}} \text{sales}_t}{|\text{InStock Days}|} \quad (\text{if } |\text{InStock Days}| > 10 \text{ else } v_{365d})
\end{aligned}$$

### 4.2 Demand Volatility (Weekly Coefficient of Variation)
$$CV_{weekly} = \frac{\sigma_{\text{weekly sales}}}{\mu_{\text{weekly sales}}}$$
* $CV \le 0.85$: Low Volatility / Highly Stable
* $0.85 < CV \le 1.25$: Moderate Volatility
* $CV > 1.25$: High Volatility / Erratic Spikes

### 4.3 Category Macro Momentum
$$\text{CatMom} = \text{clip}\left(\frac{\text{Category } v_{14d}}{\text{Category } v_{60d}}, 0.85, 1.25\right)$$

### 4.4 Relative Price Adjustment
$$\text{RelPrice} = \frac{P_{\text{current}}}{\bar{P}_{90d}}$$
$$\text{PriceAdj} = \begin{cases}
\max(0.50, 1.0 - 1.5 \times (\text{RelPrice} - 1.0)) & \text{if } \text{RelPrice} > 1.04 \text{ (Price Increase)} \\
\min(1.30, 1.0 + 1.2 \times (1.0 - \text{RelPrice})) & \text{if } \text{RelPrice} < 0.96 \text{ (Price Discount)} \\
1.0 & \text{otherwise}
\end{cases}$$

### 4.5 Spike Persistence Index (SPI)
$$SPI = \frac{v_{7d} - v_{14d}}{\max(v_{30d}, 0.5)}$$
* $SPI < -0.50$: Sharp post-peak collapse.
* $SPI \ge 0.0$: Maintained or accelerating demand pace.

---

# 5. Forecasting Architecture (Three-Forecast Engine)

For any forward forecast horizon of $N$ days (e.g., $N = 11$ or $N = 31$):

```
                               ┌───────────────────────────┐
                               │ Preprocessed Sales Data   │
                               └─────────────┬─────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
        ┌───────────────────────────┐                 ┌───────────────────────────┐
        │ 🛡️ Baseline Organic Model  │                 │ ⚡ Momentum Trend Model    │
        │ Multi-Quarter Velocity    │                 │ Fast-Reacting Recent Pace │
        └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │ 🎯 Adaptive Blending      │
                               │ Evidence-Based $w_{mom}$  │
                               └─────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │ Physical Stock Clamp      │
                               │ $\min(\text{Pred}, \text{Stock})$ │
                               └───────────────────────────┘
```

### 5.1 Baseline Forecast (Organic Anchor)
Calculates long-term organic daily demand run rate:
$$d\_rate_{base} = \begin{cases}
0.45 \cdot v_{instock\_365} + 0.25 \cdot v_{180} + 0.20 \cdot v_{90} + 0.10 \cdot v_{30} & \text{if historic stockouts occurred} \\
0.35 \cdot v_{instock\_365} + 0.25 \cdot v_{180} + 0.25 \cdot v_{90} + 0.15 \cdot v_{30} & \text{otherwise}
\end{cases}$$
$$\text{Pred}_{base} = \min(\text{round}(d\_rate_{base} \times N), \text{Current Stock})$$

### 5.2 Momentum Forecast (Recent Trend Engine)
Calculates recent run-rate velocity:
$$d\_rate_{mom} = \begin{cases}
\max(v_{20}, v_{30}) \times \text{CatMom} \times 1.15 & \text{if Tier A Summer Surge } (\text{Sales} \ge 10\text{k}, v_{20}/v_{90} \ge 1.15) \\
v_{14} \times \text{PriceAdj} & \text{if Short-Term Acceleration } (v_{14} > 1.4 \cdot v_{30}) \\
(0.70 \cdot v_7 + 0.30 \cdot v_{14}) \times \text{PriceAdj} & \text{if Deceleration } (v_7 < 0.60 \cdot v_{14}) \\
(0.60 \cdot v_{14} + 0.40 \cdot v_{30}) \times \text{PriceAdj} & \text{if Stable Core } (CV \le 0.85, \text{Sales} \ge 500) \\
(0.50 \cdot v_{14} + 0.35 \cdot v_{30} + 0.15 \cdot v_{90}) \times \text{PriceAdj} & \text{otherwise (Standard Trend)}
\end{cases}$$
$$\text{Pred}_{mom} = \min(\text{round}(d\_rate_{mom} \times N), \text{Current Stock})$$

### 5.3 Adaptive Forecast (Recommended Planning Forecast)
Blends Baseline and Momentum using pre-cutoff evidence weights:
$$w_{mom} = \begin{cases}
0.15 & \text{if Flash Spike / Erratic Surge } (CV > 1.25 \text{ or } SPI < -0.50) \\
0.90 & \text{if Sustained Multi-Horizon Trend } (v_{30} > 1.1 \cdot v_{90}, v_{14} \ge 0.85 \cdot v_{30}, CV \le 1.15, \text{Sales} \ge 1.5\text{k}) \\
0.75 & \text{if Stable High-Volume Staple } (\text{Sales} \ge 1500, CV \le 0.85) \\
0.65 & \text{if Moderate Volume } (\text{Sales} \ge 1000) \\
0.25 & \text{if High Volatility / Promo Cooldown } (CV > 1.30, \text{Sales} \ge 150) \\
0.35 & \text{otherwise (Default Baseline Anchor)}
\end{cases}$$
$$\text{Pred}_{rec} = \min(\text{round}(w_{mom} \cdot \text{Pred}_{mom} + (1.0 - w_{mom}) \cdot \text{Pred}_{base}), \text{Current Stock})$$

---

# 6. Decision Intelligence, Confidence & Risk Classification

### 6.1 Evidence-Based Confidence Tiers
* 🟢 **HIGH CONFIDENCE:** Annual Sales $\ge 1,000$ u, $CV \le 0.88$, Model Agreement ($0.80 \le \frac{\text{Mom}}{\text{Base}} \le 1.25$), In-Stock Rate $\ge 60\%$.
* 🟡 **MEDIUM CONFIDENCE:** Annual Sales $\ge 300$ u, $CV \le 1.25$, Model Ratio $0.60 \le \frac{\text{Mom}}{\text{Base}} \le 1.60$.
* 🔴 **LOW CONFIDENCE:** Intermittent lines ($>50\%$ zero-sales days), volume $<300$ u, extreme volatility ($CV > 1.30$), or large model divergence.

### 6.2 Business Risk / Status Definitions
1. `HIGH DEMAND`: Annual volume $\ge 10,000$ units.
2. `DEMAND INCREASING`: Sustained upward trend or $v_{14d} > 1.25 \times v_{30d}$.
3. `DEMAND DECLINING`: Recent velocity $v_{14d} < 0.70 \times v_{30d}$ or $v_{30d} < 0.70 \times v_{90d}$.
4. `VOLATILE`: High weekly demand variance ($CV > 1.30$).
5. `STOCK RISK`: Current warehouse stock is 0 units, or forecast $> 80\%$ of stock on hand.
6. `LOW DEMAND`: Annual volume $< 150$ units.
7. `DEAD / NEAR-DEAD`: Zero sales across entire training history.
8. `LOW CONFIDENCE`: Insufficient or conflicting demand signals.
9. `NORMAL`: Steady, predictable operational demand.

---

# 7. Inventory Health Matrix & Working Capital Optimization

The platform calculates inventory longevity to help procurement liberate trapped working capital:

$$\text{Recommended Daily Demand} = \frac{\text{Recommended Forecast}}{N}$$
$$\text{Estimated Days of Inventory} = \frac{\text{Current Stock}}{\text{Recommended Daily Demand}}$$

```
┌───────────────────────────┬─────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Health Status             │ Qualification Criteria                                      │ Recommended Procurement Action                              │
├───────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 🔴 DEAD / STUCK           │ Annual Sales < 50 u or $v_{30d} < 0.10$ u/d with Stock > 0  │ Liquidation / clearance candidate to free up trapped capital.│
├───────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 🟠 SLOW MOVING            │ Days of Inventory > 180 days with Stock ≥ 200 units         │ Slow sales pace relative to stock; consider promo discount. │
├───────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 🟡 WATCH                  │ Status in [DECLINING, VOLATILE] or Days of Inventory < 14 d │ Monitor weekly sales trajectory; reorder buffer if low.     │
├───────────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 🟢 HEALTHY                │ Active turnover with balanced stock coverage                │ Maintain regular replenishment cycle.                       │
└───────────────────────────┴─────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

# 8. Data-Grounded Reasoning Engine

The `Reason` generator in `src/explanations.py` provides strictly data-backed explanations:

* **Volatile Surge:** *"Recent 14-day velocity ({v14} u/d) is sharply higher than the 90-day baseline ({v90} u/d), but high weekly volatility (CV {cv}) indicates an erratic spike; Adaptive forecast anchors 85% to Baseline."*
* **Sustained Growth:** *"Recent 14-day velocity ({v14} u/d) is consistently above the 90-day baseline ({v90} u/d) with steady demand; Adaptive forecast trusts Momentum (90% weight)."*
* **Post-Peak Cooldown:** *"Recent 7-day velocity ({v7} u/d) has decelerated below the 14-day pace ({v14} u/d); Adaptive forecast pulls back toward the long-term baseline."*
* **Stable Core Staple:** *"Stable demand across all horizons (14-day {v14} u/d vs 90-day {v90} u/d); Baseline and Momentum are closely aligned with HIGH confidence."*
* **Declining Velocity:** *"Recent 14-day velocity ({v14} u/d) has declined below the 90-day baseline ({v90} u/d); Adaptive forecast adapts downward to match recent run-rate."*
* **Stock Clamped:** *"... Constrained by warehouse stock on hand ({stock} units)."*
* **Insufficient Signal:** *"Historical data does not provide enough evidence to determine the cause; forecast is kept at baseline organic demand."*

---

# 9. Holdout Validation Protocol vs. Dynamic Production Protocol

```
┌─────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│ 🧪 1. PROTECTED INTERNAL HOLDOUT VALIDATION GATE            │ 🚀 2. DYNAMIC PRODUCTION FORECAST                           │
├─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ • Training Window  : 01 Aug 2025 → 20 Jul 2026              │ • Training Window  : 01 Aug 2025 → 31 Jul 2026 (Full Data)  │
│ • Evaluation Gate  : 21 Jul 2026 → 31 Jul 2026 (11 Days)    │ • Forecast Horizon : User-Selectable (e.g. 1 Aug → 31 Aug)  │
│ • Ground Truth     : 10,480 observed actual sales units     │ • Actual Sales     : Left blank for client actual entry     │
│ • Data Leakage     : ZERO (Strictly isolated benchmark)     │ • Purpose          : Live inventory procurement & planning  │
│ • Results          : Recommended WAPE = 37.73%              │ • Scaling          : Dynamically scales daily rates to N d  │
└─────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

# 10. Modular Codebase Structure & File-by-File Guide

```text
ml_project/
│
├── app.py                         # Streamlit UI (Holdout Gate & Dynamic Date Production Modes)
│
├── config/
│   ├── __init__.py                # Package export
│   └── settings.py                # Database paths, holdout constants, and threshold definitions
│
├── src/
│   ├── __init__.py                # Module exports
│   ├── data_loader.py             # Loads sales records from SQLite (full_history_v4)
│   ├── preprocessing.py           # Enforces strict temporal slicing with zero data leakage
│   ├── feature_engineering.py     # Calculates multi-horizon velocities, CV, and price signals
│   ├── baseline_forecast.py       # Organic long-term demand anchor model (scales to N days)
│   ├── momentum_forecast.py       # Fast-reacting trend & category momentum model (scales to N)
│   ├── adaptive_forecast.py       # Evidence-based adaptive blending engine (scales to N days)
│   ├── confidence.py              # Evidence-based confidence classification (HIGH/MEDIUM/LOW)
│   ├── risk.py                    # Business risk & inventory status classification
│   ├── inventory_insights.py      # Days of inventory coverage & trapped working capital matrix
│   ├── explanations.py            # Non-technical data-grounded reasoning generator
│   ├── validation.py              # Holdout benchmark evaluation & error metric diagnostics
│   ├── report_generator.py        # Multi-sheet formatted Excel deliverable generator
│   └── dynamic_engine.py          # Unified pipeline facade orchestrator
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py           # Automated test suite verifying accuracy and zero leakage
│
├── documentation/                 # Comprehensive documentation repository
│   ├── MASTER_PROJECT_DOCUMENTATION_AND_VERSION_HISTORY.md
│   ├── 01_PROJECT_FORECASTING_REBUILD_DOCUMENTATION.md
│   ├── 02_PROJECT_SUMMARY_AND_APPROACH.md
│   └── 03_SYSTEM_AUDIT_AND_LIVING_DOCUMENTATION.md
│
├── reports/                       # Clean official Excel deliverables
├── archive/                       # Preserved historical experiments (new_approach, legacy src)
├── data/                          # Master clean SQLite database & raw spreadsheets
├── requirements.txt               # Locked dependencies
├── .gitignore                     # Repository hygiene rules
└── README.md                      # Quick-start project overview
```

---

# 11. Verification, Testing & Operational Deployment Guide

### 11.1 Running Automated Unit Tests
```bash
python -m unittest discover -s tests
```
*Tests verify catalog count (588), zero data leakage, holdout WAPE parity (37.73%), dynamic horizon scaling (7d, 11d, 14d, 31d), and report generation.*

### 11.2 Launching Interactive Web Application
```bash
streamlit run app.py
```

### 11.3 Generating Client Excel Deliverables
Reports are automatically generated in `reports/`:
* `Rimmel_July_21_31_Actual_vs_Predicted_Report.xlsx`: Holdout evaluation with actuals.
* `Rimmel_August_1_11_Client_Forecast_Report.xlsx`: Forward forecast with blank actuals for client entry.
* `Rimmel_Inventory_Clearance_and_Overstock_Action_Plan.xlsx`: Dead stock recovery matrix.
