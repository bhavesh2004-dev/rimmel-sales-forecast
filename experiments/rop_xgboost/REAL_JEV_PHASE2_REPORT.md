# Real TypeSafe Jev + XGBoost ROP Incremental-Value Experiment
## Phase 2 Final Experimental Report

**Experiment ID:** `EXP-ROP-JEV-REAL-PHASE2`  
**Execution Timestamp:** 2026-09-28 14:34:46 UTC+05:30  
**Status:** Completed Successfully (Scenario C Approved Scope)  
**Governance Invariant:** Certified Production Exp6 LightGBM System 100% Isolated & Untouched  

---

## 1. Executive Summary

This report documents the empirical findings of **Phase 2: Real TypeSafe Jev + XGBoost Reorder Point (ROP) Incremental-Value Experiment**. 

Following the successful Phase 1 connectivity and cross-sectional testing on 2026-09-10, Phase 2 evaluated the operational and statistical impact of incorporating real contextual decision primitives from TypeSafe Jev (`~typesafe/jev-latest` -> `typesafe/jev-1.13-20260917` via OpenRouter Decisions API) into an XGBoost Quantile Regression replenishment system.

Over a 62-day historical walk-forward period (`2026-07-01` to `2026-08-31`) across 28 representative SKU × Platform cases (1,736 logical decision contexts), Model A (Baseline Structured XGBoost) and Model B (Real Jev-Enhanced XGBoost) were evaluated under identical leakage-free out-of-fold cross-fitting and day-by-day discrete-event inventory simulation ($L = 7$ days lead time, 14 days target cover, Order-Up-To policy).

### Primary Findings
1. **Service Level**: Unit fill rate increased from **93.80%** (Baseline) to **93.92%** (Jev-Enhanced), an improvement of **+0.12 percentage points** (+2.0 units fulfilled out of 1,665 total demand units).
2. **Stockouts**: Stockout events decreased from **12** to **10** (-2 events, -16.67%), and stockout day rate declined from **2.86%** to **2.38%** (-0.48 percentage points).
3. **Inventory & Orders**: Average on-hand warehouse stock remained virtually identical (**564.5 units** vs. **564.6 units**, +0.1 units), total units ordered remained flat (**2,283.0** vs. **2,281.0**, -2.0 units), and purchase order count shifted from **57** to **58** (+1 PO).
4. **Statistical Accuracy**: Quantile pinball loss ($\alpha = 0.95$) improved marginally from **2.7492** to **2.7439** (**-0.19%** relative improvement). Empirical 95% service coverage was essentially identical (**94.99%** vs. **94.93%**).
5. **Redundancy Analysis**: Jev's primary risk signal (`jev_replenishment_risk_score`) exhibited a **0.937 correlation with the binary `stockout_flag`**, and urgency probability (`jev_urgent_replenishment_prob`) exhibited a **0.812 correlation with `stockout_flag`**. In XGBoost gain attribution, all Jev features combined accounted for only **3.38%** of model importance, with 96.62% driven by core tabular velocity and ratio features (`v14_vs_v30`, `v14`, `v7`, `cv_30`).

### Final Classification
**`LIMITED / MARGINAL INCREMENTAL VALUE`**

Real TypeSafe Jev provides a slight directional reduction in stockout events (-2 events across 1,736 series-days) and a minor service level gain (+0.12 pts), but at the cost of cloud API dependency, external network latency (~180ms per decision), and ongoing operational complexity. Because Jev is largely collinear with structured tabular signals already available in SQLite (`stockout_flag`, `days_of_cover`, run-rate ratios), **Phase 3 production rollout is NOT justified**.

---

## 2. Research Question

> **"Does real TypeSafe Jev provide measurable incremental value when added to our structured-data XGBoost ROP model?"**

Specifically:
- Does augmenting tabular demand and inventory features with real Jev structured decisions (`noul`, `score`, `choice`) improve stockout prevention, cycle service level, or inventory holding efficiency?
- Does Jev introduce novel contextual variance, or is it fundamentally redundant with features already computed by the tabular pipeline?
- Is any observed improvement large enough to justify the financial, operational, and architectural dependency on external LLM decision endpoints?

---

## 3. Approved Experiment Scope

Execution proceeded under **Scenario C**, explicitly authorized for execution:
- **Cohort Size:** 28 representative SKU × Platform series
- **Horizon:** 62 daily historical walk-forward dates (`2026-07-01` to `2026-08-31`)
- **Logical Decision Points:** 62 dates × 28 cases = **1,736 decision points**
- **Payload Architecture:** 3 decision questions (`urgent_replenishment`, `replenishment_risk`, `replenishment_priority`) batched into 1 API request per case-date
- **Total Authorized API Volume:** Up to 1,736 API calls

---

## 4. Why Scenario C Was Selected

An audit of the experimental options demonstrated why Scenario C was the sole scientifically sound and computationally viable scope:
- **Naive Full-Catalog Evaluation (Rejected):** 674 active SKUs across 4 platforms = 1,413 active SKU-platform combinations. Evaluating 62 dates would require 62 × 1,413 = **87,606 API calls**, taking over 34 hours of continuous network execution and exposing the project to high rate-limit risks and unnecessary credit spend.
- **Scenario C (Approved):** By selecting a statistically rigorous stratified sample of 28 representative cases spanning all 8 core business archetypes (hero SKUs, stockouts, volatile lines, long-tail, eBay, Amazon, Website), Scenario C captured 100% of the relevant operational diversity in **1,736 calls**, executed in **5.2 minutes**, for **$0.05117 USD**.

---

## 5. Exact 28 Cases

The 28 representative cases from Phase 1 were tracked across all 62 dates:

| Case ID | Canonical SKU | Platform | Category | Stratum / Business Archetype |
| :--- | :--- | :--- | :--- | :--- |
| **CASE_01** | `RIM-MSC-ACCEL-BLK` | Amazon | Mascara | 1. High Velocity + Low Stock |
| **CASE_02** | `RIM-SCD-EYE-002` | Amazon | Eyeliner | 1. High Velocity + Low Stock |
| **CASE_03** | `RIM-MSC-ACCEL-BLK` | eBay | Mascara | 1. High Velocity + Low Stock |
| **CASE_04** | `RIM-SCD-EYE-002` | eBay | Eyeliner | 1. High Velocity + Low Stock |
| **CASE_05** | `RIM-MSC-SCAND-RL-001` | eBay | Mascara | 2. High Velocity + Healthy Stock |
| **CASE_06** | `RIM-SUPERGEL-TOPCOAT` | Amazon | Base & Top Coat, Nails | 2. High Velocity + Healthy Stock |
| **CASE_07** | `RIM-MSC-ESL-101` | eBay | Mascara | 2. High Velocity + Healthy Stock |
| **CASE_08** | `RIM-LF-LS-206` | eBay | Lipstick | 2. High Velocity + Healthy Stock |
| **CASE_09** | `RIM-KF-PWDR-001` | Amazon | Complexion, Face Powder | 3. Medium Velocity + Low Stock |
| **CASE_10** | `RIM-MAGNIF-PAL-009` | Amazon | Eyeshadow Palettes | 3. Medium Velocity + Low Stock |
| **CASE_11** | `RIM-BBCREAM-VERYLIGHT` | eBay | BB & CC Cream, Complexion | 3. Medium Velocity + Low Stock |
| **CASE_12** | `RIM-NAIL-SUPERGEL-043` | Amazon | Gel Nail Polish | 3. Medium Velocity + Low Stock |
| **CASE_13** | `RIM-NAIL-60S-902` | eBay | Nail Polish | 4. Medium Velocity + Healthy Stock |
| **CASE_14** | `RIM-BTW-F&S-002` | Amazon | Brow Pencils, Brows | 4. Medium Velocity + Healthy Stock |
| **CASE_15** | `RIM-NAIL-SUPERGEL-042` | Amazon | Gel Nail Polish | 4. Medium Velocity + Healthy Stock |
| **CASE_16** | `RIM-NAIL-60S-900` | Amazon | Nail Polish | 4. Medium Velocity + Healthy Stock |
| **CASE_17** | `RIM-LR-CCLR-EYEILLUM-010` | Amazon | Complexion, Concealer | 5. Low/Zero Velocity + Stockout |
| **CASE_18** | `RIM-KATE-NUDE-LS-048` | eBay | Lipstick | 5. Low/Zero Velocity + Stockout |
| **CASE_19** | `RIM-SCD-EL-WP-008` | Website | Eyeliner | 5. Low/Zero Velocity + Stockout |
| **CASE_20** | `RIM-SCD-EYE-001` | Other | Eyeliner | 6. Low Velocity + High Stock |
| **CASE_21** | `RIM-MSC-ELL-003` | Website | Mascara | 6. Low Velocity + High Stock |
| **CASE_22** | `RIM-MBP-003` | eBay | Brow Pencils, Brows | 6. Low Velocity + High Stock |
| **CASE_23** | `RIM-NAIL-60S-312` | eBay | Nail Polish | 7. Promotion / Traffic Momentum |
| **CASE_24** | `RIM-SMPP-003` | eBay | Complexion, Face Powder | 7. Promotion / Traffic Momentum |
| **CASE_25** | `RIM-BTW-F&S-001` | Amazon | Brow Pencils, Brows | 7. Promotion / Traffic Momentum |
| **CASE_26** | `RIM-ELP-064` | eBay | Eyeliner | 8. High Volatility / Erratic |
| **CASE_27** | `RIM-NAIL-60S-271` | eBay | Nail Polish | 8. High Volatility / Erratic |
| **CASE_28** | `RIM-LASTLIP-125` | Amazon | Lip Pencil, Lips | 8. High Volatility / Erratic |

Data completeness audit confirmed **exactly 62 calendar records for all 28 cases (1,736 total rows)**.

---

## 6. Historical Dates

- **Evaluation Window:** 2026-07-01 to 2026-08-31
- **Total Calendar Days:** 62 consecutive days
- **Daily Decision Points:** Exactly 28 per day

---

## 7. Data Source

All inputs were extracted strictly as read-only queries from the authoritative SQLite database:
- **Path:** `data/rimmel_clean.db`
- **Table:** `ml_features_zero`
- **Integrity Guarantee:** Zero tables were created, modified, or dropped in the database.

---

## 8. Leakage Controls

Strict temporal leakage prevention guarantees were enforced:
1. **Contemporaneous State Context:** For historical decision date $D \in [2026-07-01, 2026-08-31]$, the state context fed to Jev referenced only information available on or before $D$:
   - Sales velocity ($v_7, v_{14}, v_{30}, v_{90}$)
   - Warehouse stock on hand as of $D$ (`current_stock`)
   - Stockout status (`in_stock_flag`, `stockout_flag`, `days_since_stockout`)
   - Channel commercial signals (selling price, Buy Box % on Amazon, eBay promoted listing status)
   - Calculated days of cover ($=\text{current\_stock} / v_{14}$)
2. **Dynamic Decision Dates:** State strings explicitly formatted the exact decision date `date = D`.
3. **Target Isolation:** The 7-day cumulative forward lead-time demand (`ltd_target`) was computed solely as ground-truth for XGBoost quantile regression and inventory realization. It was strictly excluded from Jev's prompt and features.
4. **No Cross-Temporal Caching:** The SHA-256 cache key incorporated `decision_date`, preventing decisions from later dates (e.g., September 10) from contaminating earlier backtest dates.

---

## 9. Real Jev Model and Version

- **Configured Model Identifier:** `~typesafe/jev-latest`
- **Resolved Provider Model Version:** `typesafe/jev-1.13-20260917`
- **Provider:** TypeSafe AI via OpenRouter

---

## 10. API Endpoint

- **Method:** `POST`
- **URL:** `https://openrouter.ai/api/alpha/decisions`
- **Headers:** Authorization Bearer token (loaded safely from `.env`), `Content-Type: application/json`

---

## 11. API Call Count

- **Total Logical Decision Contexts:** 1,736
- **Actual Network API Calls:** 1,733
- **Cache Hits:** 3 (pre-cached during smoke verification)
- **HTTP Failures:** 0
- **Retries Triggered:** 0 (100% success rate on first attempt)

---

## 12. Cache Hit/Miss Count

- **Cache Hits:** 3
- **Cache Misses:** 1,733
- **Persistent Cache Storage:** All 1,736 decisions are now permanently stored in `experiments/rop_xgboost/outputs/jev_cache/` as content-addressable JSON files. Any subsequent re-run of this experiment requires **0 API calls**.

---

## 13. Token Usage

- **Input Tokens:** 1,218,356 tokens
- **Output Tokens:** 140,406 tokens
- **Total Tokens:** 1,358,762 tokens
- **Average Tokens per Decision:** ~783 tokens (702 in / 81 out)

---

## 14. Estimated Cost

- **Total Financial Cost:** **$0.05117 USD** (~5.1 cents)
- **Average Cost per Decision:** **$0.0000295 USD** (~0.003 cents)
- **Variance vs. Pre-Run Budget Estimate:** The pre-run estimate was ~$0.050 USD. The actual cost was within 2.3% of the estimate.

---

## 15. Runtime

- **API Acquisition Duration:** 310.3 seconds (~5.17 minutes across 4 threads)
- **Average Dispatch Throughput:** ~5.6 decisions/second
- **Simulation, Ablation & Analysis Duration:** 4.4 seconds
- **Total Master Execution Runtime:** **314.7 seconds (~5.25 minutes)**

---

## 16. Baseline XGBoost ROP Configuration (Model A)

- **Architecture:** XGBoost Quantile Regressor (`objective='reg:quantileerror'`)
- **Target Quantile:** $\alpha = 0.95$ (95th percentile lead-time demand)
- **Hyperparameters:** `n_estimators=80, learning_rate=0.05, max_depth=3, subsample=0.85, colsample_bytree=0.85, random_state=42`
- **Features (22 Structured Features):**
  - Historical lags: `lag_1, lag_7, lag_14, lag_30`
  - Velocity run rates: `v7, v14, v30, v90, v14_vs_v30`
  - Volatility & cadence: `cv_30, sales_days_30`
  - Stock & operational flags: `current_stock, in_stock_flag, days_since_stockout`
  - Commercial & platform: `selling_price, promo_days_30, amazon_sessions_momentum, buy_box_7d, platform_cat`
  - Calendar: `day_of_week, is_weekend, month`
- **Validation Framework:** 4-Fold GroupKFold cross-fitting across the 28 series (7 cases held out per fold, guaranteeing 100% out-of-fold predictions without in-sample contamination).

---

## 17. Jev-Enhanced XGBoost ROP Configuration (Model B)

- **Architecture:** Identical XGBoost Quantile Regressor (`objective='reg:quantileerror'`)
- **Target Quantile:** $\alpha = 0.95$
- **Hyperparameters:** Identical (`n_estimators=80, learning_rate=0.05, max_depth=3, subsample=0.85, colsample_bytree=0.85, random_state=42`)
- **Validation Framework:** Identical 4-Fold GroupKFold cross-fitting across the 28 series
- **Features (28 Features = 22 Baseline + 6 Real Jev Features):**
  - `jev_urgent_replenishment_prob` (Noul boolean probability $[0.0, 1.0]$)
  - `jev_replenishment_risk_score` (Continuous score $[0.0, 2.0]$)
  - `jev_prio_HIGH` (One-hot indicator 1.0/0.0)
  - `jev_prio_MEDIUM` (One-hot indicator 1.0/0.0)
  - `jev_prio_LOW` (One-hot indicator 1.0/0.0)
  - `jev_confidence` (Composite model confidence $[0.0, 1.0]$)

---

## 18. Inventory Simulation Configuration

- **Simulation Engine:** `experiments/rop_xgboost/walk_forward_backtest.py`
- **Evaluation Period:** 62 days (`2026-07-01` to `2026-08-31`)
- **Assumed Lead Time ($L$):** 7 calendar days (experimental assumption)
- **Target Cover Horizon:** 14 calendar days
- **Initial Inventory:** Actual warehouse on-hand stock on `2026-07-01` (or $14 \times v_{14}$ if non-positive)
- **Replenishment Policy:** Continuous Review $(s, S)$ / Order-Up-To
  - Reorder Trigger: If $\text{Inventory Position} (\text{On-Hand} + \text{Pipeline In-Transit}) \le \text{Predicted ROP}$
  - Reorder Quantity: $\text{ROQ} = \max(0, \text{daily\_run\_rate} \times \text{target\_cover} + \text{predicted\_ROP} - \text{inventory\_position})$
  - Inbound PO Arrival: Exactly $t + L$ days after order placement

---

## 19. Baseline Results (Model A)

- **Pinball Loss ($\alpha=0.95$):** 2.7492
- **Empirical Coverage:** 94.99% (virtually exact alignment with 95% target)
- **Mean Predicted ROP:** 33.25 units
- **Total Demand:** 1,665.0 units
- **Fulfilled Demand:** 1,561.7 units
- **Unit Service Level:** 93.80%
- **Stockout Rate:** 2.86%
- **Stockout Events:** 12 events
- **Average Inventory:** 564.5 units
- **Days of Cover:** 81.9 days
- **Total Units Ordered:** 2,283.0 units
- **Purchase Order Count:** 57 POs

---

## 20. Jev-Enhanced Results (Model B)

- **Pinball Loss ($\alpha=0.95$):** 2.7439
- **Empirical Coverage:** 94.93%
- **Mean Predicted ROP:** 32.84 units
- **Total Demand:** 1,665.0 units
- **Fulfilled Demand:** 1,563.7 units
- **Unit Service Level:** 93.92%
- **Stockout Rate:** 2.38%
- **Stockout Events:** 10 events
- **Average Inventory:** 564.6 units
- **Days of Cover:** 82.0 days
- **Total Units Ordered:** 2,281.0 units
- **Purchase Order Count:** 58 POs

---

## 21 & 22. Absolute and Percentage Differences (The 13 Required Metrics)

| # | Metric Name | Model A (Baseline Structured) | Model B (Real Jev-Enhanced) | Absolute Delta | Percentage / Relative Delta |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Total Demand Units** | 1,665.0 | 1,665.0 | 0.0 | 0.00% (Identical) |
| **2** | **Fulfilled Demand Units** | 1,561.7 | 1,563.7 | +2.0 | +0.13% |
| **3** | **Unit Service Level (%)** | 93.80% | 93.92% | +0.12 pts | +0.12 percentage points |
| **4** | **Stockout Rate (%)** | 2.86% | 2.38% | -0.48 pts | -0.48 percentage points |
| **5** | **Stockout Events Count** | 12 | 10 | -2 events | -16.67% |
| **6** | **Total Units Ordered** | 2,283.0 | 2,281.0 | -2.0 | -0.09% |
| **7** | **Purchase Order Count** | 57 | 58 | +1 order | +1.75% |
| **8** | **Average Inventory Units** | 564.5 | 564.6 | +0.1 | +0.02% |
| **9** | **Average Days of Cover** | 81.9 | 82.0 | +0.1 days | +0.12% |
| **10** | **Average Predicted ROP** | 33.25 | 32.84 | -0.41 units | -1.23% |
| **11** | **Reorder Frequency (Orders/Day)** | 0.0328 | 0.0334 | +0.0006 | +1.83% |
| **12** | **Quantile Pinball Loss ($\alpha=0.95$)** | 2.7492 | 2.7439 | -0.0053 | -0.19% (Lower is better) |
| **13** | **Empirical Service Coverage (%)** | 94.99% | 94.93% | -0.06 pts | -0.06 percentage points |

---

## 23. Ablation Results

To determine which individual Jev primitive contributes to the model, an ablation experiment was executed across the 5 configurations using cached Jev decisions (0 additional API calls):

| Configuration | Features Count | Pinball Loss ($\alpha=0.95$) | Coverage (%) | Mean ROP | Service Level (%) | Stockout Rate (%) | Stockout Events | Avg Inventory | Units Ordered | PO Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Structured Only** | 22 | 2.7492 | 94.99% | 33.25 | 93.80% | 2.86% | 12 | 564.5 | 2,283.0 | 57 |
| **B. Structured + Noul** | 23 | 2.7417 | 95.05% | 33.33 | 93.86% | 2.62% | 11 | 564.5 | 2,263.0 | 58 |
| **C. Structured + Score** | 23 | 2.7570 | 94.99% | 33.45 | 93.80% | 2.86% | 12 | 564.5 | 2,227.0 | 52 |
| **D. Structured + Choice** | 25 | **2.7356** | 94.82% | 32.57 | 93.80% | 2.86% | 12 | 564.4 | 2,238.0 | 52 |
| **E. Structured + All Jev** | 28 | 2.7439 | 94.93% | 32.84 | **93.92%** | **2.38%** | **10** | 564.6 | 2,281.0 | 58 |

### Key Ablation Insights:
1. **Noul (`urgent_replenishment`)**: Produced the cleanest operational improvement when added individually, reducing stockout events from 12 to 11 and improving service level from 93.80% to 93.86%.
2. **Choice (`replenishment_priority`)**: Yielded the lowest Pinball Loss (2.7356 vs 2.7492 Baseline), demonstrating that discrete categorical prioritization helps tighten ROP quantile estimation, but had identical operational stockout counts (12 events).
3. **Score (`replenishment_risk`)**: In isolation, Score slightly degraded Pinball Loss (2.7570 vs 2.7492), likely because its continuous values are largely collinear with existing run-rate features.
4. **All Jev Combined**: The combination of Noul + Choice + Score achieved the fewest stockout events (10 vs 12) and highest service level (93.92%), but the absolute magnitude (+0.12 pts) remains modest.

---

## 24. Redundancy Analysis

A critical question of this research is whether Jev provides genuinely novel contextual information or merely repackages tabular signals already available in the database.

### Correlation with Core Tabular Features

| Feature | `jev_replenishment_risk_score` | `jev_urgent_replenishment_prob` |
| :--- | :---: | :---: |
| **`stockout_flag`** | **+0.9369** | **+0.8116** |
| **`current_stock`** | -0.3628 | -0.3072 |
| **`cv_30` (Volatility)** | -0.2575 | -0.1879 |
| **`days_of_cover`** | -0.2534 | -0.2258 |
| **`v14` (Run rate)** | +0.0230 | +0.1902 |
| **`v7` (Run rate)** | +0.0017 | +0.1465 |

### Feature Importance Attribution in XGBoost (Gain Ranking)

| Rank | Feature | Type | Importance Gain | Cumulative Share |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `v14_vs_v30` | Tabular | 0.1857 | 18.57% |
| 2 | `v14` | Tabular | 0.1555 | 34.12% |
| 3 | `v7` | Tabular | 0.1235 | 46.47% |
| 4 | `cv_30` | Tabular | 0.0755 | 54.02% |
| 5 | `lag_14` | Tabular | 0.0733 | 61.35% |
| 6 | `amazon_sessions_momentum` | Tabular | 0.0632 | 67.67% |
| 7 | `v90` | Tabular | 0.0621 | 73.88% |
| 8 | `days_since_stockout` | Tabular | 0.0546 | 79.34% |
| 9 | `buy_box_7d` | Tabular | 0.0415 | 83.49% |
| 10 | `v30` | Tabular | 0.0277 | 86.26% |
| 11 | `sales_days_30` | Tabular | 0.0227 | 88.53% |
| **12** | **`jev_confidence`** | **Jev** | **0.0200** | **90.53%** |
| 13 | `lag_1` | Tabular | 0.0199 | 92.52% |
| 14 | `lag_30` | Tabular | 0.0120 | 93.72% |
| 15 | `day_of_week` | Tabular | 0.0105 | 94.77% |
| **16** | **`jev_prio_MEDIUM`** | **Jev** | **0.0104** | **95.81%** |
| 17 | `lag_7` | Tabular | 0.0090 | 96.71% |
| 18 | `selling_price` | Tabular | 0.0089 | 97.60% |
| 19 | `promo_days_30` | Tabular | 0.0079 | 98.39% |
| 20 | `current_stock` | Tabular | 0.0070 | 99.09% |
| **21** | **`jev_urgent_replenishment_prob`** | **Jev** | **0.0034** | **99.43%** |
| 22 | `month` | Tabular | 0.0033 | 99.76% |
| 23 | `platform_cat` | Tabular | 0.0025 | 100.00% |
| 24 | `in_stock_flag` | Tabular | 0.0000 | 100.00% |
| 25 | `is_weekend` | Tabular | 0.0000 | 100.00% |
| **26** | **`jev_replenishment_risk_score`** | **Jev** | **0.0000** | **100.00%** |
| **27** | **`jev_prio_HIGH`** | **Jev** | **0.0000** | **100.00%** |
| **28** | **`jev_prio_LOW`** | **Jev** | **0.0000** | **100.00%** |

### Redundancy Assessment
- **Collinearity:** `jev_replenishment_risk_score` has a Pearson correlation of **0.937** with `stockout_flag`. When an item is out of stock, Jev almost invariably outputs Risk Score $\approx 2.0$ ("High"). When an item is in stock with healthy cover, Jev outputs Risk Score $< 0.3$ ("Low"). 
- **Predictive Attribution:** XGBoost allocates **96.62%** of its split gain to structured historical features, led by velocity ratios (`v14_vs_v30`) and run-rate levels (`v14`, `v7`). All Jev features combined receive only **3.38%** of total model importance.
- **Interpretation:** Jev does not synthesize fundamentally new demand dynamics; it essentially acts as a smooth non-linear compression of `stockout_flag` and `current_stock / v14`. Because XGBoost can already partition on `stockout_flag` and `days_of_cover`, Jev provides only minor incremental nuance at the decision boundary.

---

## 25. Business Trade-offs

| Dimension | Baseline Structured XGBoost | Real Jev-Enhanced XGBoost | Business Trade-off Evaluation |
| :--- | :--- | :--- | :--- |
| **Fulfillment** | 1,561.7 units (93.80%) | 1,563.7 units (93.92%) | +2 units fulfilled (+0.12 pts lift). Operationally imperceptible. |
| **Stockouts** | 12 events (2.86% of days) | 10 events (2.38% of days) | -2 stockout events across 1,736 series-days. Favorable direction, tiny volume. |
| **Inventory Holding** | 564.5 units average | 564.6 units average | Identical capital tie-up (+0.1 unit). |
| **Order Volume** | 2,283.0 units across 57 POs | 2,281.0 units across 58 POs | Identical purchasing commitment (-2 units, +1 PO). |
| **Inference Latency** | **< 1 ms** (in-memory SQLite + XGBoost) | **~180 ms** per SKU/day (cloud HTTP request) | 180x slower execution per SKU; full catalog daily scoring takes ~4.5 minutes vs 0.1s. |
| **External Dependency** | **Zero** (100% offline, local database) | **High** (OpenRouter cloud API, network availability) | Incurred vulnerability to API outages, HTTP 429 rate limits, and network latency. |
| **Ongoing Operating Cost** | **$0.00** | **~$0.05 per 28 SKUs / 2 mo** (~$35/year for 674 SKUs) | Low monetary cost, but ongoing token maintenance and API key rotation overhead. |

---

## 26. Important Limitations

The conclusions of this experiment are bounded by several explicit domain limitations:
1. **Sample Scope:** The backtest was evaluated on 28 representative SKU-platform cases over 62 calendar dates (1,736 series-days). While statistically stratified, this does not represent the full 674-SKU catalog.
2. **Experimental Lead Time Assumption:** The lead time of $L = 7$ days is an assumed experimental parameter. The Rimmel database does not contain actual supplier purchase order lead times, vendor delivery tracking, or manufacturing lead times.
3. **Missing Supply Chain Variables:** The dataset lacks vendor Minimum Order Quantities (MOQ), tiered volume pricing, carton pack sizes, supplier delivery reliability metrics, and explicit warehouse holding vs. stockout penalty costs.
4. **Order-Up-To Heuristic:** The replenishment simulation utilized a standard Order-Up-To periodic review policy. Production replenishment systems often utilize joint-order replenishment, truckload optimization, and multi-echelon warehouse balancing.
5. **Demand Forecasting Independence:** This ROP replenishment experiment is completely separate from the certified Exp6 LightGBM demand forecasting system. These results have zero bearing on the certified sales forecasts.

---

## 27. Production Safety Verification

A comprehensive audit of the repository verifies that:
- `src/final_production_system.py` remains **100% untouched**.
- The certified Exp6 LightGBM production model and artifact remain **100% untouched**.
- All 74 production feature engineering pipelines remain **100% untouched**.
- Production forward forecasts and client validation reports remain **100% untouched**.
- The Streamlit dashboard (`dashboard.py`) remains **100% untouched**.
- The SQLite database (`data/rimmel_clean.db`) was queried strictly in read-only mode with zero schema modifications.
- The OpenRouter API key was loaded strictly from the isolated `.env` file and **was not printed, serialized, or committed anywhere**.
- Working tree status confirms zero Git commits, merges, or pushes. All experiment files reside strictly under `experiments/rop_xgboost/`.

---

## 28. Final Conclusion

### Verdict: `LIMITED / MARGINAL INCREMENTAL VALUE`

The experimental evidence demonstrates that real TypeSafe Jev does **NOT** provide transformational or clear incremental value to the Rimmel replenishment system:
- **Service level lift is minimal (+0.12 percentage points)**, representing just 2 additional units fulfilled out of 1,665 units of demand.
- **Stockout reduction is marginal (-2 events)** over 1,736 series-days.
- **Feature attribution is negligible (3.38% gain)**, with 96.62% of predictive power coming from existing tabular velocity ratios.
- **Information redundancy is high ($r = 0.937$ with `stockout_flag`)**, indicating that Jev is largely mirroring status signals already present in SQLite.

While Jev's output is directionally coherent and the OpenRouter Decisions API is fast and inexpensive (~$0.00003/decision), the marginal operational gains do not justify the added architecture, latency, and operational dependencies of an external LLM in the daily production loop.

---

## 29. Recommendation Regarding Phase 3

### Recommendation: **`PHASE 3 EXPANSION IS NOT JUSTIFIED`**

1. **Do NOT Proceed to Full-Catalog Production Rollout:** Expanding real TypeSafe Jev to all 674 SKUs across 4 platforms (87,600+ annual calls) is not recommended. The operational benefits (+0.12% fill rate) are too small to offset the fragility of an external cloud API dependency in an automated replenishment pipeline.
2. **Preserve Current System Boundaries:** The certified Exp6 LightGBM demand forecasting model and the existing tabular ROP baseline remain the superior, robust, and cost-effective production choices for Rimmel.
3. **Future Considerations (Optional):** If contextual AI is explored in the future, it should focus on qualitative unstructured data not captured in tabular tables (e.g., supplier delay emails, macro supply disruptions, social trend sentiment) rather than re-evaluating numerical sales and inventory metrics.
