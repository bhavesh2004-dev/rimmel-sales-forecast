# LONG-TERM DEMAND FEATURE WEIGHTING EXPERIMENT

**Document ID**: `EXP-LT-WEIGHTING-20261006`  
**Pipeline**: Unified Multi-Brand Demand Forecasting Engine (`multibrand_pipeline/`)  
**Scope**: Controlled Evaluation of Long-Term Feature Contribution Weighting in Global LightGBM  
**Author**: Senior ML Demand Forecasting Architect & MLOps Reviewer  
**Status**: **COMPLETE — AUDITED & CERTIFIED**  
**Final Decision**: **`KEEP CURRENT 60-FEATURE MODEL`**  

---

## 1. Objective

Phase 3 established that recent-demand features heavily dominate LightGBM split-selection and total information gain:
- `lag_1`: $\approx 61.7\% - 71.0\%$ of total tree gain
- `v7`: $\approx 16.5\% - 21.9\%$ of total tree gain
- Combined Short-Term Demand Lags & Velocities: $> 83\%$ of total tree gain

The objective of this controlled experiment is to evaluate whether the model is under-utilizing existing medium- and long-term signals (`lag_30`, `lag_90`, `lag_180`, `lag_365`, `v30`, `v60`, `v90`, `v180`, `v365`, `same_period_last_year_7d`, `yoy_30d`, etc.) due to greedy tree split preferences, and whether gently shifting split-gain preferences toward longer-horizon features improves multi-window out-of-sample forecast accuracy.

---

## 2. Hypothesis

**Hypothesis**:  
> *"Gently penalizing short-term features while boosting split gains on medium- and long-term demand persistence, seasonality, and momentum will produce more stable predictions on products with annual seasonality, long-term trends, or recent noisy dips, thereby reducing overall out-of-sample WAPE across multiple walk-forward windows without introducing forecast bias or harming brand-level stability."*

**Falsification Criteria**:  
If weighting long-term features fails to consistently reduce out-of-sample WAPE across development windows, increases forecast bias, or causes material regression on either Rimmel London or Max Factor, the hypothesis is rejected and the model remains in its validated unweighted form.

---

## 3. Existing Feature Groups

No new features were created. The audited 60-feature schema from `config/feature_schema.json` was partitioned into four mutually exclusive groups:

```
Total Audited Schema: 60 Features
├── GROUP A — Short-Term Demand (5 Features):
│     lag_1, lag_7, lag_14, v7, v14
├── GROUP B — Medium-Term Demand (8 Features):
│     lag_30, v30, v60, sales_days_30, sales_days_90, v14_vs_v30, v30_vs_v90, v30_vs_v180
├── GROUP C — Long-Term Demand & Seasonality (12 Features):
│     lag_90, lag_180, lag_365, v90, v180, v365, v30_vs_v365, v90_vs_v365,
│     same_period_last_year_7d, same_period_last_year_30d, yoy_7d, yoy_30d
└── OTHER FEATURES — Entity, Economics, Traffic, Lifecycle (35 Features):
      canonical_sku, sales_days_180, cv_30, cv_90, current_stock, in_stock_flag,
      days_since_stockout, v14_instock, v30_instock, v90_instock, has_restock_date,
      days_from_restock, selling_price, price_vs_30d, price_vs_90d, price_change_30d,
      amazon_sessions_7d, amazon_sessions_30d, amazon_sessions_90d, amazon_sessions_momentum,
      buy_box_7d, buy_box_30d, buy_box_90d, buy_box_change, units_per_session_30d,
      promo_days_7, promo_days_30, promo_days_90, promotion_started, day_of_week,
      days_since_launch, resolved_parent_id, platform_share_30d, other_platform_sales_7d,
      other_platform_sales_30d
```

*Verification*: $5 + 8 + 12 + 35 = 60$ features total. Zero duplicates.

---

## 4. Experiment Configurations

LightGBM natively supports split-gain weighting through the `feature_contri` parameter (passed as a 60-element floating-point vector corresponding to the feature columns in the training matrix). This alters the gain calculated at each prospective split threshold without modifying raw numeric data.

Exactly three configurations were evaluated:

| Parameter / Group | Model A: Existing Baseline | Model B: Moderate Long-Term Emphasis | Model C: Stronger Long-Term Emphasis |
|:---|:---:|:---:|:---:|
| **Group A (Short-Term)** | `1.00` | `0.95` | `0.90` |
| **Group B (Medium-Term)** | `1.00` | `1.10` | `1.15` |
| **Group C (Long-Term)** | `1.00` | `1.20` | `1.30` |
| **Other 35 Features** | `1.00` | `1.00` | `1.00` |
| **Hyperparameters** | Identical (150 trees, depth 6, lr 0.05) | Identical (150 trees, depth 6, lr 0.05) | Identical (150 trees, depth 6, lr 0.05) |
| **Exp6 Calibration** | $\alpha=0.10, \beta=0.10$ | $\alpha=0.10, \beta=0.10$ | $\alpha=0.10, \beta=0.10$ |
| **Random State** | `42` | `42` | `42` |

---

## 5. Development Windows (Walk-Forward Methodology)

In accordance with strict MLOps principles, model selection was executed exclusively on three historical walk-forward development windows ending before the locked September benchmark:

```
Historical Development Walk-Forward Timeline:
├── Window 1:
│     Train: 2025-08-01 → 2026-05-31 (304 days, 684,304 rows)
│     Eval:  2026-06-01 → 2026-06-10 (10 days,  22,510 rows)
├── Window 2:
│     Train: 2025-08-01 → 2026-06-20 (324 days, 729,324 rows)
│     Eval:  2026-06-21 → 2026-07-01 (11 days,  24,761 rows)
└── Window 3:
      Train: 2025-08-01 → 2026-07-09 (343 days, 772,093 rows)
      Eval:  2026-07-10 → 2026-07-20 (11 days,  24,761 rows)

Locked Holdout Benchmark (Confirmation Only):
└── Sep Benchmark:
      Train: 2025-08-01 → 2026-08-31 (396 days, 891,396 rows)
      Eval:  2026-09-01 → 2026-09-10 (10 days,  22,510 rows)
```

No future observations entered features or training matrices in any window.

---

## 6. Per-Window Metrics

### Window 1 Evaluation (`2026-06-01` to `2026-06-10`, 10 Days)

| Model Configuration | Actual Units | Calib Predicted | Abs Error | Calib WAPE | Calib Bias | Raw WAPE | Raw Bias | MAE | RMSE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A: Baseline** | 3,523.00 | 2,927.93 | 3,330.83 | **94.55%** | -16.89% | 102.30% | -7.71% | 0.1480 | 0.6619 |
| **Model B: Moderate** | 3,523.00 | 2,891.79 | 3,324.62 | **94.37%** | -17.92% | 102.17% | -8.54% | 0.1477 | 0.6601 |
| **Model C: Stronger** | 3,523.00 | 2,896.95 | 3,317.02 | **94.15%** | -17.77% | 102.26% | -8.27% | 0.1474 | 0.6568 |

### Window 2 Evaluation (`2026-06-21` to `2026-07-01`, 11 Days)

| Model Configuration | Actual Units | Calib Predicted | Abs Error | Calib WAPE | Calib Bias | Raw WAPE | Raw Bias | MAE | RMSE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A: Baseline** | 3,540.00 | 3,300.82 | 3,512.32 | **99.22%** | -6.76% | 107.34% | +2.51% | 0.1418 | 0.6431 |
| **Model B: Moderate** | 3,540.00 | 3,281.77 | 3,501.18 | **98.90%** | -7.29% | 107.22% | +2.08% | 0.1414 | 0.6425 |
| **Model C: Stronger** | 3,540.00 | 3,288.39 | 3,506.90 | **99.06%** | -7.11% | 107.58% | +2.42% | 0.1416 | 0.6407 |

### Window 3 Evaluation (`2026-07-10` to `2026-07-20`, 11 Days)

| Model Configuration | Actual Units | Calib Predicted | Abs Error | Calib WAPE | Calib Bias | Raw WAPE | Raw Bias | MAE | RMSE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A: Baseline** | 6,112.00 | 5,005.76 | 5,182.02 | **84.78%** | -18.10% | 89.24% | -11.23% | 0.2093 | 1.6805 |
| **Model B: Moderate** | 6,112.00 | 5,028.46 | 5,224.45 | **85.48%** | -17.73% | 89.94% | -10.97% | 0.2110 | 1.6856 |
| **Model C: Stronger** | 6,112.00 | 5,084.52 | 5,205.13 | **85.16%** | -16.81% | 89.65% | -10.13% | 0.2102 | 1.6749 |

*Critical Observation on Window 3*: In Window 3, **both weighted models REGRESSED** relative to Model A Baseline (+0.70 pp on Model B, +0.38 pp on Model C), demonstrating that gains in Windows 1 and 2 did not generalize consistently.

---

## 7. Average Development Metrics

Averaged across all three historical walk-forward development windows (representing 32 out-of-sample calendar days and 72,032 SKU-day predictions):

| Development Metric | Model A: Baseline | Model B: Moderate | Model C: Stronger | Delta vs. Baseline (B - A) | Delta vs. Baseline (C - A) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Average Calibrated WAPE** | **92.85%** | **92.92%** | **92.79%** | **+0.07 pp (Worse)** | **-0.06 pp (Tie)** |
| **Average Raw WAPE** | **99.63%** | **99.78%** | **99.83%** | **+0.15 pp (Worse)** | **+0.20 pp (Worse)** |
| **Average Calibrated Bias** | -13.92% | -14.31% | -13.90% | -0.39 pp (Worse underforecast) | +0.02 pp |
| **Average Raw Bias** | -5.48% | -5.88% | -5.33% | -0.40 pp | +0.15 pp |
| **Average Calibrated MAE** | 0.1664 | 0.1667 | 0.1664 | +0.0003 | 0.0000 |
| **Average Calibrated RMSE** | 0.9952 | 0.9972 | 0.9908 | +0.0020 | -0.0044 |
| **Rimmel Average Calib WAPE**| **90.80%** | **90.54%** | **90.35%** | -0.26 pp | -0.45 pp |
| **Max Factor Avg Calib WAPE**| **102.38%** | **103.48%** | **103.60%** | **+1.10 pp (REGRESSION)** | **+1.22 pp (REGRESSION)** |

---

## 8. Rimmel London Metrics

Performance on the Rimmel London catalog (1,413 SKUs) across all development windows:

| Window / Metric | Model A: Baseline | Model B: Moderate | Model C: Stronger | Delta (B - A) | Delta (C - A) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Window 1 (Calib WAPE)** | 90.30% | 89.99% | 89.78% | -0.31 pp | -0.52 pp |
| **Window 1 (Calib Bias)** | -15.44% | -16.79% | -16.44% | -1.35 pp | -1.00 pp |
| **Window 2 (Calib WAPE)** | 96.75% | 96.15% | 96.13% | -0.60 pp | -0.62 pp |
| **Window 2 (Calib Bias)** | -4.85% | -5.54% | -5.71% | -0.69 pp | -0.86 pp |
| **Window 3 (Calib WAPE)** | 85.34% | 85.47% | 85.15% | +0.13 pp | -0.19 pp |
| **Window 3 (Calib Bias)** | -12.51% | -12.48% | -10.69% | +0.03 pp | +1.82 pp |
| **Development Average WAPE** | **90.80%** | **90.54%** | **90.35%** | **-0.26 pp** | **-0.45 pp** |

*Analysis for Rimmel*: Rimmel displays slight WAPE reductions on stable SKUs, but suffers from increased negative volume bias (underforecasting) in Windows 1 and 2.

---

## 9. Max Factor Metrics

Performance on the Max Factor catalog (838 SKUs) across all development windows:

| Window / Metric | Model A: Baseline | Model B: Moderate | Model C: Stronger | Delta (B - A) | Delta (C - A) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Window 1 (Calib WAPE)** | 113.83% | 114.31% | 114.03% | **+0.48 pp (Worse)** | **+0.20 pp (Worse)** |
| **Window 1 (Calib Bias)** | -23.48% | -23.05% | -23.80% | +0.43 pp | -0.32 pp |
| **Window 2 (Calib WAPE)** | 109.76% | 110.64% | 111.58% | **+0.88 pp (Worse)** | **+1.82 pp (Worse)** |
| **Window 2 (Calib Bias)** | -14.88% | -14.77% | -13.06% | +0.11 pp | +1.82 pp |
| **Window 3 (Calib WAPE)** | 83.55% | 85.49% | 85.19% | **+1.94 pp (Worse)** | **+1.64 pp (Worse)** |
| **Window 3 (Calib Bias)** | -30.58% | -29.44% | -30.47% | +1.14 pp | +0.11 pp |
| **Development Average WAPE** | **102.38%** | **103.48%** | **103.60%** | **+1.10 pp (REGRESSION)** | **+1.22 pp (REGRESSION)** |

*Critical Analysis for Max Factor*:  
Max Factor **regressed in every single development window**. Due to extreme catalog sparsity ($95.24\%$ zero sales days), long-term signals (`v90`, `v180`, `lag_365`) reflect stale historical transactions rather than active operational demand. Artificially forcing higher split preference on long-term signals caused trees to over-predict dormant Max Factor SKUs, increasing error across the board.

---

## 10. Raw vs. Calibrated Results

Comparing the raw tree model against Exp6-calibrated outputs across configurations:

| Metric | Model A Raw | Model A Calib | Model B Raw | Model B Calib | Model C Raw | Model C Calib |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Window 1 WAPE** | 102.30% | 94.55% | 102.17% | 94.37% | 102.26% | 94.15% |
| **Window 1 Bias** | -7.71% | -16.89% | -8.54% | -17.92% | -8.27% | -17.77% |
| **Window 2 WAPE** | 107.34% | 99.22% | 107.22% | 98.90% | 107.58% | 99.06% |
| **Window 2 Bias** | +2.51% | -6.76% | +2.08% | -7.29% | +2.42% | -7.11% |
| **Window 3 WAPE** | 89.24% | 84.78% | 89.94% | 85.48% | 89.65% | 85.16% |
| **Window 3 Bias** | -11.23% | -18.10% | -10.97% | -17.73% | -10.13% | -16.81% |
| **Dev Avg Raw WAPE** | **99.63%** | — | **99.78%** | — | **99.83%** | — |
| **Dev Avg Calib WAPE**| — | **92.85%** | — | **92.92%** | — | **92.79%** |

*Takeaway*: The raw uncalibrated model also degrades under feature weighting (Raw WAPE increases from $99.63\%$ to $99.78\%$ on Model B and $99.83\%$ on Model C). Calibration dampens the negative effects but does not reverse the fundamental trend.

---

## 11. Feature Gain Distribution

In the locked September model, the total tree gain shifted dynamically in response to `feature_contri`:

### Top 10 Features by Informational Gain (September Models)

| Rank | Model A: Baseline | Split / Gain Share | Model B: Moderate | Split / Gain Share | Model C: Stronger | Split / Gain Share |
|:---:|:---|:---:|:---|:---:|:---|:---:|
| 1 | `lag_1` | 580 / **61.74%** | `lag_1` | 555 / **59.95%** | `lag_1` | 522 / **57.88%** |
| 2 | `v7` | 354 / **19.04%** | `v7` | 310 / **16.47%** | `v30` | 137 / **14.63%** |
| 3 | `canonical_sku` | 320 / **1.68%** | `v30` | 110 / **3.24%** | `v7` | 328 / **6.56%** |
| 4 | `v14` | 105 / **1.26%** | `canonical_sku` | 313 / **1.66%** | `canonical_sku` | 301 / **1.62%** |
| 5 | `lag_7` | 116 / **1.25%** | `v90` | 67 / **1.42%** | `v90` | 72 / **1.61%** |
| 6 | `v30` | 61 / **0.77%** | `lag_7` | 90 / **1.35%** | `lag_7` | 80 / **1.35%** |
| 7 | `days_since_stockout` | 70 / **0.67%** | `buy_box_7d` | 87 / **0.79%** | `v30_vs_v180` | 63 / **0.86%** |
| 8 | `buy_box_7d` | 90 / **0.67%** | `v30_vs_v180` | 65 / **0.75%** | `amazon_sessions_7d`| 91 / **0.75%** |
| 9 | `current_stock` | 92 / **0.62%** | `current_stock` | 79 / **0.61%** | `lag_180` | 90 / **0.72%** |
| 10 | `price_vs_30d` | 117 / **0.61%** | `amazon_sessions_7d`| 81 / **0.59%** | `current_stock` | 90 / **0.70%** |

*Key Shift*: In Model C, `v30` rose to become the #2 most important feature (14.63% share), and `v90` and `lag_180` both entered the top 10. `feature_contri` succeeded mechanically in boosting long-term feature utilization.

---

## 12. Short/Medium/Long-Term Gain Distribution

Aggregate information gain and tree split counts by feature group across all evaluated periods:

| Window | Model | Group A (Short) Gain Share | Group B (Medium) Gain Share | Group C (Long) Gain Share | Other Features Share |
|:---|:---|:---:|:---:|:---:|:---:|
| **Window 1** | Model A (Baseline) | **84.09%** (1,183 splits) | **2.77%** (375 splits) | **1.85%** (264 splits) | 11.29% (1,654 splits) |
| | Model B (Moderate) | **78.95%** (1,057 splits) | **6.57%** (465 splits) | **3.50%** (444 splits) | 10.98% (1,542 splits) |
| | Model C (Stronger) | **66.70%** (955 splits) | **17.61%** (514 splits) | **5.34%** (518 splits) | 10.35% (1,352 splits) |
| **Window 2** | Model A (Baseline) | **84.12%** (1,205 splits) | **3.07%** (407 splits) | **1.71%** (285 splits) | 11.11% (1,682 splits) |
| | Model B (Moderate) | **78.67%** (1,081 splits) | **6.89%** (487 splits) | **3.90%** (520 splits) | 10.53% (1,575 splits) |
| | Model C (Stronger) | **65.84%** (980 splits) | **17.76%** (490 splits) | **5.69%** (587 splits) | 10.71% (1,525 splits) |
| **Window 3** | Model A (Baseline) | **82.82%** (1,215 splits) | **3.77%** (448 splits) | **2.03%** (320 splits) | 11.38% (1,739 splits) |
| | Model B (Moderate) | **77.23%** (1,104 splits) | **8.12%** (491 splits) | **3.76%** (541 splits) | 10.89% (1,624 splits) |
| | Model C (Stronger) | **64.33%** (958 splits) | **19.27%** (588 splits) | **5.30%** (635 splits) | 11.11% (1,564 splits) |
| **September** | Model A (Baseline) | **83.52%** (1,209 splits) | **2.77%** (328 splits) | **2.24%** (353 splits) | 11.47% (1,768 splits) |
| | Model B (Moderate) | **78.37%** (1,062 splits) | **6.08%** (484 splits) | **4.73%** (524 splits) | 10.82% (1,643 splits) |
| | Model C (Stronger) | **66.05%** (1,003 splits) | **17.61%** (514 splits) | **5.76%** (669 splits) | 10.58% (1,573 splits) |

---

## 13. September Locked Benchmark Confirmation

As required by Section 11, the configurations were verified on the locked September benchmark (`2026-09-01` to `2026-09-10`, 22,510 rows):

| Metric | Model A: Baseline | Model B: Moderate | Model C: Stronger | Delta (B - A) | Delta (C - A) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Ground Truth Actual** | 2,627.00 | 2,627.00 | 2,627.00 | — | — |
| **Raw Predicted Units** | 2,929.49 | 2,908.79 | 2,927.51 | -20.70 | -1.98 |
| **Raw WAPE (%)** | **107.95%** | **106.96%** | **107.36%** | -0.99 pp | -0.59 pp |
| **Raw Bias (%)** | +11.51% | +10.73% | +11.44% | -0.78 pp | -0.07 pp |
| **Raw MAE** | 0.1260 | 0.1248 | 0.1253 | -0.0012 | -0.0007 |
| **Raw RMSE** | 0.5785 | 0.5713 | 0.5751 | -0.0072 | -0.0034 |
| **Calibrated Units** | 2,655.79 | 2,650.14 | 2,671.93 | -5.65 | +16.14 |
| **Calibrated WAPE (%)** | **99.04%** | **98.44%** | **98.99%** | -0.60 pp | -0.05 pp |
| **Calibrated Bias (%)** | **+1.10%** | **+0.88%** | **+1.71%** | -0.22 pp | +0.61 pp |
| **Calibrated MAE** | 0.1156 | 0.1149 | 0.1155 | -0.0007 | -0.0001 |
| **Calibrated RMSE** | 0.5785 | 0.5718 | 0.5761 | -0.0067 | -0.0024 |
| **Rimmel Calib WAPE** | **92.15%** | **91.15%** | **91.38%** | -1.00 pp | -0.77 pp |
| **Max Factor Calib WAPE**| **124.58%** | **125.48%** | **127.22%** | **+0.90 pp (Worse)** | **+2.64 pp (REGRESSION)** |

### Diagnostic Analysis of SKU Archetypes

An empirical investigation was conducted on the September holdout across six distinct demand archetypes:

| SKU Archetype | Sub-Population Criteria | Series Count | Actual Units | Model A WAPE | Model B WAPE | Model C WAPE | Impact of Long-Term Weighting |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1. Stable Year-Round Demand** | `sales_days_90 >= 60` & `cv_30 <= 0.8` | 106 | 618.0 | **44.61%** | **42.69%** | 44.17% | **Improved** (-1.92 pp on Model B) |
| **2. Seasonal Behavior** | `same_period_last_year_30d > 0` & `\|yoy_30d\| > 0.3` | 9,536 | 1,493.0 | **99.16%** | **98.85%** | 98.87% | **Neutral** (-0.31 pp delta) |
| **3. Long-Term Momentum** | `v365 > 0.5` & `\|v30_vs_v365 - 1.0\| > 0.4` | 1,480 | 1,033.0 | **81.06%** | **79.40%** | 80.51% | **Improved** (-1.66 pp on Model B) |
| **4. Recent 7-Day Misleading** | `v90 > 0.5` & `\|v7 - v90\| > 1.0` | 490 | 785.0 | **72.25%** | **70.70%** | 71.28% | **Improved** (-1.55 pp on Model B) |
| **5. Intermittent Spikes** | `sales_days_90 < 20` & `cv_30 > 1.2` & `v30 > 0.1` | 1,629 | 438.0 | **132.31%** | **133.47%** | **134.28%** | **REGRESSED** (+1.16 pp on B, +1.97 pp on C) |
| **6. Recent vs. Historical Divergent** | `v365 > 1.0` & `\|v7 - v365\| > 1.5` | 549 | 962.0 | **62.00%** | **60.01%** | 60.76% | **Improved** (-1.99 pp on Model B) |

#### Structural Root Cause of the Trade-Off

1. **Why Stable SKUs Benefit**: For high-volume, established SKUs (mostly Rimmel), long-term signals (`v30`, `v90`, `v365`) provide a steady baseline anchor when recent 7-day sales experience momentary volatility.
2. **Why Intermittent SKUs Regress**: For sparse, intermittent SKUs (the vast majority of Max Factor and long-tail Rimmel), long-term features are noisy or contain ancient transaction volume. Forcing the tree to place higher split value on these features creates "phantom demand" false positives during dormant zero-demand runs.
3. **The Multi-Brand Dilemma**: Because this is a **single unified multi-brand model**, boosting long-term weights causes a direct transfer of error: a minor $+1.0\%$ benefit on Rimmel is purchased at the cost of a severe $+2.64\%$ degradation on Max Factor.

---

## 14. Final Decision

### Model Selection Evaluation (Section 10 Criteria)

| Selection Criterion | Model B: Moderate | Model C: Stronger | Gate Assessment |
|:---|:---:|:---:|:---:|
| **1. Lower average development WAPE across walk-forward windows** | 92.92% vs. 92.85% (**FAILS**) | 92.79% vs. 92.85% (**TIE**, -0.06 pp) | **FAILED** (No consistent material reduction) |
| **2. No material increase in forecast bias** | -14.31% vs. -13.92% (**Worse**) | -13.90% vs. -13.92% (Pass) | **FAILED** for B; Neutral for C |
| **3. No material deterioration in Rimmel performance** | 90.54% vs. 90.80% (Pass) | 90.35% vs. 90.80% (Pass) | **PASSED** |
| **4. No material deterioration in Max Factor performance** | 103.48% vs. 102.38% (**REGRESSED +1.10 pp**) | 103.60% vs. 102.38% (**REGRESSED +1.22 pp**) | **FAILED** (Max Factor degraded across all windows) |
| **5. Improvement appears consistently across multiple windows** | Regressed in Window 3 (**FAILS**) | Regressed in Window 3 (**FAILS**) | **FAILED** (Window 3 regressed on both models) |

---

### FINAL AUTHORITATIVE DECISION:

# **`KEEP CURRENT 60-FEATURE MODEL`**

### Technical Conclusion

1. **The Greedy Split Preference is Optimal**: LightGBM's unconstrained split selection—which devotes $\approx 83.5\%$ of gain to short-term dynamics (`lag_1`, `v7`) and allows medium/long-term features to enter organically—represents the mathematically superior bias-variance tradeoff across the combined catalog.
2. **Preservation of Multi-Brand Balance**: Unweighted LightGBM maintains equitable treatment between high-velocity Rimmel SKUs and sparse Max Factor SKUs. Artificial feature weighting harms Max Factor by over-emphasizing historical noise on low-frequency products.
3. **Stop Rule Enforced (Section 16)**: This experiment concludes all exploratory feature-weighting work. No further weights, hyperparameter tuning, or model variants will be tested. The validated 60-feature Global LightGBM architecture is frozen and ready for the Phase 4 operational refit.
