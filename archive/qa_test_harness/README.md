# Rimmel Demand Forecasting System — QA Audit Architecture & Specification

## 1. System Overview & Certified Architecture
The certified Rimmel Multi-Platform Demand Forecasting System is designated as **Exp6**:
- **Data Treatment**: ZERO Treatment (`ml_features_zero` table in `data/rimmel_clean.db`). Days with no observed sales for an active product-channel are modeled as true zero consumer demand ($y = 0$), rather than imputed with rolling averages.
- **Forecasting Engine**: LightGBM Regressor (`LGBMRegressor`).
  - `objective`: `'regression'`
  - `metric`: `'rmse'`
  - `n_estimators`: 150
  - `max_depth`: 6
  - `num_leaves`: 31
  - `learning_rate`: 0.05
  - `random_state`: 42
  - `n_jobs`: -1
  - `verbose`: -1
- **Calibration Engine**: Combined Post-Inference Calibration:
  - **$\alpha$-Dampening (0.10)**: Applied to confirmed zero-demand series when $v_7 = 0 \land v_{14} = 0 \land v_{30} = 0 \land \text{promo\_days\_30} = 0 \land \text{amazon\_sessions\_momentum} \le 1.25$.
  - **$\beta$-Dampening (0.10)**: Applied to confirmed warehouse stockouts when $\text{in\_stock\_flag} = 0 \land \text{has\_inventory\_signal} = 1$.
- **Inventory Model**: Single shared physical warehouse inventory pool across platforms. Warehouse stock is tracked per SKU and never summed across platforms.
- **Forecast Horizon**: 10 calendar days forward (operationally September 11 to September 20, 2026; validated on September 1 to September 10, 2026).
- **Aggregation & Rounding**: Continuous daily decimal predictions are summed over the 10-day planning horizon per series, and operational integer rounding is applied to the 10-day sum.

---

## 2. Inspected Production Files
- `src/final_production_system.py`: Primary productionization pipeline executing holdout validation, final model retraining, forward 10-day inference, Excel generation, and artifact serialization.
- `src/phase2_feature_engineering.py`: Engine computing all 74 causal features across 674 SKUs and 4 platforms.
- `src/sku_mapping.py` & `src/platform_mapping.py`: Canonical entity resolution and channel classification.
- `app.py`: Streamlit client dashboard consuming processed CSV/parquet deliverables and Excel reports.
- `models/production_lgbm_model.pkl`: Serialized certified LightGBM model binary.
- `models/production_features.json`: Serialized feature metadata (74 features, 8 categoricals).
- `models/production_model_config.json`: Production configuration and benchmark metrics.
- `data/rimmel_clean.db`: Authoritative SQLite database hosting `raw_transactions` and `ml_features_zero`.

---

## 3. Production Entry Points & Deliverables
1. **Pipeline Execution**: `python src/final_production_system.py`
2. **Dashboard UI**: `streamlit run app.py`
3. **Certified Artifacts**:
   - `models/production_lgbm_model.pkl` (SHA-256 verified)
   - `reports/validation_sep01_sep10_2026.xlsx` & `reports/Rimmel_Validation_Sep01_Sep10_2026.xlsx`
   - `reports/production_forecast_sep11_sep20_2026.xlsx` & `reports/Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx`
   - `reports/validation_metrics.csv`
   - `reports/feature_importance.csv`
   - `reports/final_production_forecast_sep11_sep20_2026.csv`

---

## 4. Feature Taxonomy (74 Causal Features)
1. **Long-Term Base Demand (11 features)**: `v90`, `v180`, `v365`, `sales_days_90`, `sales_days_180`, `same_period_last_year_7d`, `same_period_last_year_30d`, `yoy_7d`, `yoy_30d`, `v30_vs_v365`, `v90_vs_v365`
2. **Recent Demand (5 features)**: `lag_1`, `v7`, `v14`, `v30`, `sales_days_30`
3. **Momentum (8 features)**: `v14_vs_v30`, `v30_vs_v90`, `v30_vs_v180`, `v60`, `lag_7`, `lag_14`, `lag_30`, `lag_90`
4. **Volatility / Behavior (4 features)**: `cv_30`, `cv_90`, `day_of_week`, `is_weekend`
5. **Inventory (12 features)**: `current_stock`, `has_inventory_signal`, `in_stock_flag`, `stockout_flag`, `days_since_stockout`, `v14_instock`, `v30_instock`, `v90_instock`, `has_restock_date`, `days_from_restock`, `restock_known`, `restock_status`
6. **Platform Commercial Signals (16 features)**:
   - Pricing: `selling_price`, `zero_price_flag`, `price_vs_30d`, `price_vs_90d`, `price_change_30d`
   - Amazon: `amazon_sessions_7d`, `amazon_sessions_30d`, `amazon_sessions_90d`, `amazon_sessions_momentum`, `buy_box_7d`, `buy_box_30d`, `buy_box_90d`, `buy_box_change`, `units_per_session_30d`
   - eBay: `promo_days_7`, `promo_days_30`, `promo_days_90`, `promo_ratio_30`, `promotion_started`, `promotion_ended`
7. **Product Context & Cross-Platform (18 features)**:
   - Product: `pack_multiplier`, `category`, `days_since_launch`, `category_resolution_method`, `launch_date_resolution_method`, `canonical_sku_source`, `resolved_parent_id`, `canonical_sku`, `platform_group`
   - Cross-Platform: `platform_share_30d`, `other_platform_sales_7d`, `other_platform_sales_30d`

---

## 5. Known Production Constraints & Guardrails
- **Zero Future Leakage Rule**: At prediction date $D$, all features must use observations strictly from $t \le D$.
- **Shared Warehouse Invariant**: Physical stock on hand is shared across channels and must never be aggregated across platforms.
- **Non-Negative Forecast Guarantee**: Predictions are clipped at zero $\max(0, \hat{y})$ before calibration.
- **Platform Signal Isolation**: Platform-specific signals (e.g. Amazon Buy Box, eBay Promoted Listing) must be NULL/zero for channels where they are not applicable.
- **Decimal Parity**: Statistical metrics (WAPE, MAE, Bias) must be computed on unrounded decimals to prevent cumulative rounding distortion.
