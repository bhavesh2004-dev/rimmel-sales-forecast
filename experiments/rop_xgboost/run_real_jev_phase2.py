"""
Phase 2 Execution Engine: Real TypeSafe Jev + XGBoost ROP Incremental-Value Experiment
=====================================================================================
Executes Scenario C approved scope:
- 28 representative SKU x Platform cases
- 62 daily historical walk-forward dates (2026-07-01 to 2026-08-31)
- 1,736 logical Jev decision contexts
- Model: Real TypeSafe Jev (~typesafe/jev-latest -> typesafe/jev-1.13-20260917)
- Persistent disk caching via jev_cache.py
- Zero production impact: isolated exclusively under experiments/rop_xgboost/

Strict Leakage Invariant:
For historical date D, Jev state context strictly uses information contemporaneous
to date D (sales through D, v7/v14/v30, stock as of D, Buy Box as of D, etc.).
Forward 7-day lead-time demand is strictly ground truth for XGBoost evaluation.
"""

import os
import sys
import time
import json
import sqlite3
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rop_xgboost.config import (
    DEFAULT_LEAD_TIME_DAYS, TARGET_SERVICE_LEVEL, OUTPUTS_DIR, DB_PATH
)
from experiments.rop_xgboost.jev_cache import JevCache
from experiments.rop_xgboost.real_jev_adapter import RealJevOpenRouterAdapter
from experiments.rop_xgboost.rop_target import (
    compute_reorder_trigger, compute_reorder_quantity
)

# Constants
START_DATE = "2026-07-01"
END_DATE = "2026-08-31"
NUM_DAYS = 62
NUM_CASES = 28
TOTAL_EXPECTED_CALLS = NUM_DAYS * NUM_CASES  # 1,736
MAX_WORKERS = 4

BASE_FEATURE_COLS = [
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'v7', 'v14', 'v30', 'v90', 'v14_vs_v30',
    'cv_30', 'sales_days_30',
    'current_stock', 'in_stock_flag', 'days_since_stockout',
    'selling_price', 'promo_days_30', 'amazon_sessions_momentum', 'buy_box_7d',
    'day_of_week', 'is_weekend', 'month', 'platform_cat'
]

JEV_FEATURE_COLS = [
    'jev_urgent_replenishment_prob',
    'jev_replenishment_risk_score',
    'jev_prio_HIGH',
    'jev_prio_MEDIUM',
    'jev_prio_LOW',
    'jev_confidence'
]


def load_cohort_and_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the 28 representative cases and queries ml_features_zero for the 62 walk-forward dates.
    Also computes forward 7-day lead-time demand (ltd_target) from observed sales.
    """
    out_dir = Path(OUTPUTS_DIR)
    cases_path = out_dir / "real_jev_phase1_case_summary.csv"
    if not cases_path.exists():
        raise FileNotFoundError(f"Cohort case summary not found at {cases_path}")

    cases_df = pd.read_csv(cases_path)
    print(f"[DATA SETUP] Loaded {len(cases_df)} representative SKU x Platform cases.")

    unique_skus = [f"'{s}'" for s in cases_df['sku'].unique()]
    sku_clause = ", ".join(unique_skus)

    conn = sqlite3.connect(DB_PATH)
    # Query up to 2026-09-08 so the 7-day forward demand for 2026-08-31 is fully populated
    query = f"""
    SELECT 
        date, canonical_sku, platform_group, category, resolved_parent_id,
        observed_units_sold, model_units_sold,
        lag_1, lag_7, lag_14, lag_30,
        v7, v14, v30, v90, v14_vs_v30,
        cv_30, sales_days_30,
        current_stock, in_stock_flag, stockout_flag, days_since_stockout,
        selling_price, promo_days_30, ebay_promoted_flag,
        amazon_sessions_momentum, buy_box_7d,
        day_of_week, is_weekend
    FROM ml_features_zero
    WHERE date >= '2026-06-20' AND date <= '2026-09-10'
      AND canonical_sku IN ({sku_clause})
    ORDER BY canonical_sku, platform_group, date
    """
    raw_df = pd.read_sql_query(query, conn)
    conn.close()

    # Filter to only the 28 cases
    case_keys = set(zip(cases_df['sku'], cases_df['platform']))
    raw_df['key'] = list(zip(raw_df['canonical_sku'], raw_df['platform_group']))
    cohort_raw = raw_df[raw_df['key'].isin(case_keys)].copy().drop(columns=['key'])

    # Compute 7-day forward lead-time demand per series
    cohort_raw['ltd_target'] = 0.0
    for (sku, plat), grp_indices in cohort_raw.groupby(['canonical_sku', 'platform_group']).groups.items():
        s = cohort_raw.loc[grp_indices, 'observed_units_sold']
        fwd = s.iloc[::-1].rolling(DEFAULT_LEAD_TIME_DAYS, min_periods=1).sum().iloc[::-1].shift(-1).fillna(0.0)
        cohort_raw.loc[grp_indices, 'ltd_target'] = fwd.values.astype(np.float32)

    # Filter down to the strict 62 walk-forward dates: 2026-07-01 to 2026-08-31
    mask = (cohort_raw['date'] >= START_DATE) & (cohort_raw['date'] <= END_DATE)
    df_eval = cohort_raw[mask].copy().reset_index(drop=True)

    # Derived days of cover
    df_eval['days_of_cover'] = df_eval.apply(
        lambda r: round(r['current_stock'] / r['v14'], 1) if r['v14'] > 0 else (999.0 if r['current_stock'] > 0 else 0.0),
        axis=1
    )

    # Calendar features
    df_eval['date_dt'] = pd.to_datetime(df_eval['date'])
    df_eval['month'] = df_eval['date_dt'].dt.month

    # Platform category code
    platform_map = {'Amazon': 0, 'eBay': 1, 'Website': 2, 'Other': 3}
    df_eval['platform_cat'] = df_eval['platform_group'].map(platform_map).fillna(3).astype(int)

    # Clean fill nulls
    df_eval['current_stock'] = df_eval['current_stock'].fillna(0.0)
    df_eval['selling_price'] = df_eval['selling_price'].fillna(df_eval['selling_price'].median())
    df_eval['amazon_sessions_momentum'] = df_eval['amazon_sessions_momentum'].fillna(1.0)
    df_eval['buy_box_7d'] = df_eval['buy_box_7d'].fillna(0.0)
    df_eval['cv_30'] = df_eval['cv_30'].fillna(0.0)
    df_eval['v14_vs_v30'] = df_eval['v14_vs_v30'].fillna(1.0)
    df_eval['days_since_stockout'] = df_eval['days_since_stockout'].fillna(999.0)

    # Merge case metadata (case_id, stratum)
    df_eval = df_eval.merge(
        cases_df[['sku', 'platform', 'case_id', 'stratum']],
        left_on=['canonical_sku', 'platform_group'],
        right_on=['sku', 'platform'],
        how='left'
    ).drop(columns=['sku', 'platform'])

    print(f"[DATA SETUP] Evaluation dataset ready: {len(df_eval)} rows across {df_eval['date'].nunique()} dates.")
    return cases_df, df_eval


def format_business_state_dynamic(row: pd.Series) -> str:
    """
    Format a single SKU x Platform case into a factual business decision state.
    Strictly contemporaneous to row['date'] — zero future leakage.
    """
    plat = row['platform_group']
    stock = row['current_stock']
    in_stock = "In Stock" if row['in_stock_flag'] == 1 else "Out of Stock"
    doc = f"{row['days_of_cover']:.1f} days" if row['days_of_cover'] < 900 else "Excess (>900 days)"
    v7 = f"{row['v7']:.2f}"
    v14 = f"{row['v14']:.2f}"
    v30 = f"{row['v30']:.2f}"
    v90 = f"{row['v90']:.2f}"
    cv = f"{row['cv_30']:.2f}" if not pd.isna(row['cv_30']) and row['cv_30'] > 0 else "0.00 (Stable/Intermittent)"
    price = f"£{row['selling_price']:.2f}" if not pd.isna(row['selling_price']) and row['selling_price'] > 0 else "Not recorded"

    if plat == 'Amazon':
        bb = f"{row['buy_box_7d']:.1f}%" if not pd.isna(row['buy_box_7d']) else "Not available"
        traffic = f"{row['amazon_sessions_momentum']:.2f}x relative to 30d baseline" if not pd.isna(row['amazon_sessions_momentum']) else "Normal"
        promo = "None active"
    elif plat == 'eBay':
        bb = "Not applicable (eBay channel)"
        traffic = "Normal"
        promo = f"eBay Promoted Listing active (Promoted in {int(row['promo_days_30'])} of last 30 days)" if row.get('ebay_promoted_flag', 0) == 1 else "None active"
    else:
        bb = f"Not applicable ({plat} channel)"
        traffic = "Normal"
        promo = "None active"

    state_text = (
        f"Product SKU: {row['canonical_sku']} (Category: {row['category']}) on platform: {plat} as of decision date {row['date']}.\n"
        f"- Inventory Position: On-hand warehouse stock = {stock:.0f} units ({in_stock}, Stockout flag = {row['stockout_flag']}). "
        f"Calculated days of cover based on 14-day run rate = {doc}. Days since last stockout = {row['days_since_stockout']} days.\n"
        f"- Sales Velocity & Demand: 7-day average = {v7} units/day, 14-day average = {v14} units/day, "
        f"30-day average = {v30} units/day, 90-day average = {v90} units/day. 30-day demand volatility CV = {cv}. "
        f"Recent sales: lag 1 day = {row['lag_1']:.0f} units, lag 7 day = {row['lag_7']:.0f} units.\n"
        f"- Commercial Context: Selling price = {price}. Buy Box ownership = {bb}. Traffic momentum = {traffic}. Marketing promotions = {promo}.\n"
        f"- Supply Chain Constraints: Supplier lead time = Not available in source data; Supplier MOQ = Not available in source data; In-transit purchase orders = Not available in source data."
    )
    return state_text


def fetch_all_jev_decisions(df_eval: pd.DataFrame, adapter: RealJevOpenRouterAdapter) -> pd.DataFrame:
    """
    Fetches real TypeSafe Jev decisions for all 1,736 rows in parallel using persistent cache.
    """
    print(f"\n[REAL JEV ACQUISITION] Gathering {len(df_eval)} decisions across {MAX_WORKERS} workers...")
    start_time = time.time()

    # Pre-generate states
    tasks = []
    for idx, row in df_eval.iterrows():
        state_text = format_business_state_dynamic(row)
        tasks.append((idx, row['date'], row['canonical_sku'], row['platform_group'], state_text))

    results = {}
    completed_count = 0

    def worker_func(task_tuple):
        idx, date, sku, plat, state = task_tuple
        decision = adapter.get_decision(date=date, sku=sku, platform=plat, state_text=state)
        return idx, decision

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker_func, t): t for t in tasks}
        for future in as_completed(futures):
            idx, decision = future.result()
            results[idx] = decision
            completed_count += 1
            if completed_count % 100 == 0 or completed_count == len(df_eval):
                elapsed = time.time() - start_time
                print(f"  [Progress] {completed_count}/{len(df_eval)} decisions processed "
                      f"({adapter.cache_hits} cache hits, {adapter.api_calls_made} API calls, "
                      f"{adapter.retries_count} retries, est cost: ${adapter.total_cost_usd:.4f}, elapsed: {elapsed:.1f}s)")

    # Unpack into DataFrame
    df_enriched = df_eval.copy()
    for col in JEV_FEATURE_COLS:
        df_enriched[col] = 0.0

    df_enriched['jev_is_cached'] = False
    df_enriched['jev_priority_choice'] = 'MEDIUM'

    for idx, decision in results.items():
        df_enriched.at[idx, 'jev_urgent_replenishment_prob'] = decision['jev_urgent_replenishment_prob']
        df_enriched.at[idx, 'jev_replenishment_risk_score'] = decision['jev_replenishment_risk_score']
        df_enriched.at[idx, 'jev_prio_HIGH'] = decision['jev_prio_HIGH']
        df_enriched.at[idx, 'jev_prio_MEDIUM'] = decision['jev_prio_MEDIUM']
        df_enriched.at[idx, 'jev_prio_LOW'] = decision['jev_prio_LOW']
        df_enriched.at[idx, 'jev_confidence'] = decision['jev_confidence']
        df_enriched.at[idx, 'jev_is_cached'] = decision['is_cached']
        df_enriched.at[idx, 'jev_priority_choice'] = decision['jev_replenishment_priority_label']

    total_time = time.time() - start_time
    print(f"[REAL JEV ACQUISITION] Finished in {total_time:.1f}s. "
          f"Total API calls: {adapter.api_calls_made}, Cache hits: {adapter.cache_hits}, "
          f"Total cost: ${adapter.total_cost_usd:.5f}, Tokens: {adapter.total_input_tokens + adapter.total_output_tokens:,}")

    return df_enriched


def run_cross_fitting(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 4,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, float], Any]:
    """
    Fits XGBoost Quantile Regressor using out-of-fold cross-fitting across the 28 series.
    Guarantees that every observation is predicted out-of-fold without in-sample contamination.
    """
    unique_cases = df['case_id'].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(len(df), dtype=np.float32)
    last_model = None

    params = {
        'objective': 'reg:quantileerror',
        'quantile_alpha': TARGET_SERVICE_LEVEL,
        'n_estimators': 80,
        'learning_rate': 0.05,
        'max_depth': 3,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': random_state,
        'n_jobs': 2
    }

    for fold, (train_idx_cases, val_idx_cases) in enumerate(kf.split(unique_cases)):
        train_cases = unique_cases[train_idx_cases]
        val_cases = unique_cases[val_idx_cases]

        train_mask = df['case_id'].isin(train_cases)
        val_mask = df['case_id'].isin(val_cases)

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, 'ltd_target']
        X_val = df.loc[val_mask, feature_cols]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        last_model = model

        preds = model.predict(X_val)
        oof_preds[val_mask] = np.maximum(0.0, preds)

    # Pinball loss & empirical coverage
    y_true = df['ltd_target'].values
    errors = y_true - oof_preds
    alpha = TARGET_SERVICE_LEVEL
    pinball = np.maximum(alpha * errors, (alpha - 1.0) * errors)
    mean_pinball = float(np.mean(pinball))
    coverage = float(np.mean(y_true <= oof_preds) * 100.0)

    metrics = {
        'mean_pinball_loss': round(mean_pinball, 4),
        'empirical_coverage_pct': round(coverage, 2),
        'mean_predicted_rop': round(float(np.mean(oof_preds)), 2)
    }

    return oof_preds, metrics, last_model


def run_inventory_simulation(
    df_eval: pd.DataFrame,
    predicted_rop_col: str,
    model_name: str,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    target_cover_days: int = 14
) -> Dict[str, Any]:
    """
    Executes discrete-event inventory simulation on the 28 series over the 62 walk-forward days.
    """
    df_sim = df_eval.copy().sort_values(['canonical_sku', 'platform_group', 'date'])
    unique_series = df_sim[['canonical_sku', 'platform_group']].drop_duplicates().values
    dates = sorted(df_sim['date'].unique())

    total_demand_units = 0.0
    total_fulfilled_units = 0.0
    total_stockout_events = 0
    total_active_days = 0
    total_pos_triggered = 0
    total_units_ordered = 0.0
    daily_inventory_samples = []
    daily_doc_samples = []

    for sku, platform in unique_series:
        s_data = df_sim[(df_sim['canonical_sku'] == sku) & (df_sim['platform_group'] == platform)].set_index('date')

        first_row = s_data.iloc[0]
        run_rate = max(float(first_row.get('v14', 0.1)), 0.05)
        curr_stock = float(first_row.get('current_stock', run_rate * target_cover_days))
        if curr_stock <= 0:
            curr_stock = run_rate * target_cover_days

        in_transit_orders = {}  # arrival_date -> qty

        for current_date in dates:
            if current_date not in s_data.index:
                continue

            row = s_data.loc[current_date]
            actual_demand = float(row.get('observed_units_sold', 0.0))
            rop = float(row[predicted_rop_col])
            daily_v = max(float(row.get('v14', run_rate)), 0.05)

            # 1. Inbound PO deliveries arrive
            if current_date in in_transit_orders:
                curr_stock += in_transit_orders.pop(current_date)

            # 2. Demand fulfillment
            total_demand_units += actual_demand
            if actual_demand > 0:
                total_active_days += 1
                if curr_stock >= actual_demand:
                    total_fulfilled_units += actual_demand
                    curr_stock -= actual_demand
                else:
                    total_fulfilled_units += curr_stock
                    total_stockout_events += 1
                    curr_stock = 0.0

            daily_inventory_samples.append(curr_stock)
            doc = curr_stock / daily_v if daily_v > 0 else 90.0
            daily_doc_samples.append(min(doc, 90.0))

            # 3. Replenishment decision at end of day
            pipeline_qty = sum(in_transit_orders.values())
            inventory_position = curr_stock + pipeline_qty

            if compute_reorder_trigger(inventory_position, rop):
                roq = compute_reorder_quantity(
                    inventory_position=inventory_position,
                    rop=rop,
                    daily_run_rate=daily_v,
                    target_cover_days=target_cover_days
                )
                if roq > 0:
                    arr_idx = dates.index(current_date) + lead_time_days
                    if arr_idx < len(dates):
                        arr_date = dates[arr_idx]
                        in_transit_orders[arr_date] = in_transit_orders.get(arr_date, 0.0) + roq
                    total_pos_triggered += 1
                    total_units_ordered += roq

    service_level = (total_fulfilled_units / total_demand_units * 100.0) if total_demand_units > 0 else 100.0
    stockout_rate = (total_stockout_events / total_active_days * 100.0) if total_active_days > 0 else 0.0
    mean_inv = float(np.mean(daily_inventory_samples)) if daily_inventory_samples else 0.0
    mean_doc = float(np.mean(daily_doc_samples)) if daily_doc_samples else 0.0
    reorder_freq = (total_pos_triggered / (len(unique_series) * len(dates))) if len(dates) > 0 else 0.0

    return {
        'model_name': model_name,
        'total_demand_units': round(total_demand_units, 1),
        'total_fulfilled_units': round(total_fulfilled_units, 1),
        'service_level_pct': round(service_level, 2),
        'stockout_rate_pct': round(stockout_rate, 2),
        'stockout_events': int(total_stockout_events),
        'total_units_ordered': round(total_units_ordered, 1),
        'po_count': int(total_pos_triggered),
        'average_inventory': round(mean_inv, 1),
        'days_of_cover': round(mean_doc, 1),
        'reorder_frequency_orders_per_day': round(reorder_freq, 4)
    }


def main():
    print("=" * 90)
    print("PHASE 2 MASTER RUNNER: REAL TYPESAFE JEV + XGBOOST ROP EXPERIMENT")
    print("Scenario C: 28 Representative Cases x 62 Backtest Dates = 1,736 Decisions")
    print("Model: ~typesafe/jev-latest (OpenRouter Decisions API)")
    print("=" * 90)

    start_exec = time.time()
    out_dir = Path(OUTPUTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data & Cases
    cases_df, df_eval = load_cohort_and_data()

    # 2. Acquire Real Jev Decisions (Cache-First)
    adapter = RealJevOpenRouterAdapter()
    df_enriched = fetch_all_jev_decisions(df_eval, adapter)

    # 3. Fit Model A: Structured Only
    print("\n[EVALUATION] Training & Evaluating Model A (Structured XGBoost ROP)...")
    preds_a, eval_a, model_a_last = run_cross_fitting(df_enriched, BASE_FEATURE_COLS, n_splits=4)
    df_enriched['pred_rop_model_a'] = preds_a
    print(f"  Model A -> Pinball: {eval_a['mean_pinball_loss']} | Coverage: {eval_a['empirical_coverage_pct']}% | Mean ROP: {eval_a['mean_predicted_rop']}")

    # 4. Fit Model B: Structured + Real Jev
    print("\n[EVALUATION] Training & Evaluating Model B (Real Jev-Enhanced XGBoost ROP)...")
    model_b_features = BASE_FEATURE_COLS + JEV_FEATURE_COLS
    preds_b, eval_b, model_b_last = run_cross_fitting(df_enriched, model_b_features, n_splits=4)
    df_enriched['pred_rop_model_b'] = preds_b
    print(f"  Model B -> Pinball: {eval_b['mean_pinball_loss']} | Coverage: {eval_b['empirical_coverage_pct']}% | Mean ROP: {eval_b['mean_predicted_rop']}")

    # 5. Inventory Simulation
    print("\n[SIMULATION] Running Walk-Forward Discrete-Event Inventory Backtest...")
    sim_a = run_inventory_simulation(df_enriched, 'pred_rop_model_a', "Model A (Baseline Structured)")
    sim_b = run_inventory_simulation(df_enriched, 'pred_rop_model_b', "Model B (Real Jev-Enhanced)")

    # 6. Ablation Study
    print("\n[ABLATION STUDY] Running Feature Ablation Experiments (Zero Extra API Calls)...")
    ablation_configs = {
        'A. Structured Only': BASE_FEATURE_COLS,
        'B. Structured + Noul': BASE_FEATURE_COLS + ['jev_urgent_replenishment_prob'],
        'C. Structured + Score': BASE_FEATURE_COLS + ['jev_replenishment_risk_score'],
        'D. Structured + Choice': BASE_FEATURE_COLS + ['jev_prio_HIGH', 'jev_prio_MEDIUM', 'jev_prio_LOW'],
        'E. Structured + All Jev': model_b_features
    }

    ablation_rows = []
    for cfg_name, cols in ablation_configs.items():
        p, ev, _ = run_cross_fitting(df_enriched, cols, n_splits=4)
        col_name = f'pred_rop_ablation_{len(ablation_rows)}'
        df_enriched[col_name] = p
        sim = run_inventory_simulation(df_enriched, col_name, cfg_name)
        ablation_rows.append({
            'Configuration': cfg_name,
            'Features_Count': len(cols),
            'Pinball_Loss': ev['mean_pinball_loss'],
            'Coverage_Pct': ev['empirical_coverage_pct'],
            'Mean_ROP': ev['mean_predicted_rop'],
            'Service_Level_Pct': sim['service_level_pct'],
            'Stockout_Rate_Pct': sim['stockout_rate_pct'],
            'Stockout_Events': sim['stockout_events'],
            'Avg_Inventory': sim['average_inventory'],
            'Total_Units_Ordered': sim['total_units_ordered'],
            'PO_Count': sim['po_count']
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = out_dir / "real_jev_phase2_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"  Saved ablation results to {ablation_path}")

    # 7. Redundancy & Feature Importance
    print("\n[ANALYSIS] Redundancy & Feature Importance Analysis...")
    imp_scores = model_b_last.feature_importances_
    imp_df = pd.DataFrame({
        'feature': model_b_features,
        'is_jev': [f in JEV_FEATURE_COLS for f in model_b_features],
        'gain': imp_scores
    }).sort_values('gain', ascending=False).reset_index(drop=True)

    # Correlation Matrix
    check_cols = ['jev_urgent_replenishment_prob', 'jev_replenishment_risk_score', 'days_of_cover', 'current_stock', 'v7', 'v14', 'v30', 'cv_30', 'stockout_flag']
    corr_matrix = df_enriched[check_cols].corr()

    # 8. Compile Master Comparison Table (All 13 Metrics)
    print("\n[REPORTING] Compiling 13 Required Operational & Statistical Metrics...")
    metrics_summary = [
        {
            'Metric_Number': 1,
            'Metric_Name': 'Total Demand Units',
            'Baseline_Model_A': sim_a['total_demand_units'],
            'Jev_Enhanced_Model_B': sim_b['total_demand_units'],
            'Absolute_Difference': round(sim_b['total_demand_units'] - sim_a['total_demand_units'], 1),
            'Percentage_Difference': "0.00% (Identical)"
        },
        {
            'Metric_Number': 2,
            'Metric_Name': 'Fulfilled Demand Units',
            'Baseline_Model_A': sim_a['total_fulfilled_units'],
            'Jev_Enhanced_Model_B': sim_b['total_fulfilled_units'],
            'Absolute_Difference': round(sim_b['total_fulfilled_units'] - sim_a['total_fulfilled_units'], 1),
            'Percentage_Difference': f"{(sim_b['total_fulfilled_units'] - sim_a['total_fulfilled_units']) / sim_a['total_fulfilled_units'] * 100:+.2f}%"
        },
        {
            'Metric_Number': 3,
            'Metric_Name': 'Service Level (%)',
            'Baseline_Model_A': sim_a['service_level_pct'],
            'Jev_Enhanced_Model_B': sim_b['service_level_pct'],
            'Absolute_Difference': round(sim_b['service_level_pct'] - sim_a['service_level_pct'], 2),
            'Percentage_Difference': f"{sim_b['service_level_pct'] - sim_a['service_level_pct']:+.2f} pts"
        },
        {
            'Metric_Number': 4,
            'Metric_Name': 'Stockout Rate (%)',
            'Baseline_Model_A': sim_a['stockout_rate_pct'],
            'Jev_Enhanced_Model_B': sim_b['stockout_rate_pct'],
            'Absolute_Difference': round(sim_b['stockout_rate_pct'] - sim_a['stockout_rate_pct'], 2),
            'Percentage_Difference': f"{sim_b['stockout_rate_pct'] - sim_a['stockout_rate_pct']:+.2f} pts"
        },
        {
            'Metric_Number': 5,
            'Metric_Name': 'Stockout Events Count',
            'Baseline_Model_A': sim_a['stockout_events'],
            'Jev_Enhanced_Model_B': sim_b['stockout_events'],
            'Absolute_Difference': sim_b['stockout_events'] - sim_a['stockout_events'],
            'Percentage_Difference': f"{(sim_b['stockout_events'] - sim_a['stockout_events']) / max(1, sim_a['stockout_events']) * 100:+.2f}%"
        },
        {
            'Metric_Number': 6,
            'Metric_Name': 'Total Units Ordered',
            'Baseline_Model_A': sim_a['total_units_ordered'],
            'Jev_Enhanced_Model_B': sim_b['total_units_ordered'],
            'Absolute_Difference': round(sim_b['total_units_ordered'] - sim_a['total_units_ordered'], 1),
            'Percentage_Difference': f"{(sim_b['total_units_ordered'] - sim_a['total_units_ordered']) / sim_a['total_units_ordered'] * 100:+.2f}%"
        },
        {
            'Metric_Number': 7,
            'Metric_Name': 'Purchase Order Count',
            'Baseline_Model_A': sim_a['po_count'],
            'Jev_Enhanced_Model_B': sim_b['po_count'],
            'Absolute_Difference': sim_b['po_count'] - sim_a['po_count'],
            'Percentage_Difference': f"{(sim_b['po_count'] - sim_a['po_count']) / max(1, sim_a['po_count']) * 100:+.2f}%"
        },
        {
            'Metric_Number': 8,
            'Metric_Name': 'Average Inventory Units',
            'Baseline_Model_A': sim_a['average_inventory'],
            'Jev_Enhanced_Model_B': sim_b['average_inventory'],
            'Absolute_Difference': round(sim_b['average_inventory'] - sim_a['average_inventory'], 1),
            'Percentage_Difference': f"{(sim_b['average_inventory'] - sim_a['average_inventory']) / sim_a['average_inventory'] * 100:+.2f}%"
        },
        {
            'Metric_Number': 9,
            'Metric_Name': 'Days of Cover (DoC)',
            'Baseline_Model_A': sim_a['days_of_cover'],
            'Jev_Enhanced_Model_B': sim_b['days_of_cover'],
            'Absolute_Difference': round(sim_b['days_of_cover'] - sim_a['days_of_cover'], 1),
            'Percentage_Difference': f"{(sim_b['days_of_cover'] - sim_a['days_of_cover']) / sim_a['days_of_cover'] * 100:+.2f}%"
        },
        {
            'Metric_Number': 10,
            'Metric_Name': 'Average Predicted ROP',
            'Baseline_Model_A': eval_a['mean_predicted_rop'],
            'Jev_Enhanced_Model_B': eval_b['mean_predicted_rop'],
            'Absolute_Difference': round(eval_b['mean_predicted_rop'] - eval_a['mean_predicted_rop'], 2),
            'Percentage_Difference': f"{(eval_b['mean_predicted_rop'] - eval_a['mean_predicted_rop']) / eval_a['mean_predicted_rop'] * 100:+.2f}%"
        },
        {
            'Metric_Number': 11,
            'Metric_Name': 'Reorder Frequency (Orders/Series/Day)',
            'Baseline_Model_A': sim_a['reorder_frequency_orders_per_day'],
            'Jev_Enhanced_Model_B': sim_b['reorder_frequency_orders_per_day'],
            'Absolute_Difference': round(sim_b['reorder_frequency_orders_per_day'] - sim_a['reorder_frequency_orders_per_day'], 4),
            'Percentage_Difference': f"{(sim_b['reorder_frequency_orders_per_day'] - sim_a['reorder_frequency_orders_per_day']) / sim_a['reorder_frequency_orders_per_day'] * 100:+.2f}%"
        },
        {
            'Metric_Number': 12,
            'Metric_Name': 'Quantile Pinball Loss (Lower is Better)',
            'Baseline_Model_A': eval_a['mean_pinball_loss'],
            'Jev_Enhanced_Model_B': eval_b['mean_pinball_loss'],
            'Absolute_Difference': round(eval_b['mean_pinball_loss'] - eval_a['mean_pinball_loss'], 4),
            'Percentage_Difference': f"{(eval_b['mean_pinball_loss'] - eval_a['mean_pinball_loss']) / eval_a['mean_pinball_loss'] * 100:+.2f}%"
        },
        {
            'Metric_Number': 13,
            'Metric_Name': 'Empirical Service Coverage (%)',
            'Baseline_Model_A': eval_a['empirical_coverage_pct'],
            'Jev_Enhanced_Model_B': eval_b['empirical_coverage_pct'],
            'Absolute_Difference': round(eval_b['empirical_coverage_pct'] - eval_a['empirical_coverage_pct'], 2),
            'Percentage_Difference': f"{eval_b['empirical_coverage_pct'] - eval_a['empirical_coverage_pct']:+.2f} pts"
        }
    ]

    comp_df = pd.DataFrame(metrics_summary)
    comp_path = out_dir / "real_jev_phase2_model_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"  Saved comparison table to {comp_path}")

    # Save detailed JSON metrics
    total_runtime = time.time() - start_exec
    metrics_json = {
        'experiment_name': 'Real TypeSafe Jev + XGBoost ROP Phase 2',
        'execution_date': datetime.datetime.now().isoformat(),
        'scenario': 'Scenario C (Approved)',
        'walk_forward_start': START_DATE,
        'walk_forward_end': END_DATE,
        'num_dates': NUM_DAYS,
        'num_cases': NUM_CASES,
        'total_logical_decisions': len(df_enriched),
        'api_accounting': {
            'model_name': 'typesafe/jev-1.13-20260917',
            'api_calls_made': adapter.api_calls_made,
            'cache_hits': adapter.cache_hits,
            'retries_count': adapter.retries_count,
            'failures_count': adapter.failures_count,
            'input_tokens': adapter.total_input_tokens,
            'output_tokens': adapter.total_output_tokens,
            'total_tokens': adapter.total_input_tokens + adapter.total_output_tokens,
            'total_estimated_cost_usd': round(adapter.total_cost_usd, 5),
            'average_cost_per_decision_usd': round(adapter.total_cost_usd / max(1, len(df_enriched)), 6),
            'runtime_seconds': round(total_runtime, 1)
        },
        'comparison_metrics': metrics_summary,
        'ablation_results': ablation_rows,
        'feature_importance_ranking': imp_df.to_dict(orient='records'),
        'correlation_with_structured_signals': corr_matrix.to_dict()
    }

    json_path = out_dir / "real_jev_phase2_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Saved JSON metrics to {json_path}")

    # Save row-level predictions for full auditability
    save_cols = [
        'date', 'case_id', 'canonical_sku', 'platform_group', 'category', 'stratum',
        'current_stock', 'v14', 'days_of_cover', 'observed_units_sold', 'ltd_target',
        'pred_rop_model_a', 'pred_rop_model_b',
        'jev_urgent_replenishment_prob', 'jev_replenishment_risk_score',
        'jev_prio_HIGH', 'jev_prio_MEDIUM', 'jev_prio_LOW', 'jev_confidence', 'jev_is_cached'
    ]
    detail_path = out_dir / "real_jev_phase2_case_predictions.csv"
    df_enriched[save_cols].to_csv(detail_path, index=False)
    print(f"  Saved detailed row-level predictions to {detail_path}")

    print("\n" + "=" * 90)
    print("PHASE 2 COMPLETED SUCCESSFULLY")
    print(f"Total Runtime: {total_runtime:.1f}s | Est API Cost: ${adapter.total_cost_usd:.5f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
