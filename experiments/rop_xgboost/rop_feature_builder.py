"""
Feature Builder for ROP Experiment
===================================
Constructs isolated training and backtesting datasets for the ROP experiment.
Reads from data/rimmel_clean.db strictly as read-only input.
Does NOT modify any production database tables or existing features.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from experiments.rop_xgboost.config import (
    DB_PATH, DEFAULT_LEAD_TIME_DAYS, TRAIN_START_DATE, TRAIN_END_DATE,
    BACKTEST_START_DATE, BACKTEST_END_DATE
)
from experiments.rop_xgboost.jev_adapter import (
    DeterministicJevProvider, JevContextInput
)


BASE_FEATURE_COLS = [
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'v7', 'v14', 'v30', 'v90', 'v14_vs_v30',
    'cv_30', 'sales_days_30',
    'current_stock', 'in_stock_flag', 'days_since_stockout',
    'selling_price', 'promo_days_30', 'amazon_sessions_momentum', 'buy_box_7d',
    'day_of_week', 'is_weekend', 'month'
]

JEV_FEATURE_COLS = [
    'jev_replenishment_urgency',
    'jev_promo_demand_lift',
    'jev_stockout_severity_penalty',
    'jev_channel_priority_score',
    'jev_confidence'
]


def load_raw_rop_dataset(lead_time_days: int = DEFAULT_LEAD_TIME_DAYS) -> pd.DataFrame:
    """
    Loads data from ml_features_zero up to 2026-08-31 and constructs
    the cumulative lead-time demand target (LTD).
    """
    print(f"[ROP DATASET] Querying features from SQLite: {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    # Read relevant columns to keep memory light and fast
    query = """
        SELECT 
            date, canonical_sku, platform_group, category, resolved_parent_id,
            observed_units_sold, model_units_sold,
            lag_1, lag_7, lag_14, lag_30,
            v7, v14, v30, v90, v14_vs_v30,
            cv_30, sales_days_30,
            current_stock, in_stock_flag, days_since_stockout,
            selling_price, promo_days_30, amazon_sessions_momentum, buy_box_7d,
            day_of_week, is_weekend
        FROM ml_features_zero
        WHERE date >= '2025-08-01' AND date <= '2026-08-31'
        ORDER BY canonical_sku, platform_group, date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Loaded {len(df):,} records across 396 calendar days.")

    df['date_dt'] = pd.to_datetime(df['date'])
    df['month'] = df['date_dt'].dt.month

    # Construct the Lead Time Demand (LTD) Target: sum of actual sales in next L days
    print(f"[ROP DATASET] Computing {lead_time_days}-day forward Lead Time Demand target...")
    
    # Vectorized forward rolling sum per series
    # Target for date t is the sum of observed_units_sold from t+1 to t+L
    def compute_forward_ltd(series_df):
        s = series_df['observed_units_sold']
        fwd_sum = s.iloc[::-1].rolling(lead_time_days, min_periods=1).sum().iloc[::-1].shift(-1).fillna(0.0)
        series_df['ltd_target'] = fwd_sum.values.astype(np.float32)
        return series_df

    df = df.groupby(['canonical_sku', 'platform_group'], group_keys=False, observed=True).apply(compute_forward_ltd)
    
    # Fill nulls safely
    df['current_stock'] = df['current_stock'].fillna(0.0)
    df['selling_price'] = df['selling_price'].fillna(df['selling_price'].median())
    df['amazon_sessions_momentum'] = df['amazon_sessions_momentum'].fillna(1.0)
    df['buy_box_7d'] = df['buy_box_7d'].fillna(0.0)
    df['cv_30'] = df['cv_30'].fillna(0.0)
    df['v14_vs_v30'] = df['v14_vs_v30'].fillna(1.0)
    df['days_since_stockout'] = df['days_since_stockout'].fillna(999.0)

    return df


def enrich_with_jev_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates TypeSafe AI Jev structured context features for each observation.
    Vectorized for high performance on large historical series.
    """
    print("[JEV ENRICHMENT] Applying TypeSafe AI Jev Context Adapter (vectorized)...")
    
    # 1. Replenishment Urgency
    v_rate = np.maximum(df['v14'].values, 0.01)
    stock = df['current_stock'].values
    doc = np.where(stock > 0, stock / v_rate, 0.0)
    
    urgency = np.where(
        stock <= 0,
        1.0,
        np.where(
            doc <= 7.0,
            np.clip(1.0 - (doc / 14.0), 0.5, 0.95),
            np.where(
                doc <= 14.0,
                np.clip(0.5 - ((doc - 7.0) / 28.0), 0.25, 0.5),
                np.clip(0.2 - ((doc - 14.0) / 100.0), 0.05, 0.2)
            )
        )
    )

    # 2. Promo Demand Lift Risk
    promo_lift = np.zeros(len(df), dtype=np.float32)
    promo_lift += np.where(df['promo_days_30'].values > 0, 0.35, 0.0)
    promo_lift += np.where(
        df['amazon_sessions_momentum'].values > 1.25,
        0.40,
        np.where(df['amazon_sessions_momentum'].values > 1.05, 0.20, 0.0)
    )
    promo_lift += np.where(df['buy_box_7d'].values > 85.0, 0.15, 0.0)
    promo_lift = np.clip(promo_lift, 0.0, 1.0)

    # 3. Stockout Severity Penalty
    revenue_rate = df['selling_price'].values * v_rate
    severity = np.where(revenue_rate > 50.0, 2.5, np.where(revenue_rate > 15.0, 1.8, 1.1))

    # 4. Channel Priority Score
    p_map = {'Amazon': 0.90, 'eBay': 0.70, 'Website': 0.50, 'Other': 0.35}
    channel_prio = df['platform_group'].map(p_map).fillna(0.40).values.astype(np.float32)

    # 5. Model Confidence
    confidence = np.where(df['cv_30'].values <= 1.0, 0.92, 0.75).astype(np.float32)

    df['jev_replenishment_urgency'] = np.round(urgency, 4)
    df['jev_promo_demand_lift'] = np.round(promo_lift, 4)
    df['jev_stockout_severity_penalty'] = np.round(severity, 4)
    df['jev_channel_priority_score'] = np.round(channel_prio, 4)
    df['jev_confidence'] = np.round(confidence, 4)

    print(f"  Enriched with {len(JEV_FEATURE_COLS)} Jev typed context features.")
    return df


def prepare_rop_datasets(lead_time_days: int = DEFAULT_LEAD_TIME_DAYS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares the complete training and backtest sets with both baseline
    and Jev-augmented feature representations.
    """
    df = load_raw_rop_dataset(lead_time_days=lead_time_days)
    df = enrich_with_jev_context(df)

    # Encode categorical platform_group
    df['platform_cat'] = df['platform_group'].astype('category').cat.codes

    # Split train and backtest partitions
    train_mask = (df['date'] >= TRAIN_START_DATE) & (df['date'] <= TRAIN_END_DATE)
    backtest_mask = (df['date'] >= BACKTEST_START_DATE) & (df['date'] <= BACKTEST_END_DATE)

    df_train = df[train_mask].copy()
    df_backtest = df[backtest_mask].copy()

    print(f"  Training set: {len(df_train):,} rows ({TRAIN_START_DATE} to {TRAIN_END_DATE})")
    print(f"  Backtest set: {len(df_backtest):,} rows ({BACKTEST_START_DATE} to {BACKTEST_END_DATE})")
    
    return df_train, df_backtest
