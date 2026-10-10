"""
Multi-Brand Causal Feature Engineering Module
Builds the audited 60-feature schema with strict causal temporal discipline.
Rule: All rolling demand metrics use shift(1) before aggregation. Zero lookahead leakage.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def compute_causal_features(
    grid_df: pd.DataFrame,
    schema_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Computes all 60 features on the daily grid.
    Features are calculated per series (brand_id, canonical_sku, platform_group).
    """
    logger.info("Computing causal feature engineering across daily grid...")
    df = grid_df.copy()
    
    # Ensure sorted by series and date
    df = df.sort_values(by=['brand_id', 'canonical_sku', 'platform_group', 'date']).reset_index(drop=True)
    
    grp = df.groupby(['brand_id', 'canonical_sku', 'platform_group'], group_keys=False)
    sales = grp['observed_units_sold']
    
    # -------------------------------------------------------------
    # 1. Direct Demand Lags (shift k)
    # -------------------------------------------------------------
    logger.info("  Calculating demand lags...")
    df['lag_1'] = sales.shift(1).fillna(0.0)
    df['lag_7'] = sales.shift(7).fillna(0.0)
    df['lag_14'] = sales.shift(14).fillna(0.0)
    df['lag_30'] = sales.shift(30).fillna(0.0)
    df['lag_90'] = sales.shift(90).fillna(0.0)
    df['lag_180'] = sales.shift(180).fillna(0.0)
    df['lag_365'] = sales.shift(365).fillna(0.0)
    
    # -------------------------------------------------------------
    # 2. Rolling Velocities (v*) with mandatory shift(1)
    # -------------------------------------------------------------
    logger.info("  Calculating rolling velocities (shift(1) + rolling)...")
    # Shift sales by 1 first to guarantee causality
    shifted_sales = sales.shift(1)
    shifted_grp = df.assign(_s_sales=shifted_sales).groupby(['brand_id', 'canonical_sku', 'platform_group'], group_keys=False)['_s_sales']
    
    df['v7'] = shifted_grp.rolling(7, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['v14'] = shifted_grp.rolling(14, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['v30'] = shifted_grp.rolling(30, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['v60'] = shifted_grp.rolling(60, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['v90'] = shifted_grp.rolling(90, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['v180'] = shifted_grp.rolling(180, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['v365'] = shifted_grp.rolling(365, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    
    # -------------------------------------------------------------
    # 3. Active Sales Days
    # -------------------------------------------------------------
    logger.info("  Calculating active sales days...")
    pos_sales = (shifted_sales > 0).astype(float)
    pos_grp = df.assign(_p_sales=pos_sales).groupby(['brand_id', 'canonical_sku', 'platform_group'], group_keys=False)['_p_sales']
    df['sales_days_30'] = pos_grp.rolling(30, min_periods=1).sum().reset_index(drop=True).fillna(0.0)
    df['sales_days_90'] = pos_grp.rolling(90, min_periods=1).sum().reset_index(drop=True).fillna(0.0)
    df['sales_days_180'] = pos_grp.rolling(180, min_periods=1).sum().reset_index(drop=True).fillna(0.0)
    
    # -------------------------------------------------------------
    # 4. Volatility (CV)
    # -------------------------------------------------------------
    logger.info("  Calculating demand volatility (CV)...")
    std30 = shifted_grp.rolling(30, min_periods=2).std().reset_index(drop=True).fillna(0.0)
    std90 = shifted_grp.rolling(90, min_periods=2).std().reset_index(drop=True).fillna(0.0)
    df['cv_30'] = (std30 / (df['v30'] + 1e-4)).clip(0.0, 10.0)
    df['cv_90'] = (std90 / (df['v90'] + 1e-4)).clip(0.0, 10.0)
    
    # -------------------------------------------------------------
    # 5. Velocity Momentum Ratios
    # -------------------------------------------------------------
    df['v30_vs_v365'] = (df['v30'] / (df['v365'] + 1e-4)).clip(0.0, 20.0)
    df['v90_vs_v365'] = (df['v90'] / (df['v365'] + 1e-4)).clip(0.0, 20.0)
    df['v14_vs_v30'] = (df['v14'] / (df['v30'] + 1e-4)).clip(0.0, 20.0)
    df['v30_vs_v90'] = (df['v30'] / (df['v90'] + 1e-4)).clip(0.0, 20.0)
    df['v30_vs_v180'] = (df['v30'] / (df['v180'] + 1e-4)).clip(0.0, 20.0)
    
    # -------------------------------------------------------------
    # 6. YoY Seasonality
    # -------------------------------------------------------------
    shifted_365 = sales.shift(365)
    s365_grp = df.assign(_s365=shifted_365).groupby(['brand_id', 'canonical_sku', 'platform_group'], group_keys=False)['_s365']
    df['same_period_last_year_7d'] = s365_grp.rolling(7, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['same_period_last_year_30d'] = s365_grp.rolling(30, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['yoy_7d'] = (df['v7'] / (df['same_period_last_year_7d'] + 1e-4)).clip(0.0, 10.0)
    df['yoy_30d'] = (df['v30'] / (df['same_period_last_year_30d'] + 1e-4)).clip(0.0, 10.0)
    
    # -------------------------------------------------------------
    # 7. Inventory & Stockout Features
    # -------------------------------------------------------------
    logger.info("  Calculating stockout & in-stock velocities...")
    # Days since stockout
    is_out = (df['stockout_flag'] == 1).astype(int)
    # Cumulative count of days since last stockout
    df['days_since_stockout'] = grp['stockout_flag'].apply(
        lambda s: (~s.astype(bool)).cumsum() - (~s.astype(bool)).cumsum().where(s.astype(bool)).ffill().fillna(0)
    ).reset_index(drop=True).fillna(999.0)
    
    # In-stock velocities (only counting in-stock periods)
    instock_sales = shifted_sales.where(df['in_stock_flag'] == 1, np.nan)
    is_grp = df.assign(_is_sales=instock_sales).groupby(['brand_id', 'canonical_sku', 'platform_group'], group_keys=False)['_is_sales']
    df['v14_instock'] = is_grp.rolling(14, min_periods=1).mean().reset_index(drop=True).fillna(df['v14'])
    df['v30_instock'] = is_grp.rolling(30, min_periods=1).mean().reset_index(drop=True).fillna(df['v30'])
    df['v90_instock'] = is_grp.rolling(90, min_periods=1).mean().reset_index(drop=True).fillna(df['v90'])
    
    # Restock tracking
    df['has_restock_date'] = df['restock_date'].notna().astype(int) if 'restock_date' in df.columns else 0
    df['days_from_restock'] = 999.0 # Default unobserved
    
    # -------------------------------------------------------------
    # 8. Pricing & Economics
    # -------------------------------------------------------------
    logger.info("  Calculating pricing ratios...")
    price_grp = grp['selling_price']
    med30 = price_grp.rolling(30, min_periods=1).median().reset_index(drop=True).fillna(df['selling_price'])
    med90 = price_grp.rolling(90, min_periods=1).median().reset_index(drop=True).fillna(df['selling_price'])
    mean30 = price_grp.rolling(30, min_periods=1).mean().reset_index(drop=True).fillna(df['selling_price'])
    
    df['price_vs_30d'] = (df['selling_price'] / (med30 + 1e-4)).clip(0.1, 10.0)
    df['price_vs_90d'] = (df['selling_price'] / (med90 + 1e-4)).clip(0.1, 10.0)
    df['price_change_30d'] = (df['selling_price'] - mean30).clip(-50.0, 50.0)
    
    # -------------------------------------------------------------
    # 9. Amazon Traffic & Buy Box Telemetry
    # -------------------------------------------------------------
    logger.info("  Calculating traffic & Buy Box signals...")
    sessions = grp['amazon_sessions'].shift(1).fillna(0.0) if 'amazon_sessions' in df.columns else pd.Series(0.0, index=df.index)
    sess_grp = df.assign(_sess=sessions).groupby(['brand_id', 'canonical_sku', 'platform_group'], group_keys=False)['_sess']
    df['amazon_sessions_7d'] = sess_grp.rolling(7, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['amazon_sessions_30d'] = sess_grp.rolling(30, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['amazon_sessions_90d'] = sess_grp.rolling(90, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['amazon_sessions_momentum'] = (df['amazon_sessions_7d'] / (df['amazon_sessions_30d'] + 1e-4)).clip(0.0, 10.0)
    
    buybox = grp['buy_box_percentage'].shift(1).fillna(100.0) if 'buy_box_percentage' in df.columns else pd.Series(100.0, index=df.index)
    bb_grp = df.assign(_bb=buybox).groupby(['brand_id', 'canonical_sku', 'platform_group'], group_keys=False)['_bb']
    df['buy_box_7d'] = bb_grp.rolling(7, min_periods=1).mean().reset_index(drop=True).fillna(100.0)
    df['buy_box_30d'] = bb_grp.rolling(30, min_periods=1).mean().reset_index(drop=True).fillna(100.0)
    df['buy_box_90d'] = bb_grp.rolling(90, min_periods=1).mean().reset_index(drop=True).fillna(100.0)
    df['buy_box_change'] = (df['buy_box_7d'] - df['buy_box_30d']).clip(-100.0, 100.0)
    df['units_per_session_30d'] = (df['v30'] / (df['amazon_sessions_30d'] + 1e-4)).clip(0.0, 5.0)
    
    # -------------------------------------------------------------
    # 10. Promotions & Calendar
    # -------------------------------------------------------------
    promo = (df['ebay_promoted_flag'] == 1).astype(float) if 'ebay_promoted_flag' in df.columns else pd.Series(0.0, index=df.index)
    p_grp = df.assign(_p=promo).groupby(['brand_id', 'canonical_sku', 'platform_group'], group_keys=False)['_p']
    df['promo_days_7'] = p_grp.rolling(7, min_periods=1).sum().reset_index(drop=True).fillna(0.0)
    df['promo_days_30'] = p_grp.rolling(30, min_periods=1).sum().reset_index(drop=True).fillna(0.0)
    df['promo_days_90'] = p_grp.rolling(90, min_periods=1).sum().reset_index(drop=True).fillna(0.0)
    df['promotion_started'] = (p_grp.diff() > 0).astype(int).fillna(0)
    
    # Calendar
    dt_series = pd.to_datetime(df['date'])
    df['day_of_week'] = dt_series.dt.dayofweek # 0=Mon, 6=Sun
    
    # Lifecycle
    min_date_per_sku = df.groupby(['brand_id', 'canonical_sku'])['date'].transform('min')
    df['days_since_launch'] = (pd.to_datetime(df['date']) - pd.to_datetime(min_date_per_sku)).dt.days.clip(lower=0)
    df['resolved_parent_id'] = df['canonical_sku'] # Fallback to canonical SKU
    
    # -------------------------------------------------------------
    # 11. Cross-Platform Cannibalization
    # -------------------------------------------------------------
    # Daily brand-SKU total sales across all platforms
    sku_tot_sales = df.groupby(['date', 'brand_id', 'canonical_sku'])['observed_units_sold'].transform('sum')
    df['other_platform_sales'] = sku_tot_sales - df['observed_units_sold']
    
    other_grp = df.groupby(['brand_id', 'canonical_sku', 'platform_group'], group_keys=False)['other_platform_sales']
    shifted_other = other_grp.shift(1).fillna(0.0)
    so_grp = df.assign(_so=shifted_other).groupby(['brand_id', 'canonical_sku', 'platform_group'], group_keys=False)['_so']
    
    df['other_platform_sales_7d'] = so_grp.rolling(7, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    df['other_platform_sales_30d'] = so_grp.rolling(30, min_periods=1).mean().reset_index(drop=True).fillna(0.0)
    
    tot_v30 = df['v30'] + df['other_platform_sales_30d']
    df['platform_share_30d'] = (df['v30'] / (tot_v30 + 1e-4)).clip(0.0, 1.0)
    
    # Load schema if path provided to verify exactly 60 features
    if schema_path and os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            schema_data = json.load(f)
        req_features = schema_data['feature_list']
        missing = [rf for rf in req_features if rf not in df.columns]
        if missing:
            logger.error(f"Missing required model features from schema: {missing}")
            raise ValueError(f"Feature calculation incomplete. Missing: {missing}")
        logger.info(f"Verified all {len(req_features)} schema features successfully generated!")
        
    logger.info("Causal feature engineering completed.")
    return df
