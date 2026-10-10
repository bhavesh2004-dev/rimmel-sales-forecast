"""
Multi-Brand Forward Forecasting Module
Executes true day-by-day forward simulation across the 10-day operational horizon.
Rule: Zero prediction * 10 shortcuts. Day-of-week calendar dynamics dynamically updated.
"""

import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def generate_forward_simulation(
    model: Any,
    df_features: pd.DataFrame,
    config: Dict[str, Any],
    feature_schema: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes a 10-day day-by-day forward simulation from anchor date T to T+10.
    Dynamically updates calendar date, day_of_week, and applies Exp6 calibration.
    
    Returns:
    - df_daily: Daily forecasts (date x brand x canonical_sku x platform_group)
    - df_10d_sku: 10-day SKU summary aggregated across platforms
    """
    timeline = config['timeline']
    calib_cfg = config.get('calibration', {})
    feature_list = feature_schema['feature_list']
    cat_features = feature_schema['categorical_features']
    
    start_date = timeline['forward_forecast_start']
    end_date = timeline['forward_forecast_end']
    horizon_days = timeline.get('forecast_horizon_days', 10)
    
    # Anchor state is the latest available date before forward horizon
    anchor_date = pd.to_datetime(start_date) - pd.Timedelta(days=1)
    anchor_str = anchor_date.strftime('%Y-%m-%d')
    logger.info(f"Generating 10-day forward simulation from {start_date} to {end_date} (Anchor: {anchor_str})...")
    
    # Extract anchor state
    anchor_df = df_features[df_features['date'] == anchor_str].copy()
    if anchor_df.empty:
        # Fall back to latest available date
        latest_date = df_features['date'].max()
        logger.warning(f"No records found on exact anchor date {anchor_str}. Falling back to latest date: {latest_date}")
        anchor_df = df_features[df_features['date'] == latest_date].copy()
        
    logger.info(f"Anchor catalog state contains {len(anchor_df):,} active series.")
    
    forward_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_records = []
    
    alpha = calib_cfg.get('zero_demand_alpha', 0.10)
    beta = calib_cfg.get('stockout_beta', 0.10)
    
    # Step day-by-day through the forward horizon
    for f_dt in forward_dates:
        f_str = f_dt.strftime('%Y-%m-%d')
        dow = f_dt.dayofweek
        
        sim_df = anchor_df.copy()
        sim_df['date'] = f_str
        sim_df['day_of_week'] = dow
        
        # Prepare feature matrix
        X_sim = sim_df[feature_list].copy()
        for col in cat_features:
            if col in X_sim.columns:
                X_sim[col] = X_sim[col].astype('category')
                
        # Raw Model Prediction
        raw_pred = np.clip(model.predict(X_sim), 0, None)
        
        # Exp6 Calibration
        z_mask = (sim_df['v7'] == 0) & (sim_df['v14'] == 0) & (sim_df['v30'] == 0)
        stk_mask = (sim_df['in_stock_flag'] == 0)
        
        calib_pred = raw_pred.copy()
        calib_pred[z_mask.values] *= alpha
        calib_pred[stk_mask.values] *= beta
        
        sim_df['raw_daily_prediction'] = raw_pred
        sim_df['calib_daily_prediction'] = calib_pred
        daily_records.append(sim_df)
        
    df_daily = pd.concat(daily_records, ignore_index=True)
    logger.info(f"Generated {len(df_daily):,} daily forward prediction rows across {len(forward_dates)} days.")
    
    # Ensure product description columns exist
    if 'product_title' not in df_daily.columns:
        df_daily['product_title'] = df_daily.get('category', df_daily['canonical_sku'])
    if 'category' not in df_daily.columns:
        df_daily['category'] = df_daily.get('product_title', df_daily['canonical_sku'])
        
    # Aggregate to 10-day SKU x Platform level
    series_agg = df_daily.groupby(['brand_id', 'canonical_sku', 'product_title', 'category', 'platform_group']).agg(
        expected_demand_10d=('calib_daily_prediction', 'sum'),
        current_stock=('current_stock', 'first'),
        in_stock_flag=('in_stock_flag', 'first'),
        v7=('v7', 'first'),
        v14=('v14', 'first'),
        v30=('v30', 'first'),
        v90=('v90', 'first'),
        cv_30=('cv_30', 'first')
    ).reset_index()
    
    # Deterministic physical unit rounding
    series_agg['recommended_10d_units'] = np.round(series_agg['expected_demand_10d']).astype(int)
    
    # Pivot platforms into Amazon, eBay, Website, Other by SKU entity
    plat_pivot = series_agg.pivot_table(
        index=['brand_id', 'canonical_sku'],
        columns='platform_group',
        values='recommended_10d_units',
        fill_value=0
    ).reset_index()
    
    # Ensure all 4 platforms exist
    for p in ['Amazon', 'eBay', 'Website', 'Other']:
        if p not in plat_pivot.columns:
            plat_pivot[p] = 0
            
    # Additive Channel Law: Total = Amazon + eBay + Website + Other
    plat_pivot['Total Predicted'] = (
        plat_pivot['Amazon'] + plat_pivot['eBay'] + plat_pivot['Website'] + plat_pivot['Other']
    )
    
    # Attach SKU metadata safely without dropping NaNs
    sku_meta = series_agg.groupby(['brand_id', 'canonical_sku']).agg(
        product_title=('product_title', 'first'),
        category=('category', 'first'),
        current_stock=('current_stock', 'first'),
        in_stock_flag=('in_stock_flag', 'first'),
        v7=('v7', 'first'),
        v14=('v14', 'first'),
        v30=('v30', 'first'),
        v90=('v90', 'first'),
        cv_30=('cv_30', lambda s: float(s.fillna(0.0).iloc[0]))
    ).reset_index()
    
    plat_pivot = pd.merge(sku_meta, plat_pivot, on=['brand_id', 'canonical_sku'], how='left')
    plat_pivot['Forecast Period'] = f"{start_date} to {end_date}"
    logger.info(f"Completed 10-day forward SKU platform aggregation for {len(plat_pivot):,} SKUs.")
    return df_daily, plat_pivot
