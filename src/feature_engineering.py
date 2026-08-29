"""
FEATURE ENGINEERING MODULE
==========================
Calculates multi-horizon sales velocities, demand volatility (CV),
relative price adjustments, and category-level macro momentum signals.
"""
import pandas as pd
import numpy as np

def calculate_category_momentum(training_dataframe):
    """
    Computes macro momentum at the category level (14-day velocity vs 60-day velocity).
    
    Returns:
        dict: Mapping of category name -> momentum multiplier [0.85, 1.25].
    """
    category_daily = training_dataframe.groupby(['category', 'date'])['total_sales'].sum().reset_index()
    category_momentum_map = {}
    
    for category in training_dataframe['category'].dropna().unique():
        df_cat = category_daily[category_daily['category'] == category].sort_values('date')
        if len(df_cat) == 0:
            category_momentum_map[category] = 1.0
            continue
            
        last_date = df_cat['date'].max()
        sales_14d = df_cat[df_cat['date'] >= last_date - pd.Timedelta(days=14)]['total_sales'].sum()
        sales_60d = df_cat[df_cat['date'] >= last_date - pd.Timedelta(days=60)]['total_sales'].sum()
        
        velocity_14d = sales_14d / 14.0
        days_60_actual = max(float((last_date - (last_date - pd.Timedelta(days=60))).days), 1.0)
        velocity_60d = sales_60d / days_60_actual
        
        raw_momentum = (velocity_14d / velocity_60d) if velocity_60d > 0 else 1.0
        category_momentum_map[category] = float(np.clip(raw_momentum, 0.85, 1.25))
        
    return category_momentum_map

def extract_sku_features(training_dataframe, sku, category_momentum_map=None):
    """
    Extracts all grounded historical signals for a single SKU up to the training cutoff.
    
    Returns:
        dict: Feature dictionary containing all velocity horizons, CV, and stock metadata.
    """
    if category_momentum_map is None:
        category_momentum_map = {}
        
    df_sku = training_dataframe[training_dataframe['sku'] == sku].sort_values('date')
    
    if len(df_sku) == 0:
        return {
            'sku': sku,
            'is_empty': True,
            'annual_sales': 0.0,
            'current_stock': 0,
            'selling_price': 4.99,
            'category': 'Cosmetics General',
            'category_momentum': 1.0,
            'weekly_cv': 99.0,
            'zero_days_pct': 100.0,
            'in_stock_rate': 0.0,
            'in_stock_days': 0,
            'velocity_7d': 0.0,
            'velocity_14d': 0.0,
            'velocity_20d': 0.0,
            'velocity_30d': 0.0,
            'velocity_90d': 0.0,
            'velocity_180d': 0.0,
            'velocity_365d': 0.0,
            'velocity_instock_365d': 0.0,
            'price_adjustment': 1.0,
            'is_flash_spike': False,
            'is_sustained_trend': False
        }
        
    total_sales   = float(df_sku['total_sales'].sum())
    current_stock = int(df_sku['current_stock'].iloc[-1])
    selling_price = float(df_sku['selling_price'].iloc[-1])
    category      = df_sku['category'].iloc[-1] if 'category' in df_sku.columns else 'Cosmetics General'
    cat_momentum  = category_momentum_map.get(category, 1.0)
    
    last_date = df_sku['date'].max()
    slice_7d   = df_sku[df_sku['date'] >= last_date - pd.Timedelta(days=7)]
    slice_14d  = df_sku[df_sku['date'] >= last_date - pd.Timedelta(days=14)]
    slice_20d  = df_sku[df_sku['date'] >= last_date - pd.Timedelta(days=20)]
    slice_30d  = df_sku[df_sku['date'] >= last_date - pd.Timedelta(days=30)]
    slice_90d  = df_sku[df_sku['date'] >= last_date - pd.Timedelta(days=90)]
    slice_180d = df_sku[df_sku['date'] >= last_date - pd.Timedelta(days=180)]
    
    velocity_7d   = float(slice_7d['total_sales'].sum()) / 7.0
    velocity_14d  = float(slice_14d['total_sales'].sum()) / 14.0
    velocity_20d  = float(slice_20d['total_sales'].sum()) / 20.0
    velocity_30d  = float(slice_30d['total_sales'].sum()) / 30.0
    velocity_90d  = float(slice_90d['total_sales'].sum()) / 90.0
    velocity_180d = float(slice_180d['total_sales'].sum()) / 180.0 if len(slice_180d) > 0 else velocity_90d
    velocity_365d = total_sales / max(float(len(df_sku)), 1.0)
    
    in_stock_mask = df_sku['in_stock_flag'] == 1
    in_stock_days = int(in_stock_mask.sum())
    velocity_instock_365d = (float(df_sku[in_stock_mask]['total_sales'].sum()) / float(in_stock_days)) if in_stock_days > 10 else velocity_365d
    in_stock_rate = float(in_stock_days) / max(float(len(df_sku)), 1.0)
    zero_days_pct = float((df_sku['total_sales'] == 0).mean()) * 100.0
    
    # Weekly Demand Volatility (CV = standard_deviation / mean)
    df_weekly = df_sku.set_index('date').resample('W-SUN')['total_sales'].sum()
    weekly_mean = float(df_weekly.mean())
    weekly_std  = float(df_weekly.std())
    weekly_cv   = (weekly_std / weekly_mean) if weekly_mean > 0 else 99.0
    
    # Relative Price Ratio (Current Selling Price vs 90-Day Moving Average)
    price_90d_avg = float(slice_90d['selling_price'].mean()) if len(slice_90d) > 0 and slice_90d['selling_price'].mean() > 0 else selling_price
    relative_price = selling_price / price_90d_avg
    if relative_price > 1.04:
        price_adjustment = max(0.50, 1.0 - 1.5 * (relative_price - 1.0))
    elif relative_price < 0.96:
        price_adjustment = min(1.30, 1.0 + 1.2 * (1.0 - relative_price))
    else:
        price_adjustment = 1.0
        
    # Spike Persistence Index (SPI): checks if 7d velocity is maintaining pace or collapsing vs 14d
    spi = (velocity_7d - velocity_14d) / max(velocity_30d, 0.5)
    
    # Behavioral state flags
    is_flash_spike = (weekly_cv > 1.25) or (velocity_14d > 1.5 * velocity_30d and velocity_7d < 0.65 * velocity_14d) or (spi < -0.50 and velocity_14d > 5.0)
    is_sustained_trend = (velocity_30d > 1.10 * velocity_90d) and (velocity_14d >= 0.85 * velocity_30d) and (weekly_cv <= 1.15) and (total_sales >= 1500)
    
    return {
        'sku': sku,
        'is_empty': (current_stock == 0 or total_sales == 0),
        'annual_sales': total_sales,
        'current_stock': current_stock,
        'selling_price': selling_price,
        'category': category,
        'category_momentum': cat_momentum,
        'weekly_cv': weekly_cv,
        'zero_days_pct': zero_days_pct,
        'in_stock_rate': in_stock_rate,
        'in_stock_days': in_stock_days,
        'velocity_7d': velocity_7d,
        'velocity_14d': velocity_14d,
        'velocity_20d': velocity_20d,
        'velocity_30d': velocity_30d,
        'velocity_90d': velocity_90d,
        'velocity_180d': velocity_180d,
        'velocity_365d': velocity_365d,
        'velocity_instock_365d': velocity_instock_365d,
        'price_adjustment': price_adjustment,
        'is_flash_spike': is_flash_spike,
        'is_sustained_trend': is_sustained_trend
    }
