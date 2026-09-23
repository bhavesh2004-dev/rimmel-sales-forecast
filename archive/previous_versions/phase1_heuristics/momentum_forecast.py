"""
MOMENTUM FORECAST MODULE
========================
Calculates the fast-reacting recent trend forecast using 7-day, 14-day,
and MTD velocity signals along with category-level macro momentum.
"""

def compute_momentum_forecast(sku_features, forecast_horizon_days=11):
    """
    Computes the trend-based momentum forecast scaled dynamically to N forecast days.
    
    Args:
        sku_features (dict): Extracted feature dictionary from feature_engineering.py.
        forecast_horizon_days (int): Number of days in the forward forecast window.
        
    Returns:
        int: Units predicted by the momentum trend model.
    """
    if sku_features['is_empty'] or sku_features['current_stock'] == 0:
        return 0
        
    v_7           = sku_features['velocity_7d']
    v_14          = sku_features['velocity_14d']
    v_20          = sku_features['velocity_20d']
    v_30          = sku_features['velocity_30d']
    v_90          = sku_features['velocity_90d']
    annual_sales  = sku_features['annual_sales']
    weekly_cv     = sku_features['weekly_cv']
    cat_momentum  = sku_features['category_momentum']
    price_adj     = sku_features['price_adjustment']
    current_stock = sku_features['current_stock']
    
    momentum_ratio = (v_20 / max(v_90, 0.1)) if v_90 > 0 else 1.0
    
    # 1. Powerhouse Tier A Summer Surge
    if (annual_sales >= 10000) and (momentum_ratio >= 1.15):
        daily_momentum_rate = max(v_20, v_30) * cat_momentum * 1.15
        
    # 2. Breakout Short-Term Acceleration
    elif (v_14 > 1.40 * v_30) and (v_30 >= 3.0):
        daily_momentum_rate = v_14 * price_adj
        
    # 3. Post-Spike Sharp Deceleration
    elif (v_7 < 0.60 * v_14) and (v_14 > 5.0):
        daily_momentum_rate = (0.70 * v_7 + 0.30 * v_14) * price_adj
        
    # 4. Low-Volatility Stable Core Line
    elif weekly_cv <= 0.85 and annual_sales >= 500:
        daily_momentum_rate = (0.60 * v_14 + 0.40 * v_30) * price_adj
        
    # 5. Standard Multi-Horizon Trend
    else:
        daily_momentum_rate = (0.50 * v_14 + 0.35 * v_30 + 0.15 * v_90) * price_adj
        
    momentum_units = int(round(daily_momentum_rate * forecast_horizon_days))
    
    # Enforce physical warehouse stock ceiling
    return min(momentum_units, current_stock)
