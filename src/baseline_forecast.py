"""
BASELINE FORECAST MODULE
========================
Calculates the long-term organic demand anchor using multi-quarter velocity signals.
Protects procurement and planning against temporary promotional flash-sale noise.
"""

def compute_baseline_forecast(sku_features, forecast_horizon_days=11):
    """
    Computes the organic baseline forecast scaled dynamically to N forecast days.
    
    Args:
        sku_features (dict): Extracted feature dictionary from feature_engineering.py.
        forecast_horizon_days (int): Number of days in the forward forecast window.
        
    Returns:
        int: Units predicted by the baseline organic anchor model.
    """
    if sku_features['is_empty'] or sku_features['current_stock'] == 0:
        return 0
        
    v_instock_365 = sku_features['velocity_instock_365d']
    v_180         = sku_features['velocity_180d']
    v_90          = sku_features['velocity_90d']
    v_30          = sku_features['velocity_30d']
    in_stock_days = sku_features['in_stock_days']
    current_stock = sku_features['current_stock']
    
    # Anchor weighting: compensates if the product suffered historic stockouts
    if in_stock_days < 200 and in_stock_days > 20:
        daily_baseline_rate = (0.45 * v_instock_365 + 
                               0.25 * v_180 + 
                               0.20 * v_90 + 
                               0.10 * v_30)
    else:
        daily_baseline_rate = (0.35 * v_instock_365 + 
                               0.25 * v_180 + 
                               0.25 * v_90 + 
                               0.15 * v_30)
                               
    baseline_units = int(round(daily_baseline_rate * forecast_horizon_days))
    
    # Enforce physical inventory ceiling
    return min(baseline_units, current_stock)
