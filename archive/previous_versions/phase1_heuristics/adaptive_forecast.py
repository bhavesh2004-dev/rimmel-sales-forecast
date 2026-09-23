"""
ADAPTIVE FORECAST MODULE
========================
Combines Baseline and Momentum forecasts using evidence-based stability
and persistence weights derived strictly from pre-cutoff signals.
"""

def compute_adaptive_forecast(sku_features, baseline_forecast, momentum_forecast, forecast_horizon_days=11):
    """
    Computes the recommended adaptive forecast combining Baseline and Momentum.
    
    Args:
        sku_features (dict): Feature dictionary.
        baseline_forecast (int): Units predicted by baseline model.
        momentum_forecast (int): Units predicted by momentum model.
        forecast_horizon_days (int): Number of days in the forward forecast window.
        
    Returns:
        tuple: (recommended_forecast_units: int, momentum_weight: float)
    """
    if sku_features['is_empty'] or sku_features['current_stock'] == 0:
        return 0, 0.0
        
    annual_sales       = sku_features['annual_sales']
    weekly_cv          = sku_features['weekly_cv']
    is_flash_spike     = sku_features['is_flash_spike']
    is_sustained_trend = sku_features['is_sustained_trend']
    current_stock      = sku_features['current_stock']
    
    # Evidence-based dynamic weighting
    if is_flash_spike:
        momentum_weight = 0.15  # Anchor heavily to Baseline (85%)
    elif is_sustained_trend:
        momentum_weight = 0.90  # Trust Momentum (90%)
    elif annual_sales >= 1500 and weekly_cv <= 0.85:
        momentum_weight = 0.75  # High-volume stable staple
    elif annual_sales >= 1000:
        momentum_weight = 0.65  # Stable volume with moderate spikes
    elif weekly_cv > 1.30 and annual_sales >= 150:
        momentum_weight = 0.25  # Volatile / promo-driven line
    else:
        momentum_weight = 0.35  # Slow mover / intermittent baseline anchor
        
    raw_adaptive_units = int(round(momentum_weight * momentum_forecast + (1.0 - momentum_weight) * baseline_forecast))
    
    # Enforce warehouse stock ceiling
    recommended_forecast = min(raw_adaptive_units, current_stock)
    
    return recommended_forecast, round(momentum_weight, 2)
