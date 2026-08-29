"""
CONFIDENCE CLASSIFICATION MODULE
=================================
Assigns evidence-based confidence levels (HIGH / MEDIUM / LOW)
based on historical demand stability, stock availability, and model agreement.
"""

def classify_confidence(sku_features, baseline_forecast, momentum_forecast):
    """
    Evaluates historical evidence strength to determine forecast confidence.
    
    Args:
        sku_features (dict): Extracted feature dictionary.
        baseline_forecast (int): Units predicted by baseline model.
        momentum_forecast (int): Units predicted by momentum model.
        
    Returns:
        str: 'HIGH', 'MEDIUM', or 'LOW'.
    """
    if sku_features['is_empty'] or sku_features['current_stock'] == 0:
        return "LOW"
        
    annual_sales  = sku_features['annual_sales']
    weekly_cv     = sku_features['weekly_cv']
    in_stock_rate = sku_features['in_stock_rate']
    
    momentum_baseline_ratio = momentum_forecast / max(baseline_forecast, 1)
    is_aligned = (0.80 <= momentum_baseline_ratio <= 1.25)
    
    # 🟢 HIGH: Sufficient historical sales, low volatility, high shelf availability, aligned forecasts
    if annual_sales >= 1000 and weekly_cv <= 0.88 and is_aligned and in_stock_rate >= 0.60:
        return "HIGH"
        
    # 🟡 MEDIUM: Usable volume, moderate volatility, or active growth trends
    elif annual_sales >= 300 and weekly_cv <= 1.25 and (0.60 <= momentum_baseline_ratio <= 1.60):
        return "MEDIUM"
        
    # 🔴 LOW: Intermittent, low volume, extreme volatility, or large model disagreement
    else:
        return "LOW"
