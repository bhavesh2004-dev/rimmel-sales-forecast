"""
BUSINESS RISK & STATUS MODULE
=============================
Assigns practical, non-technical business status and inventory risk labels
to guide procurement and supply chain decisions.
"""

def classify_risk_and_status(sku_features, baseline_forecast, momentum_forecast, adaptive_forecast, confidence_level):
    """
    Determines business operational status and inventory risk label.
    
    Args:
        sku_features (dict): Extracted feature dictionary.
        baseline_forecast (int): Units predicted by baseline model.
        momentum_forecast (int): Units predicted by momentum model.
        adaptive_forecast (int): Recommended forecast units.
        confidence_level (str): 'HIGH', 'MEDIUM', or 'LOW'.
        
    Returns:
        str: Business status label (e.g. 'NORMAL', 'DEMAND INCREASING', 'STOCK RISK', etc.).
    """
    current_stock      = sku_features['current_stock']
    annual_sales       = sku_features['annual_sales']
    weekly_cv          = sku_features['weekly_cv']
    velocity_14d       = sku_features['velocity_14d']
    velocity_30d       = sku_features['velocity_30d']
    velocity_90d       = sku_features['velocity_90d']
    is_sustained_trend = sku_features['is_sustained_trend']
    
    if current_stock == 0 and annual_sales > 0:
        return "STOCK RISK"
    elif annual_sales == 0 or sku_features['is_empty']:
        return "DEAD / NEAR-DEAD"
    elif annual_sales >= 10000:
        return "HIGH DEMAND"
    elif is_sustained_trend or (velocity_14d > 1.25 * velocity_30d and velocity_30d >= 5.0):
        return "DEMAND INCREASING"
    elif (velocity_14d < 0.70 * velocity_30d and velocity_30d >= 5.0) or (velocity_30d < 0.70 * velocity_90d and velocity_90d >= 5.0):
        return "DEMAND DECLINING"
    elif weekly_cv > 1.30:
        return "VOLATILE"
    elif annual_sales < 150:
        return "LOW DEMAND"
    elif confidence_level == "LOW":
        return "LOW CONFIDENCE"
    elif adaptive_forecast > current_stock * 0.80:
        return "STOCK RISK"
    else:
        return "NORMAL"
