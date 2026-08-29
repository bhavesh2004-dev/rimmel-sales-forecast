"""
INVENTORY INSIGHTS & ACTION MODULE
==================================
Calculates days of inventory coverage, identifies dead/stuck stock,
and generates actionable liquidation and reorder recommendations.
"""

def compute_inventory_metrics(sku_features, recommended_forecast, forecast_horizon_days=11, risk_status="NORMAL"):
    """
    Computes days of inventory and health category for supply chain planning.
    
    Args:
        sku_features (dict): Extracted feature dictionary.
        recommended_forecast (int): Recommended forecast units over the horizon.
        forecast_horizon_days (int): Number of days in forecast period.
        risk_status (str): Current business risk status.
        
    Returns:
        dict: Inventory metrics including days of inventory, health status, and action text.
    """
    current_stock = sku_features['current_stock']
    annual_sales  = sku_features['annual_sales']
    velocity_30d  = sku_features['velocity_30d']
    selling_price = sku_features['selling_price']
    
    recommended_daily_demand = recommended_forecast / float(max(forecast_horizon_days, 1))
    
    if recommended_daily_demand > 0:
        days_of_inventory_val = current_stock / recommended_daily_demand
        days_of_inventory_str = f"{int(round(days_of_inventory_val)):,} days"
    else:
        days_of_inventory_val = 9999.0 if current_stock > 0 else 0.0
        days_of_inventory_str = "N/A – No recent demand" if current_stock > 0 else "0 days (Out of stock)"
        
    # Categorize Inventory Health
    if (annual_sales < 50 or velocity_30d < 0.10) and current_stock > 0:
        health_status = "DEAD / STUCK"
        action_recommendation = "Liquidation / clearance candidate to free up trapped capital."
    elif current_stock == 0 and annual_sales == 0:
        health_status = "DEAD / DISCONTINUED"
        action_recommendation = "Delist product from master catalog."
    elif (days_of_inventory_val > 180 and current_stock >= 200) or (annual_sales < 200 and current_stock >= 300):
        health_status = "SLOW MOVING"
        action_recommendation = "Slow sales pace relative to stock; consider promotional discount."
    elif risk_status in ["DEMAND DECLINING", "VOLATILE"] or (days_of_inventory_val < 14 and current_stock > 0):
        health_status = "WATCH"
        action_recommendation = "Monitor weekly sales trajectory and reorder buffer if stock is low."
    else:
        health_status = "HEALTHY"
        action_recommendation = "Maintain regular replenishment cycle."
        
    trapped_capital = round(current_stock * selling_price, 2)
    
    return {
        'recommended_daily_demand': round(recommended_daily_demand, 2),
        'days_of_inventory_str': days_of_inventory_str,
        'days_of_inventory_numeric': round(days_of_inventory_val, 1),
        'inventory_health_status': health_status,
        'inventory_action_recommendation': action_recommendation,
        'trapped_capital': trapped_capital
    }
