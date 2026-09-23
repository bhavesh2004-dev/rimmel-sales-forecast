"""
DATA-GROUNDED EXPLANATION GENERATOR MODULE
==========================================
Generates strictly data-backed business explanations using only observed
historical metrics (v7d, v14d, v30d, v90d, weekly CV, zero-sales frequency,
Momentum/Baseline ratio, and current stock).

Strict Rules:
- No invented external causes (no mentions of unrecorded promotions, ads, campaigns, or competitor actions).
- Clearly explains why the Recommended Adaptive forecast differs from Baseline or Momentum.
- If data does not provide conclusive signals, explicitly states:
  "Historical data does not provide enough evidence to determine the cause."
"""

def generate_forecast_explanation(sku_features, baseline_forecast, momentum_forecast, recommended_forecast, confidence_level, risk_status):
    """
    Constructs a purely data-grounded explanation for the final forecast.
    
    Args:
        sku_features (dict): Extracted feature dictionary.
        baseline_forecast (int): Units predicted by baseline model.
        momentum_forecast (int): Units predicted by momentum model.
        recommended_forecast (int): Final recommended forecast units.
        confidence_level (str): 'HIGH', 'MEDIUM', or 'LOW'.
        risk_status (str): Current risk status.
        
    Returns:
        str: Purely data-grounded explanation.
    """
    current_stock      = sku_features['current_stock']
    annual_sales       = int(sku_features['annual_sales'])
    weekly_cv          = sku_features['weekly_cv']
    zero_days_pct      = sku_features['zero_days_pct']
    v7                 = sku_features['velocity_7d']
    v14                = sku_features['velocity_14d']
    v30                = sku_features['velocity_30d']
    v90                = sku_features['velocity_90d']
    is_flash_spike     = sku_features['is_flash_spike']
    is_sustained_trend = sku_features['is_sustained_trend']
    
    mom_base_ratio = momentum_forecast / max(baseline_forecast, 1)
    is_aligned = (0.85 <= mom_base_ratio <= 1.20)
    
    # 1. Out-of-Stock Condition
    if current_stock == 0 and annual_sales > 0:
        return f"Current stock is 0 units in the warehouse; reorder needed to resume sales ({v30:.1f} u/d historical 30-day velocity)."
        
    # 2. Zero Historical Sales
    if annual_sales == 0 or sku_features['is_empty']:
        return "No sales recorded across historical data; classified as inactive or dead stock candidate."
        
    # 3. Intermittent / Sparse Demand Lines
    if zero_days_pct >= 50.0 or annual_sales < 100:
        return f"Intermittent demand with {zero_days_pct:.0f}% zero-sales days ({annual_sales} u/yr total volume); Adaptive forecast anchors to long-term Baseline organic run-rate."
        
    # 4. Volatile Spikes (High Weekly Volatility / High Ratio)
    if is_flash_spike or (weekly_cv > 1.25 and mom_base_ratio > 1.35):
        explanation = f"Recent 14-day velocity ({v14:.1f} u/d) is sharply higher than the 90-day baseline ({v90:.1f} u/d), but high weekly volatility (CV {weekly_cv:.2f}) indicates an erratic spike; Adaptive forecast anchors 85% to Baseline."
        
    # 5. Post-Peak Deceleration (7d slowing vs 14d)
    elif (v7 < 0.65 * v14) and (v14 >= 3.0):
        explanation = f"Recent 7-day velocity ({v7:.1f} u/d) has decelerated below the 14-day pace ({v14:.1f} u/d); Adaptive forecast pulls back toward the long-term baseline."
        
    # 6. Sustained Growth Trend (High Volume & Low Volatility)
    elif is_sustained_trend or (v14 >= 1.25 * v90 and weekly_cv <= 1.10 and annual_sales >= 1000):
        explanation = f"Recent 14-day velocity ({v14:.1f} u/d) is consistently above the 90-day baseline ({v90:.1f} u/d) with steady demand; Adaptive forecast trusts Momentum (90% weight)."
        
    # 7. Stable Core Staples (Baseline & Momentum Closely Aligned)
    elif is_aligned and weekly_cv <= 0.88:
        explanation = f"Stable demand across all horizons (14-day {v14:.1f} u/d vs 90-day {v90:.1f} u/d); Baseline and Momentum are closely aligned with {confidence_level} confidence."
        
    # 8. Demand Declining Trend
    elif (v14 < 0.70 * v30 and v30 >= 3.0) or (v30 < 0.70 * v90 and v90 >= 3.0):
        explanation = f"Recent 14-day velocity ({v14:.1f} u/d) has declined below the 90-day baseline ({v90:.1f} u/d); Adaptive forecast adapts downward to match recent run-rate."
        
    # 9. Moderate Demand Pace
    elif annual_sales >= 300:
        explanation = f"Consistent historical run-rate ({v30:.1f} u/d); Adaptive forecast blends organic baseline ({baseline_forecast} u) with recent velocity ({momentum_forecast} u)."
        
    # 10. Inconclusive / Insufficient Evidence Fallback
    else:
        explanation = "Historical data does not provide enough evidence to determine the cause; forecast is kept at baseline organic demand."
        
    # Check physical warehouse stock limitation
    if recommended_forecast == current_stock and current_stock < baseline_forecast:
        explanation += f" Constrained by warehouse stock on hand ({current_stock} units)."
        
    return explanation
