"""
Multi-Brand Inventory Planning & Risk Layer
Calculates dynamic uncapped Days of Cover and assigns operational inventory risk status.
Rule: Exactly 7 columns. Zero 999 sentinel caps. No experimental replenishment fields.
"""

import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def compute_days_of_cover(stock: float, forecast_10d: float) -> Any:
    """
    Computes uncapped dynamic Days of Cover.
    - If stock <= 0: 0.0
    - If forecast > 0: stock / (forecast / 10.0)
    - If forecast == 0 and stock > 0: 'No Projected Demand'
    """
    if pd.isna(stock) or stock <= 0:
        return 0.0
        
    daily_rate = forecast_10d / 10.0
    if daily_rate > 0.001:
        return round(stock / daily_rate, 1)
    else:
        return "No Projected Demand"

def classify_inventory_risk_and_action(
    stock: float,
    in_stock_flag: int,
    forecast_10d: float,
    v90: float,
    cv_30: float,
    config: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Assigns operational inventory risk category and recommended action.
    """
    risk_cfg = config.get('inventory_risk', {})
    cv_thresh = risk_cfg.get('cv_volatility_threshold', 1.2)
    healthy_exp = risk_cfg.get('healthy_exp_min', 5.0)
    healthy_cv = risk_cfg.get('healthy_cv_max', 0.6)
    low_anchor = risk_cfg.get('low_demand_anchor_max', 1.0)
    
    if in_stock_flag == 0 or stock <= 0:
        return "STOCKOUT RISK", "Urgent Restock"
    elif cv_30 > cv_thresh:
        return "HIGH VOLATILITY", "Monitor Closely"
    elif forecast_10d >= healthy_exp and cv_30 <= healthy_cv:
        return "NORMAL", "Maintain Current Flow"
    elif forecast_10d < low_anchor and v90 < low_anchor:
        return "LOW DEMAND", "No Replenishment"
    elif stock < forecast_10d:
        return "STOCKOUT RISK", "Order Replenishment"
    else:
        return "NORMAL", "Maintain Current Flow"

def generate_inventory_planning_layer(
    df_10d_sku: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Builds the certified 7-column inventory risk planning table.
    """
    logger.info("Generating certified 7-column inventory planning table...")
    records = []
    
    for _, r in df_10d_sku.iterrows():
        sku = r['canonical_sku']
        prod = r['category'] if pd.notna(r['category']) and r['category'] != '' else r['product_title']
        stock = float(r['current_stock']) if pd.notna(r['current_stock']) else 0.0
        fwd_10d = float(r['Total Predicted'])
        in_stock = int(r.get('in_stock_flag', 1 if stock > 0 else 0))
        v90 = float(r.get('v90', 0.0))
        cv = float(r.get('cv_30', 0.0))
        
        brand = r.get('brand_id', 'UNKNOWN')
        doc = compute_days_of_cover(stock, fwd_10d)
        risk, action = classify_inventory_risk_and_action(stock, in_stock, fwd_10d, v90, cv, config)
        
        records.append({
            'Brand': brand,
            'SKU': sku,
            'Product': prod,
            'Current Stock': int(stock),
            '10-Day Forecast': int(fwd_10d),
            'Days of Cover': doc,
            'Risk': risk,
            'Recommended Action': action
        })
        
    df_inv = pd.DataFrame(records)
    
    # Expected columns enforcement (Brand + 7 operational inventory columns)
    expected_cols = config.get('reporting', {}).get('inventory_columns', [
        'Brand', 'SKU', 'Product', 'Current Stock', '10-Day Forecast',
        'Days of Cover', 'Risk', 'Recommended Action'
    ])
    df_inv = df_inv[expected_cols]
    logger.info(f"Generated inventory planning table for {len(df_inv):,} SKUs with {len(expected_cols)} columns.")
    return df_inv
