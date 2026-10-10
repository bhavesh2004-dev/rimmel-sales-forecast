"""
Multi-Brand Downstream Replenishment & Reorder Point (ROP) Module (Phase 5C)
=============================================================================
Calculates operational replenishment requirements and reorder points (ROP)
based strictly on validated business rules provided by business ownership.

Simplified 10-Column Output:
1. Brand
2. SKU
3. Product
4. Current Stock
5. 10-Day Forecast
6. Avg Daily Usage
7. Lead-Time Demand
8. Target Stock
9. Replenishment Qty
10. ROP Status

Policy configuration (Lead Time = 10 Days, Minimum Stock Level = 6 Units)
is displayed compactly in the header banner, eliminating repetitive row-level columns.
"""

import math
import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def compute_rop_status(
    current_stock: float,
    target_stock: float,
    min_stock: float,
    forecast_10d: float
) -> str:
    """
    Assigns deterministic ROP status based on inventory runway and business targets.
    
    Status Logic:
    - If 10-Day Forecast == 0 and Current Stock >= 6: NO PROJECTED DEMAND
    - If 10-Day Forecast == 0 and Current Stock < 6: BELOW ROP
    - If Current Stock > Target Stock: ABOVE ROP
    - Else If Current Stock == 6: AT MINIMUM STOCK
    - Else: BELOW ROP
    """
    if forecast_10d == 0:
        if current_stock < min_stock:
            return "BELOW ROP"
        elif current_stock == min_stock:
            return "AT MINIMUM STOCK"
        else:
            return "NO PROJECTED DEMAND"
            
    if current_stock > target_stock:
        return "ABOVE ROP"
    elif current_stock == min_stock:
        return "AT MINIMUM STOCK"
    else:
        return "BELOW ROP"

def calculate_sku_replenishment(
    current_stock: float,
    forecast_10d: float,
    lead_time_days: int = 10,
    min_stock_level: int = 6,
    daily_rate_window: int = 10,
    rounding_method: str = "ceil"
) -> Dict[str, Any]:
    """
    Computes simplified ROP metrics for a single canonical SKU.
    
    Formulas:
    - Avg Daily Usage = 10-Day Forecast / daily_rate_window
    - Lead-Time Demand = Avg Daily Usage * Lead Time
    - Target Stock = Lead-Time Demand + Minimum Stock Level
    - Replenishment Qty = ceil(max(Target Stock - Current Stock, 0))
    """
    stock = float(current_stock) if pd.notna(current_stock) else 0.0
    fwd = float(forecast_10d) if pd.notna(forecast_10d) else 0.0
    
    # 1. Average Daily Usage derived strictly from ML forecast
    adu = round(fwd / float(daily_rate_window), 2)
    
    # 2. Lead-Time Demand
    lead_time_demand = round(adu * float(lead_time_days), 2)
    
    # 3. Target Stock
    target_stock = round(lead_time_demand + float(min_stock_level), 2)
    
    # 4. Replenishment Need & Actionable Order Quantity (Replenishment Qty)
    rep_need = max(target_stock - stock, 0.0)
    if rounding_method == "ceil":
        rec_order_qty = int(max(math.ceil(rep_need), 0))
    else:
        rec_order_qty = int(max(round(rep_need), 0))
        
    # 5. Deterministic Status
    status = compute_rop_status(stock, target_stock, float(min_stock_level), fwd)
    
    return {
        'Avg Daily Usage': adu,
        'Lead-Time Demand': lead_time_demand,
        'Target Stock': target_stock,
        'Replenishment Qty': rec_order_qty,
        'ROP Status': status
    }

def generate_replenishment_layer(
    df_forecast: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generates the certified downstream ROP / Replenishment table for the entire catalog.
    
    Produces EXACTLY 10 simplified columns:
    1. Brand
    2. SKU
    3. Product
    4. Current Stock
    5. 10-Day Forecast
    6. Avg Daily Usage
    7. Lead-Time Demand
    8. Target Stock
    9. Replenishment Qty
    10. ROP Status
    """
    logger.info("Generating downstream ROP / Replenishment operational planning layer...")
    
    rop_cfg = config.get('replenishment', {})
    lead_time = int(rop_cfg.get('lead_time_days', 10))
    min_stock = int(rop_cfg.get('minimum_stock_level', 6))
    daily_window = int(rop_cfg.get('daily_rate_window_days', 10))
    rounding = rop_cfg.get('rounding_method', 'ceil')
    
    records = []
    
    for _, r in df_forecast.iterrows():
        brand_raw = r.get('brand_id', r.get('Brand', 'UNKNOWN'))
        brand = 'Rimmel' if 'RIMMEL' in str(brand_raw).upper() else ('Max Factor' if 'MAX' in str(brand_raw).upper() else str(brand_raw))
        sku = r.get('canonical_sku', r.get('SKU', ''))
        prod = r.get('category', '') if pd.notna(r.get('category')) and r.get('category') != '' else r.get('product_title', r.get('Product', sku))
        stock = int(float(r.get('current_stock', r.get('Current Stock', 0.0)))) if pd.notna(r.get('current_stock', r.get('Current Stock', 0.0))) else 0
        
        # 10-Day Forecast canonical total
        fwd_10d = int(round(float(r.get('10-Day Forecast', r.get('Total Predicted', 0.0)))))
        
        # Downstream calculation
        rop_metrics = calculate_sku_replenishment(
            current_stock=stock,
            forecast_10d=fwd_10d,
            lead_time_days=lead_time,
            min_stock_level=min_stock,
            daily_rate_window=daily_window,
            rounding_method=rounding
        )
        
        row_dict = {
            'Brand': brand,
            'SKU': sku,
            'Product': prod,
            'Current Stock': stock,
            '10-Day Forecast': fwd_10d,
            **rop_metrics
        }
        records.append(row_dict)
        
    df_rop = pd.DataFrame(records)
    
    expected_cols = config.get('reporting', {}).get('rop_columns', [
        'Brand', 'SKU', 'Product', 'Current Stock', '10-Day Forecast',
        'Avg Daily Usage', 'Lead-Time Demand', 'Target Stock',
        'Replenishment Qty', 'ROP Status'
    ])
    df_rop = df_rop[expected_cols]
    
    logger.info(f"Replenishment layer successfully generated for {len(df_rop):,} SKUs with {len(expected_cols)} columns.")
    return df_rop

def validate_replenishment_invariants(df_rop: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Executes mandatory mathematical and architectural validation assertions on the simplified ROP dataset.
    """
    logger.info("Executing comprehensive ROP validation assertions...")
    rop_cfg = config.get('replenishment', {})
    lead_time = int(rop_cfg.get('lead_time_days', 10))
    min_stock = int(rop_cfg.get('minimum_stock_level', 6))
    
    # 1. Formula assertions
    expected_adu = np.round(df_rop['10-Day Forecast'] / 10.0, 2)
    assert np.allclose(df_rop['Avg Daily Usage'], expected_adu, atol=1e-2), "Avg Daily Usage mismatch!"
    
    expected_ltd = np.round(df_rop['Avg Daily Usage'] * lead_time, 2)
    assert np.allclose(df_rop['Lead-Time Demand'], expected_ltd, atol=1e-2), "Lead-Time Demand mismatch!"
    
    expected_target = np.round(df_rop['Lead-Time Demand'] + min_stock, 2)
    assert np.allclose(df_rop['Target Stock'], expected_target, atol=1e-2), "Target Stock mismatch!"
    
    # 2. Non-negativity
    assert (df_rop['Replenishment Qty'] >= 0).all(), "Negative replenishment qty found!"
    
    # 3. Projected Stock buffer invariant:
    # Projected Stock after consuming lead time demand + arrival of replenishment >= min_stock
    projected_stock = df_rop['Current Stock'] + df_rop['Replenishment Qty'] - df_rop['Lead-Time Demand']
    min_projected = projected_stock.min()
    assert min_projected >= min_stock - 1e-4, f"Projected stock dropped below {min_stock}! Min was: {min_projected}"
    
    # 4. Grain uniqueness
    duplicates = df_rop.duplicated(subset=['Brand', 'SKU']).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate Brand+SKU combinations in ROP layer!"
    
    logger.info("All ROP validation invariants verified successfully (0 violations).")
