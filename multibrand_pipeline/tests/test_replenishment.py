"""
Unit Tests for Replenishment & ROP Module (Phase 5C)
Verifies:
- All 5 mandated test cases from business specification with simplified keys
- Zero forecast handling
- Projected stock buffer invariant >= 6
- Brand agnosticism
- ROP feature isolation (zero leakage into model features)
- 10 simplified columns in replenishment layer
"""

import os
import unittest
import json
import yaml
import pandas as pd
import numpy as np

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.replenishment import calculate_sku_replenishment, generate_replenishment_layer, validate_replenishment_invariants
from src.inventory import compute_days_of_cover

class TestReplenishment(unittest.TestCase):

    def setUp(self):
        config_path = os.path.join(PIPELINE_ROOT, "config", "pipeline_config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        schema_path = os.path.join(PIPELINE_ROOT, "config", "feature_schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

    def test_case_1_normal_replenishment(self):
        """Case 1: Forecast = 40, Stock = 25 -> ADU = 4, LTD = 40, Target = 46, Replenishment Qty = 21"""
        res = calculate_sku_replenishment(current_stock=25, forecast_10d=40)
        self.assertEqual(res['Avg Daily Usage'], 4.0)
        self.assertEqual(res['Lead-Time Demand'], 40.0)
        self.assertEqual(res['Target Stock'], 46.0)
        self.assertEqual(res['Replenishment Qty'], 21)
        self.assertEqual(res['ROP Status'], 'BELOW ROP')

    def test_case_2_zero_forecast_excess_stock(self):
        """Case 2: Forecast = 0, Stock = 100 -> ADU = 0, DoC = 'No Projected Demand', Replenishment Qty = 0"""
        res = calculate_sku_replenishment(current_stock=100, forecast_10d=0)
        doc = compute_days_of_cover(stock=100, forecast_10d=0)
        self.assertEqual(res['Avg Daily Usage'], 0.0)
        self.assertEqual(res['Lead-Time Demand'], 0.0)
        self.assertEqual(res['Target Stock'], 6.0)
        self.assertEqual(res['Replenishment Qty'], 0)
        self.assertEqual(doc, 'No Projected Demand')
        self.assertEqual(res['ROP Status'], 'NO PROJECTED DEMAND')

    def test_case_3_zero_forecast_low_stock(self):
        """Case 3: Forecast = 0, Stock = 3 -> ADU = 0, DoC = 'No Projected Demand', Target = 6, Replenishment Qty = 3"""
        res = calculate_sku_replenishment(current_stock=3, forecast_10d=0)
        doc = compute_days_of_cover(stock=3, forecast_10d=0)
        self.assertEqual(res['Avg Daily Usage'], 0.0)
        self.assertEqual(res['Lead-Time Demand'], 0.0)
        self.assertEqual(res['Target Stock'], 6.0)
        self.assertEqual(res['Replenishment Qty'], 3)
        self.assertEqual(doc, 'No Projected Demand')
        self.assertEqual(res['ROP Status'], 'BELOW ROP')

    def test_case_4_high_forecast_excess_stock(self):
        """Case 4: Forecast = 100, Stock = 500 -> ADU = 10, LTD = 100, Target = 106, Replenishment Qty = 0"""
        res = calculate_sku_replenishment(current_stock=500, forecast_10d=100)
        self.assertEqual(res['Avg Daily Usage'], 10.0)
        self.assertEqual(res['Lead-Time Demand'], 100.0)
        self.assertEqual(res['Target Stock'], 106.0)
        self.assertEqual(res['Replenishment Qty'], 0)
        self.assertEqual(res['ROP Status'], 'ABOVE ROP')

    def test_case_5_high_forecast_low_stock(self):
        """Case 5: Forecast = 100, Stock = 50 -> ADU = 10, LTD = 100, Target = 106, Replenishment Qty = 56"""
        res = calculate_sku_replenishment(current_stock=50, forecast_10d=100)
        self.assertEqual(res['Avg Daily Usage'], 10.0)
        self.assertEqual(res['Lead-Time Demand'], 100.0)
        self.assertEqual(res['Target Stock'], 106.0)
        self.assertEqual(res['Replenishment Qty'], 56)
        self.assertEqual(res['ROP Status'], 'BELOW ROP')

    def test_projected_stock_buffer_invariant(self):
        """Proves that in all possible stock and forecast states, projected stock >= 6."""
        for stock in [0, 1, 3, 5, 6, 7, 20, 50, 100, 500]:
            for fwd in [0, 1, 5, 10, 40, 100, 250]:
                res = calculate_sku_replenishment(stock, fwd)
                proj = stock + res['Replenishment Qty'] - res['Lead-Time Demand']
                self.assertGreaterEqual(proj, 6.0 - 1e-4, f"Failed for stock={stock}, fwd={fwd}")

    def test_rop_fields_excluded_from_model_features(self):
        """Verifies that ROP/replenishment fields are NEVER present in LightGBM feature schema."""
        feature_list = self.schema['feature_list']
        forbidden_fields = [
            'ROP', 'rop', 'replenishment', 'replenishment_need', 'replenishment_qty',
            'recommended_order_qty', 'target_stock', 'lead_time',
            'minimum_stock_level', 'lead_time_demand'
        ]
        for f in feature_list:
            for forbidden in forbidden_fields:
                self.assertNotEqual(f.lower(), forbidden.lower(), f"Forbidden ROP field {f} found in model feature schema!")

    def test_layer_generation_and_invariants(self):
        """Verifies dataframe generation across multi-brand SKUs and validation assertions for 10 simplified columns."""
        df_sample = pd.DataFrame({
            'Brand': ['Rimmel', 'Max Factor', 'Rimmel'],
            'SKU': ['RIM-1', 'MF-2', 'RIM-3'],
            'Product': ['Mascara', 'Foundation', 'Lipstick'],
            'Current Stock': [25, 100, 3],
            '10-Day Forecast': [40, 0, 0]
        })
        df_rop = generate_replenishment_layer(df_sample, self.config)
        self.assertEqual(len(df_rop), 3)
        self.assertEqual(len(df_rop.columns), 10)
        validate_replenishment_invariants(df_rop, self.config)

if __name__ == "__main__":
    unittest.main()
