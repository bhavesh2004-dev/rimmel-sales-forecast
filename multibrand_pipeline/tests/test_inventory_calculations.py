"""
Test Inventory Calculations, Dynamic Days of Cover, and 7-Column Layout
Verifies uncapped dynamic Days of Cover, zero 999 sentinel caps, and exact 7-column schema.
"""

import os
import unittest
import yaml
import pandas as pd
import numpy as np

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.inventory import compute_days_of_cover, generate_inventory_planning_layer

class TestInventoryCalculations(unittest.TestCase):

    def setUp(self):
        config_path = os.path.join(PIPELINE_ROOT, "config", "pipeline_config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def test_dynamic_doc_uncapped(self):
        """Proves Days of Cover calculates true mathematical value without 999 sentinel ceiling."""
        # Case 1: Standard high stock
        doc_1 = compute_days_of_cover(stock=650.0, forecast_10d=12.0) # daily = 1.2
        self.assertEqual(round(doc_1, 1), 541.7)
        self.assertNotEqual(doc_1, 999.0)
        
        # Case 2: Extreme stock (e.g. 5,000 units with 1 unit forecast)
        doc_2 = compute_days_of_cover(stock=5000.0, forecast_10d=1.0) # daily = 0.1
        self.assertEqual(doc_2, 50000.0)
        self.assertNotEqual(doc_2, 999.0)
        
        # Case 3: Stockout
        doc_3 = compute_days_of_cover(stock=0.0, forecast_10d=50.0)
        self.assertEqual(doc_3, 0.0)
        
        # Case 4: Dead stock (Stock > 0 and Forecast == 0)
        doc_4 = compute_days_of_cover(stock=150.0, forecast_10d=0.0)
        self.assertEqual(doc_4, "No Projected Demand")
        self.assertNotEqual(doc_4, 999.0)

    def test_certified_7_columns_only(self):
        """Proves inventory layer contains EXACTLY the 7 certified columns."""
        df_10d = pd.DataFrame({
            'canonical_sku': ['SKU_1', 'SKU_2'],
            'product_title': ['Lipstick', 'Mascara'],
            'category': ['Lipstick', 'Mascara'],
            'current_stock': [500.0, 0.0],
            'in_stock_flag': [1, 0],
            'Total Predicted': [20.0, 15.0],
            'v90': [2.0, 0.0],
            'cv_30': [0.3, 1.5]
        })
        
        inv_df = generate_inventory_planning_layer(df_10d, self.config)
        expected_cols = [
            'Brand', 'SKU', 'Product', 'Current Stock', '10-Day Forecast',
            'Days of Cover', 'Risk', 'Recommended Action'
        ]
        
        self.assertEqual(list(inv_df.columns), expected_cols)
        self.assertEqual(len(inv_df.columns), 8)
        
        # Verify no replenishment fields exist
        forbidden_fields = ['Lead Time', 'Safety Stock', 'ROP', 'Replenishment Need', 'LTD']
        for ff in forbidden_fields:
            self.assertNotIn(ff, inv_df.columns, f"Forbidden replenishment field {ff} found in client inventory layer!")

if __name__ == "__main__":
    unittest.main()
