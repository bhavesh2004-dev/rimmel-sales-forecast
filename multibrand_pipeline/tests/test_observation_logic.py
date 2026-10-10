"""
Test Observation Logic, Cartesian Grid, and Stock Telemetry Discipline
Verifies Cartesian grid completeness, shared stock isolation, and no stock backfill before August 2025.
"""

import os
import unittest
import pandas as pd
import numpy as np

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.observation_grid import construct_daily_grid

class TestObservationLogic(unittest.TestCase):

    def setUp(self):
        # Transaction records spanning pre-August 2025 and post-August 2025
        self.df_norm = pd.DataFrame({
            'brand_id': ['RIMMEL', 'RIMMEL', 'RIMMEL', 'RIMMEL'],
            'canonical_sku': ['SKU_A', 'SKU_A', 'SKU_A', 'SKU_A'],
            'platform_group': ['Amazon', 'eBay', 'Amazon', 'eBay'],
            'date': ['2025-05-10', '2025-05-10', '2025-08-15', '2025-08-15'],
            'units_sold': [5.0, 3.0, 10.0, 8.0],
            'orders_count': [3, 2, 8, 5],
            'selling_price': [6.50, 6.50, 6.50, 6.50],
            'current_stock': [np.nan, np.nan, 250.0, 250.0], # Shared warehouse stock of 250 units
            'product_title': ['Mascara Black', 'Mascara Black', 'Mascara Black', 'Mascara Black'],
            'category': ['Mascara', 'Mascara', 'Mascara', 'Mascara'],
            'pack_multiplier': [1, 1, 1, 1]
        })

    def test_no_stock_backfill_before_august_2025(self):
        """Proves stock telemetry before August 2025 remains unobserved and has_inventory_signal = 0."""
        grid = construct_daily_grid(
            df_norm=self.df_norm,
            start_date="2025-05-01",
            end_date="2025-08-31",
            stock_telemetry_valid_start="2025-08-01"
        )
        
        # Check pre-August 2025 dates
        pre_aug = grid[grid['date'] < '2025-08-01']
        self.assertTrue((pre_aug['has_inventory_signal'] == 0).all(), "has_inventory_signal was non-zero before Aug 2025!")
        
        # Check post-August 2025 dates where stock was reported
        post_aug = grid[(grid['date'] >= '2025-08-15') & (grid['has_inventory_signal'] == 1)]
        self.assertTrue(len(post_aug) > 0)
        self.assertEqual(post_aug['current_stock'].iloc[0], 250.0)

    def test_shared_stock_not_multiplied(self):
        """Proves shared warehouse stock is associated with canonical SKU and not summed across channels."""
        grid = construct_daily_grid(
            df_norm=self.df_norm,
            start_date="2025-08-15",
            end_date="2025-08-15",
            stock_telemetry_valid_start="2025-08-01"
        )
        
        # On 2025-08-15, we have Amazon and eBay rows
        # Both must report the shared warehouse stock of 250.0, NOT 500.0 or split 125.0
        amz_row = grid[grid['platform_group'] == 'Amazon'].iloc[0]
        ebay_row = grid[grid['platform_group'] == 'eBay'].iloc[0]
        
        self.assertEqual(amz_row['current_stock'], 250.0)
        self.assertEqual(ebay_row['current_stock'], 250.0)

    def test_complete_cartesian_zero_imputation(self):
        """Proves absent transaction days are populated with model_units_sold = 0 and data_treatment = 'ZERO'."""
        grid = construct_daily_grid(
            df_norm=self.df_norm,
            start_date="2025-08-01",
            end_date="2025-08-31",
            stock_telemetry_valid_start="2025-08-01"
        )
        
        # 31 days x 2 platforms = 62 rows
        self.assertEqual(len(grid), 62)
        
        # Check an unobserved date (e.g. 2025-08-02)
        unobs = grid[grid['date'] == '2025-08-02']
        self.assertEqual(len(unobs), 2)
        self.assertTrue((unobs['observed_units_sold'] == 0.0).all())
        self.assertTrue((unobs['is_observed_sale'] == 0).all())
        self.assertTrue((unobs['data_treatment'] == 'ZERO').all())

if __name__ == "__main__":
    unittest.main()
