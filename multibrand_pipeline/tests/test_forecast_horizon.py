"""
Test Forecast Horizon, Forward Simulation Dynamics, and Platform Aggregation
Verifies exact 10-day horizon, no prediction * 10 shortcut, and additive platform aggregation law.
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

from src.forecast import generate_forward_simulation

class DummyModel:
    """Mock model that simulates demand sensitive to day_of_week."""
    def predict(self, X):
        # Base 5.0 units + 2.0 on weekends (dow 5 or 6)
        dow = X['day_of_week'].values
        preds = 5.0 + np.where(dow >= 5, 2.0, 0.0)
        return preds

class TestForecastHorizon(unittest.TestCase):

    def setUp(self):
        config_path = os.path.join(PIPELINE_ROOT, "config", "pipeline_config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.feature_schema = {
            'feature_list': ['canonical_sku', 'day_of_week', 'v7', 'v14', 'v30', 'in_stock_flag'],
            'categorical_features': ['canonical_sku']
        }
        
        # Synthetic feature state at anchor date 2026-09-10
        self.anchor_df = pd.DataFrame({
            'date': ['2026-09-10', '2026-09-10'],
            'brand_id': ['RIMMEL', 'RIMMEL'],
            'canonical_sku': ['SKU_A', 'SKU_A'],
            'platform_group': ['Amazon', 'eBay'],
            'product_title': ['Mascara', 'Mascara'],
            'category': ['Mascara', 'Mascara'],
            'day_of_week': [3, 3],
            'v7': [5.0, 5.0],
            'v14': [5.0, 5.0],
            'v30': [5.0, 5.0],
            'v90': [5.0, 5.0],
            'cv_30': [0.4, 0.4],
            'in_stock_flag': [1, 1],
            'current_stock': [100.0, 100.0]
        })

    def test_exact_10_day_horizon(self):
        """Proves forward simulation produces exactly 10 distinct calendar dates."""
        model = DummyModel()
        df_daily, df_10d_sku = generate_forward_simulation(
            model=model,
            df_features=self.anchor_df,
            config=self.config,
            feature_schema=self.feature_schema
        )
        
        unique_dates = df_daily['date'].nunique()
        self.assertEqual(unique_dates, 10, f"Expected exactly 10 forward dates, got {unique_dates}")
        self.assertEqual(df_daily['date'].min(), "2026-09-11")
        self.assertEqual(df_daily['date'].max(), "2026-09-20")

    def test_no_static_multiplier_shortcut(self):
        """Proves model predictions dynamically vary by forward calendar day (weekend vs weekday)."""
        model = DummyModel()
        df_daily, df_10d_sku = generate_forward_simulation(
            model=model,
            df_features=self.anchor_df,
            config=self.config,
            feature_schema=self.feature_schema
        )
        
        amz_daily = df_daily[df_daily['platform_group'] == 'Amazon']
        weekday_preds = amz_daily[amz_daily['day_of_week'] < 5]['calib_daily_prediction'].unique()
        weekend_preds = amz_daily[amz_daily['day_of_week'] >= 5]['calib_daily_prediction'].unique()
        
        # Must have distinct predictions reflecting dynamic day-of-week simulation
        self.assertNotEqual(list(weekday_preds), list(weekend_preds), "Predictions were static across all forward dates!")

    def test_platform_additive_aggregation(self):
        """Proves Total Predicted strictly equals Amazon + eBay + Website + Other."""
        model = DummyModel()
        _, df_10d_sku = generate_forward_simulation(
            model=model,
            df_features=self.anchor_df,
            config=self.config,
            feature_schema=self.feature_schema
        )
        
        for _, r in df_10d_sku.iterrows():
            expected_tot = r['Amazon'] + r['eBay'] + r['Website'] + r['Other']
            self.assertEqual(r['Total Predicted'], expected_tot, "Platform additive aggregation failed!")

if __name__ == "__main__":
    unittest.main()
