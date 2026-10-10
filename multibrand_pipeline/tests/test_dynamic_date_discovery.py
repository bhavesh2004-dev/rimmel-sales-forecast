"""
Unit tests for dynamic data discovery and timeline calculation.
Verifies that:
- Every brand's dates are discovered dynamically from MySQL.
- Validation windows have exactly 10 calendar dates.
- Forecast windows have exactly 10 calendar dates.
- Training target cutoffs strictly precede the validation window.
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from multibrand_pipeline.src.db_manager import DBManager
from multibrand_pipeline.src.data_discovery import discover_brand_coverage

class TestDynamicDateDiscovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DBManager()
        cls.df_brands, cls.df_quality = discover_brand_coverage(cls.db, horizon_days=10)

    def test_all_seven_brands_discovered(self):
        expected_brands = {'RIMMEL', 'MAX_FACTOR', 'GEEK_GORGEOUS', 'KIFRA', 'WELEDA', 'DELILAH', 'FRANK_BODY'}
        discovered_brands = set(self.df_brands['brand_id'].unique())
        self.assertEqual(expected_brands, discovered_brands, "All seven brands must be discovered from MySQL")

    def test_validation_window_exact_ten_days(self):
        for _, row in self.df_brands.iterrows():
            start = pd.to_datetime(row['validation_start'])
            end = pd.to_datetime(row['validation_end'])
            num_days = (end - start).days + 1
            self.assertEqual(num_days, 10, f"Validation window for {row['brand_id']} must be exactly 10 calendar dates (got {num_days})")

    def test_forecast_window_exact_ten_days(self):
        for _, row in self.df_brands.iterrows():
            start = pd.to_datetime(row['forecast_start'])
            end = pd.to_datetime(row['forecast_end'])
            num_days = (end - start).days + 1
            self.assertEqual(num_days, 10, f"Forecast window for {row['brand_id']} must be exactly 10 calendar dates (got {num_days})")

    def test_training_cutoff_precedes_validation(self):
        for _, row in self.df_brands.iterrows():
            train_cutoff = pd.to_datetime(row['validation_training_cutoff'])
            val_start = pd.to_datetime(row['validation_start'])
            self.assertTrue(train_cutoff < val_start, f"Training cutoff ({train_cutoff}) must precede validation start ({val_start})")
            self.assertEqual((val_start - train_cutoff).days, 1, f"Training cutoff must be exactly the day before validation start for {row['brand_id']}")

    def test_forecast_starts_day_after_cutoff(self):
        for _, row in self.df_brands.iterrows():
            cutoff_t = pd.to_datetime(row['cutoff_date_T'])
            fwd_start = pd.to_datetime(row['forecast_start'])
            self.assertEqual((fwd_start - cutoff_t).days, 1, f"Forecast start must be T + 1 day for {row['brand_id']}")

if __name__ == '__main__':
    unittest.main()
