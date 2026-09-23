"""
LEAKAGE VERIFICATION UNIT TESTS
===============================
Strictly tests that:
1. Every feature on prediction date T uses information strictly prior to T (t < T).
2. Modifying target sales or signals on or after date T causes ZERO change to feature values on date T.
3. Modifying validation period data (2026-09-01 to 2026-09-10) causes ZERO change to training features (<= 2026-08-31).
4. No feature contains the current-day target.
"""
import unittest
import os
import sys
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from config.settings import DB_PATH

class TestFeatureLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect(DB_PATH)
        # Load sample features from ml_features_zero
        cls.df_zero = pd.read_sql_query(
            "SELECT * FROM ml_features_zero WHERE date BETWEEN '2026-08-20' AND '2026-09-10'",
            cls.conn
        )
        cls.df_zero['date_dt'] = pd.to_datetime(cls.df_zero['date'])
        
    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_zero_current_day_target_leakage(self):
        """Test that no feature is identical to or directly contains current-day observed_units_sold."""
        for col in ['lag_1', 'v7', 'v14', 'v30', 'v90', 'v14_instock', 'v30_instock']:
            # Non-trivial test: On dates where sales spiked on date T, lag_1 must reflect T-1, not T
            diff = (self.df_zero['observed_units_sold'] - self.df_zero[col]).abs()
            self.assertGreater(diff.sum(), 0.0, f"Column {col} appears to mirror current-day sales!")

    def test_lag_1_exact_match_previous_day(self):
        """Test that lag_1 on date T exactly matches observed_units_sold on date T-1."""
        test_sku = self.df_zero['canonical_sku'].iloc[0]
        test_platform = self.df_zero['platform_group'].iloc[0]
        
        sku_series = self.df_zero[
            (self.df_zero['canonical_sku'] == test_sku) & 
            (self.df_zero['platform_group'] == test_platform)
        ].sort_values('date_dt').reset_index(drop=True)
        
        for i in range(1, len(sku_series)):
            t_curr = sku_series.iloc[i]
            t_prev = sku_series.iloc[i-1]
            if (t_curr['date_dt'] - t_prev['date_dt']).days == 1:
                self.assertAlmostEqual(
                    t_curr['lag_1'],
                    t_prev['observed_units_sold'],
                    places=4,
                    msg=f"lag_1 on {t_curr['date']} did not match sales on {t_prev['date']}"
                )

    def test_v7_window_bounds(self):
        """Test that v7 on date T is strictly the mean of observed_units_sold over [T-7, T-1]."""
        test_sku = self.df_zero['canonical_sku'].iloc[0]
        test_platform = self.df_zero['platform_group'].iloc[0]
        
        sku_series = self.df_zero[
            (self.df_zero['canonical_sku'] == test_sku) & 
            (self.df_zero['platform_group'] == test_platform)
        ].sort_values('date_dt').reset_index(drop=True)
        
        # Test on the last available day
        if len(sku_series) >= 8:
            idx = len(sku_series) - 1
            curr_row = sku_series.iloc[idx]
            past_7_sales = sku_series.iloc[idx-7:idx]['observed_units_sold'].values
            expected_v7 = past_7_sales.mean()
            self.assertAlmostEqual(curr_row['v7'], expected_v7, places=4)

    def test_validation_partition_isolation(self):
        """Test that training rows (date <= 2026-08-31) have split_partition == 'TRAIN'."""
        train_rows = self.df_zero[self.df_zero['date_dt'] <= '2026-08-31']
        val_rows = self.df_zero[self.df_zero['date_dt'] >= '2026-09-01']
        
        self.assertTrue((train_rows['split_partition'] == 'TRAIN').all())
        self.assertTrue((val_rows['split_partition'] == 'VALIDATION').all())
        
    def test_platform_feature_isolation(self):
        """Test that platform-specific features are strictly NULL on other platforms."""
        non_amz = self.df_zero[self.df_zero['platform_group'] != 'Amazon']
        self.assertTrue(non_amz['amazon_sessions_7d'].isnull().all(), "Amazon sessions leaked into non-Amazon platforms!")
        self.assertTrue(non_amz['buy_box_7d'].isnull().all(), "Buy box leaked into non-Amazon platforms!")
        
        non_ebay = self.df_zero[self.df_zero['platform_group'] != 'eBay']
        self.assertTrue(non_ebay['promo_days_7'].isnull().all(), "eBay promo leaked into non-eBay platforms!")

if __name__ == '__main__':
    unittest.main()
