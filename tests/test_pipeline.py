"""
AUTOMATED TEST SUITE: PIPELINE INTEGRITY & HOLDOUT BENCHMARK VERIFICATION
=========================================================================
Validates:
1. Master catalog SKU integrity (588 SKUs).
2. Clean production pipeline defaults (100% full training history Aug 1, 2025 to Jul 31, 2026).
3. Isolated holdout benchmark execution via src.validation (37.73% WAPE parity).
4. Dynamic date horizon scaling (7d, 11d, 14d, 31d).
5. Multi-sheet Excel report generation.
"""
import os
import sys
import unittest
import pandas as pd

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.settings import (
    DB_PATH, HOLDOUT_TRAIN_START, HOLDOUT_TRAIN_END, 
    HOLDOUT_EVAL_START, HOLDOUT_EVAL_END, HOLDOUT_DAYS, 
    DEFAULT_PROD_TRAIN_START, DEFAULT_PROD_CUTOFF, CATALOG_SKU_COUNT
)
from src.data_loader import load_sales_data
from src.preprocessing import preprocess_sales_data
from src.dynamic_engine import run_dynamic_forecast
from src.validation import run_holdout_benchmark, evaluate_holdout_performance
from src.report_generator import generate_client_excel_report

class TestForecastingPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Loads master dataset once for test efficiency."""
        cls.df_sales = load_sales_data(DB_PATH)
        
    def test_data_loader_and_catalog_count(self):
        """Verifies master catalog has exactly 588 unique SKUs."""
        unique_skus = self.df_sales['sku'].nunique()
        self.assertEqual(unique_skus, CATALOG_SKU_COUNT, f"Expected {CATALOG_SKU_COUNT} SKUs, got {unique_skus}")
        
    def test_clean_production_pipeline_defaults(self):
        """Verifies production pipeline defaults to full data (Aug 1, 2025 to Jul 31, 2026)."""
        train_slice = preprocess_sales_data(self.df_sales)
        max_train_date = train_slice['date'].max()
        self.assertEqual(
            max_train_date, pd.to_datetime(DEFAULT_PROD_CUTOFF),
            f"Production pipeline must default to {DEFAULT_PROD_CUTOFF}, got {max_train_date}"
        )
        
        # Verify run_dynamic_forecast runs cleanly with defaults
        df_prod = run_dynamic_forecast(self.df_sales, forecast_days=11)
        self.assertEqual(len(df_prod), CATALOG_SKU_COUNT)
        self.assertIn('Recommended Forecast', df_prod.columns)
        
    def test_isolated_holdout_benchmark_performance(self):
        """Verifies isolated holdout benchmark on July 21-31 matches verified 37.73% WAPE benchmark."""
        df_evaluated, metrics = run_holdout_benchmark(self.df_sales)
        
        self.assertEqual(len(df_evaluated), CATALOG_SKU_COUNT)
        self.assertEqual(metrics['total_actual_units'], 10480, "Ground truth units on July 21-31 must be exactly 10,480.")
        self.assertEqual(metrics['wape_recommended_pct'], 37.73, "Recommended WAPE must remain exactly 37.73%.")
        self.assertEqual(metrics['wape_baseline_pct'], 51.89, "Baseline WAPE must remain exactly 51.89%.")
        self.assertEqual(metrics['wape_momentum_pct'], 39.06, "Momentum WAPE must remain exactly 39.06%.")
        
    def test_dynamic_horizon_scaling(self):
        """Verifies engine scales predictions dynamically for different forecast horizons."""
        # Test 7 days
        df_7d = run_dynamic_forecast(self.df_sales, forecast_days=7)
        # Test 31 days (full month)
        df_31d = run_dynamic_forecast(self.df_sales, forecast_days=31)
        
        sum_7d  = df_7d['Recommended Forecast'].sum()
        sum_31d = df_31d['Recommended Forecast'].sum()
        
        self.assertGreater(sum_31d, sum_7d, "31-day forecast total must exceed 7-day forecast total.")
        
    def test_report_generation(self):
        """Verifies Excel reports can be generated and written successfully."""
        df_forecast = run_dynamic_forecast(self.df_sales, forecast_days=11)
        test_out_path = os.path.join(BASE_DIR, 'reports', 'test_pipeline_export.xlsx')
        
        saved_path = generate_client_excel_report(df_forecast, test_out_path, is_evaluation=False)
        self.assertTrue(os.path.exists(saved_path), "Excel report was not saved successfully.")
        
        # Clean up test artifact
        if os.path.exists(saved_path):
            os.remove(saved_path)

if __name__ == '__main__':
    unittest.main()
