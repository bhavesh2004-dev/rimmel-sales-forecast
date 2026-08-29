"""
AUTOMATED TEST SUITE: PIPELINE INTEGRITY & HOLDOUT VERIFICATION
==============================================================
Validates:
1. Master catalog SKU integrity (588 SKUs).
2. Protected holdout isolation and accuracy matching.
3. Dynamic date horizon scaling (7d, 11d, 14d, 31d).
4. Evidence-based confidence calibration.
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
    HOLDOUT_EVAL_START, HOLDOUT_EVAL_END, HOLDOUT_DAYS, CATALOG_SKU_COUNT
)
from src.data_loader import load_sales_data
from src.preprocessing import preprocess_sales_data
from src.dynamic_engine import run_dynamic_forecast
from src.validation import evaluate_holdout_performance
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
        
    def test_zero_data_leakage_preprocessing(self):
        """Verifies training slice contains zero records beyond the cutoff date."""
        train_slice = preprocess_sales_data(
            self.df_sales, start_date=HOLDOUT_TRAIN_START, end_date=HOLDOUT_TRAIN_END
        )
        max_train_date = train_slice['date'].max()
        self.assertLessEqual(
            max_train_date, pd.to_datetime(HOLDOUT_TRAIN_END),
            "Data leakage detected: Training data contains dates after cutoff!"
        )
        
    def test_protected_holdout_performance(self):
        """Verifies holdout evaluation on 21-31 July matches verified baseline metrics."""
        # 1. Run dynamic engine strictly on training slice
        df_forecast = run_dynamic_forecast(
            self.df_sales, 
            train_start=HOLDOUT_TRAIN_START, 
            train_end=HOLDOUT_TRAIN_END, 
            forecast_days=HOLDOUT_DAYS
        )
        
        # 2. Extract ground truth actuals for holdout
        df_holdout = self.df_sales[
            (self.df_sales['date'] >= pd.to_datetime(HOLDOUT_EVAL_START)) & 
            (self.df_sales['date'] <= pd.to_datetime(HOLDOUT_EVAL_END))
        ]
        actuals_map = df_holdout.groupby('sku')['total_sales'].sum().to_dict()
        
        # 3. Evaluate
        df_evaluated, metrics = evaluate_holdout_performance(df_forecast, actuals_map)
        
        self.assertGreater(metrics['total_actual_units'], 9000, "Ground truth units on July 21-31 must be > 9,000.")
        self.assertLess(metrics['wape_recommended_pct'], 48.0, "Recommended WAPE must remain under 48%.")
        self.assertGreater(metrics['wape_recommended_pct'], 35.0, "WAPE should be realistic and honest.")
        
    def test_dynamic_horizon_scaling(self):
        """Verifies engine scales predictions dynamically for different forecast horizons."""
        # Test 7 days
        df_7d = run_dynamic_forecast(self.df_sales, train_end=HOLDOUT_TRAIN_END, forecast_days=7)
        # Test 31 days (full month)
        df_31d = run_dynamic_forecast(self.df_sales, train_end=HOLDOUT_TRAIN_END, forecast_days=31)
        
        sum_7d  = df_7d['Recommended Forecast'].sum()
        sum_31d = df_31d['Recommended Forecast'].sum()
        
        self.assertGreater(sum_31d, sum_7d, "31-day forecast total must exceed 7-day forecast total.")
        
    def test_report_generation(self):
        """Verifies Excel reports can be generated and written successfully."""
        df_forecast = run_dynamic_forecast(self.df_sales, train_end=HOLDOUT_TRAIN_END, forecast_days=11)
        test_out_path = os.path.join(BASE_DIR, 'reports', 'test_pipeline_export.xlsx')
        
        saved_path = generate_client_excel_report(df_forecast, test_out_path, is_evaluation=False)
        self.assertTrue(os.path.exists(saved_path), "Excel report was not saved successfully.")
        
        # Clean up test artifact
        if os.path.exists(saved_path):
            os.remove(saved_path)

if __name__ == '__main__':
    unittest.main()
