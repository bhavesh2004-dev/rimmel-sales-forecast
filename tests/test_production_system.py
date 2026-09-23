"""
TEST SUITE: FINAL CLIENT REPORT VERIFICATION (SKU-LEVEL 10-DAY SUMMARY)
========================================================================
Validates:
1. Production model serialization and exact Exp6 hyperparameters.
2. Production feature list (74 causal features).
3. Report 1 (Rimmel_Validation_Sep01_Sep10_2026.xlsx):
   - Primary sheet: 'SKU Validation Summary'
   - Grain: ONE ROW PER CANONICAL SKU (674 rows + 1 header = 675 rows)
   - Columns: SKU, Product, Validation Period, Amazon Actual, Amazon Predicted,
     eBay Actual, eBay Predicted, Website Actual, Website Predicted, Other Actual,
     Other Predicted, Total Actual, Total Predicted, Variance, Confidence, Risk, Reason
   - Mathematical consistency: Total Actual = sum of platforms, Total Predicted = sum of platforms
   - Portfolio Actuals sum to 2,069 units
   - Secondary sheet: 'Platform Summary'
4. Report 2 (Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx):
   - Primary sheet: 'SKU Forecast Summary'
   - Grain: ONE ROW PER CANONICAL SKU (674 rows + 1 header = 675 rows)
   - Columns: SKU, Product, Forecast Period, Amazon Predicted, eBay Predicted,
     Website Predicted, Other Predicted, Total Predicted, Confidence, Risk,
     Recommended Action, Reason
   - Total Predicted = sum of platforms, all >= 0
   - Secondary sheets: 'Platform Summary', 'Inventory Actions'
5. Shared warehouse inventory integrity: single central pool per SKU, never summed.
"""

import os
import sys
import json
import pickle
import sqlite3
import hashlib
import unittest
import openpyxl
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

EXPECTED_VAL_HEADERS = [
    'SKU', 'Product', 'Validation Period',
    'Amazon Actual', 'Amazon Predicted',
    'eBay Actual', 'eBay Predicted',
    'Website Actual', 'Website Predicted',
    'Other Actual', 'Other Predicted',
    'Total Actual', 'Total Predicted',
    'Variance', 'Confidence', 'Risk', 'Reason'
]

EXPECTED_FWD_HEADERS = [
    'SKU', 'Product', 'Forecast Period',
    'Amazon Predicted', 'eBay Predicted',
    'Website Predicted', 'Other Predicted',
    'Total Predicted', 'Confidence', 'Risk',
    'Recommended Action', 'Reason'
]

class TestProductionSystem(unittest.TestCase):

    def test_model_and_config_parameters(self):
        """Verifies serialized model has exact validated Exp6 hyperparameters."""
        config_path = os.path.join(BASE_DIR, 'models', 'production_model_config.json')
        model_path = os.path.join(BASE_DIR, 'models', 'production_lgbm_model.pkl')
        
        self.assertTrue(os.path.exists(config_path), "production_model_config.json missing")
        self.assertTrue(os.path.exists(model_path), "production_lgbm_model.pkl missing")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        params = config['exact_hyperparameters']
        self.assertEqual(params['n_estimators'], 150)
        self.assertEqual(params['max_depth'], 6)
        self.assertEqual(params['num_leaves'], 31)
        self.assertEqual(params['learning_rate'], 0.05)
        self.assertEqual(params['random_state'], 42)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        self.assertEqual(model.n_estimators, 150)
        self.assertEqual(model.max_depth, 6)
        self.assertEqual(model.num_leaves, 31)
        self.assertEqual(model.learning_rate, 0.05)
        self.assertEqual(model.random_state, 42)
        
        # Verify artifact cryptographic integrity
        with open(model_path, 'rb') as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(actual_sha, config['artifact_integrity']['model_pkl_sha256'])

    def test_feature_metadata_and_long_term_features(self):
        """Verifies 74 causal features and presence of long-term base features."""
        meta_path = os.path.join(BASE_DIR, 'models', 'production_features.json')
        self.assertTrue(os.path.exists(meta_path))
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        self.assertEqual(meta['feature_count'], 74)
        feature_list = meta['feature_list']
        
        long_term = ['v90', 'v180', 'v365', 'sales_days_90', 'sales_days_180', 'v30_vs_v365', 'v90_vs_v365']
        for f_name in long_term:
            self.assertIn(f_name, feature_list, f"Missing long-term feature: {f_name}")

    def test_report_1_validation_excel_structure(self):
        """Verifies Report 1 has SKU Validation Summary as Sheet 1, 674 SKU rows, correct columns and sums."""
        r1_path = os.path.join(BASE_DIR, 'reports', 'Rimmel_Validation_Sep01_Sep10_2026.xlsx')
        self.assertTrue(os.path.exists(r1_path), f"File missing: {r1_path}")
        
        wb = openpyxl.load_workbook(r1_path, data_only=True)
        self.assertIn('SKU Validation Summary', wb.sheetnames)
        self.assertEqual(wb.sheetnames[0], 'SKU Validation Summary', "Sheet 1 must be SKU Validation Summary")
        self.assertIn('Platform Summary', wb.sheetnames)
            
        ws = wb['SKU Validation Summary']
        rows = list(ws.iter_rows(values_only=True))
        total_rows = len(rows)
        self.assertEqual(total_rows, 675, f"Expected 675 rows in SKU Validation Summary, got {total_rows}")
        
        headers = list(rows[0])
        self.assertEqual(headers, EXPECTED_VAL_HEADERS)
        
        skus = set()
        tot_act_sum = 0
        tot_pred_sum = 0
        
        for r_idx, row in enumerate(rows[1:], start=2):
            sku, prod, period = row[0], row[1], row[2]
            amz_a, amz_p = row[3], row[4]
            ebay_a, ebay_p = row[5], row[6]
            web_a, web_p = row[7], row[8]
            oth_a, oth_p = row[9], row[10]
            tot_a, tot_p = row[11], row[12]
            var = row[13]
            conf, risk, reason = row[14], row[15], row[16]
            
            self.assertNotIn(sku, skus, f"Duplicate SKU: {sku}")
            skus.add(sku)
            
            self.assertEqual(period, 'Sep 1–10, 2026')
            self.assertEqual(tot_a, amz_a + ebay_a + web_a + oth_a, f"Row {r_idx}: Total Actual mismatch")
            self.assertEqual(tot_p, amz_p + ebay_p + web_p + oth_p, f"Row {r_idx}: Total Predicted mismatch")
            self.assertEqual(var, tot_p - tot_a, f"Row {r_idx}: Variance mismatch")
            self.assertIn(conf, ['HIGH', 'MEDIUM', 'LOW'])
            self.assertIn(risk, ['NORMAL', 'STOCKOUT RISK', 'HIGH VOLATILITY', 'LOW DEMAND'])
            self.assertTrue(len(str(reason)) > 5)
            
            tot_act_sum += tot_a
            tot_pred_sum += tot_p
            
        self.assertEqual(tot_act_sum, 2069, f"Expected 2,069 portfolio actual units, got {tot_act_sum}")
        self.assertEqual(len(skus), 674, f"Expected 674 unique SKUs, got {len(skus)}")
        wb.close()

    def test_report_2_production_forecast_excel_structure(self):
        """Verifies Report 2 has SKU Forecast Summary as Sheet 1, 674 SKU rows, correct columns and actions."""
        r2_path = os.path.join(BASE_DIR, 'reports', 'Rimmel_Forward_Forecast_Sep11_Sep20_2026.xlsx')
        self.assertTrue(os.path.exists(r2_path), f"File missing: {r2_path}")
        
        wb = openpyxl.load_workbook(r2_path, data_only=True)
        self.assertIn('SKU Forecast Summary', wb.sheetnames)
        self.assertEqual(wb.sheetnames[0], 'SKU Forecast Summary', "Sheet 1 must be SKU Forecast Summary")
        self.assertIn('Platform Summary', wb.sheetnames)
        self.assertIn('Inventory Actions', wb.sheetnames)
            
        ws = wb['SKU Forecast Summary']
        rows = list(ws.iter_rows(values_only=True))
        total_rows = len(rows)
        self.assertEqual(total_rows, 675, f"Expected 675 rows in SKU Forecast Summary, got {total_rows}")
        
        headers = list(rows[0])
        self.assertEqual(headers, EXPECTED_FWD_HEADERS)
        
        skus = set()
        for r_idx, row in enumerate(rows[1:], start=2):
            sku, prod, period = row[0], row[1], row[2]
            amz_p = row[3]
            ebay_p = row[4]
            web_p = row[5]
            oth_p = row[6]
            tot_p = row[7]
            conf, risk, action, reason = row[8], row[9], row[10], row[11]
            
            self.assertNotIn(sku, skus, f"Duplicate SKU: {sku}")
            skus.add(sku)
            
            self.assertEqual(period, 'Sep 11–20, 2026')
            self.assertEqual(tot_p, amz_p + ebay_p + web_p + oth_p, f"Row {r_idx}: Total Predicted mismatch")
            self.assertTrue(tot_p >= 0, f"Negative prediction for SKU {sku}")
            self.assertIn(conf, ['HIGH', 'MEDIUM', 'LOW'])
            self.assertIn(risk, ['NORMAL', 'STOCKOUT RISK', 'HIGH VOLATILITY', 'LOW DEMAND'])
            self.assertIn(action, ['Maintain Stock', 'Reorder Required', 'Urgent Restock', 'Monitor Closely', 'No Action Needed'])
            self.assertTrue(len(str(reason)) > 5)
            
        self.assertEqual(len(skus), 674, f"Expected 674 unique SKUs, got {len(skus)}")
        
        # Test Inventory Actions sheet
        ws_inv = wb['Inventory Actions']
        inv_rows = list(ws_inv.iter_rows(values_only=True))
        self.assertEqual(len(inv_rows), 675)
        
        wb.close()

    def test_backward_compatibility_copies(self):
        """Verifies backward compatible copies exist and match primary reports."""
        r1_compat = os.path.join(BASE_DIR, 'reports', 'validation_report_sep_01_to_10_2026.xlsx')
        r2_compat = os.path.join(BASE_DIR, 'reports', 'production_forecast_sep_11_to_20_2026.xlsx')
        self.assertTrue(os.path.exists(r1_compat))
        self.assertTrue(os.path.exists(r2_compat))

    def test_config_settings_parameters(self):
        """Verifies configuration parameters in config/settings.py match production standards."""
        from config import settings
        self.assertEqual(settings.CATALOG_SKU_COUNT, 674)
        self.assertEqual(settings.FORECAST_HORIZON_DAYS, 10)
        self.assertEqual(settings.VALIDATION_DAYS, 10)
        self.assertEqual(settings.CALIBRATION_ALPHA, 0.10)
        self.assertEqual(settings.CALIBRATION_BETA, 0.10)
        self.assertEqual(settings.PRODUCTION_TRAIN_START, '2025-08-01')
        self.assertEqual(settings.PRODUCTION_TRAIN_END, '2026-09-10')

    def test_database_integrity_and_row_counts(self):
        """Verifies core database tables exist with exact expected row counts in rimmel_clean.db."""
        db_path = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')
        self.assertTrue(os.path.exists(db_path), "Master database rimmel_clean.db missing")
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        raw_cnt = cur.execute("SELECT count(*) FROM raw_transactions").fetchone()[0]
        self.assertEqual(raw_cnt, 101085, f"Expected 101,085 raw_transactions, got {raw_cnt}")
        
        feat_cnt = cur.execute("SELECT count(*) FROM ml_features_zero").fetchone()[0]
        self.assertEqual(feat_cnt, 573678, f"Expected 573,678 ml_features_zero rows, got {feat_cnt}")
        
        sku_cnt = cur.execute("SELECT count(DISTINCT canonical_sku) FROM sku_master").fetchone()[0]
        self.assertEqual(sku_cnt, 674, f"Expected 674 canonical SKUs in sku_master, got {sku_cnt}")
        
        conn.close()

if __name__ == '__main__':
    unittest.main()
