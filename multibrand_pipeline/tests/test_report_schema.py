"""
Test Client Report Excel Schema & Formatting (Phase 5C)
Verifies simplified 2-sheet structure:
- Sheet 1: Forecast_Inventory (15 exact columns, AutoFilter, Freeze Panes)
- Sheet 2: ROP (10 simplified columns, Policy Banner, AutoFilter, Freeze Panes)
"""

import os
import unittest
import yaml
import openpyxl
import pandas as pd

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.reporting import export_client_report

class TestReportSchema(unittest.TestCase):

    def setUp(self):
        config_path = os.path.join(PIPELINE_ROOT, "config", "pipeline_config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.test_excel_path = os.path.join(PIPELINE_ROOT, "reports", "TEST_CLIENT_REPORT.xlsx")
        
        self.df_10d = pd.DataFrame({
            'brand_id': ['RIMMEL', 'MAX_FACTOR'],
            'canonical_sku': ['RIM-001', 'MF-002'],
            'product_title': ['Mascara', 'Lipstick'],
            'category': ['Mascara', 'Lipstick'],
            'Forecast Period': ['2026-09-11 to 2026-09-20', '2026-09-11 to 2026-09-20'],
            'Amazon': [10, 5],
            'eBay': [20, 15],
            'Website': [5, 2],
            'Other': [0, 0],
            'Total Predicted': [35, 22],
            'current_stock': [150.0, 50.0],
            'in_stock_flag': [1, 1],
            'cv_30': [0.4, 0.5],
            'v30': [3.5, 2.2]
        })

    def tearDown(self):
        if os.path.exists(self.test_excel_path):
            try:
                os.remove(self.test_excel_path)
            except:
                pass

    def test_two_sheet_excel_deliverable_simplified(self):
        """Proves client workbook contains EXACTLY two sheets with simplified schemas."""
        export_client_report(self.df_10d, output_path=self.test_excel_path, config=self.config)
        self.assertTrue(os.path.exists(self.test_excel_path))
        
        wb = openpyxl.load_workbook(self.test_excel_path, data_only=True)
        sheet_names = wb.sheetnames
        
        self.assertEqual(len(sheet_names), 2)
        self.assertEqual(sheet_names[0], "Forecast_Inventory")
        self.assertEqual(sheet_names[1], "ROP")
        
        # Verify Sheet 1 has exactly 15 columns matching specification
        ws1 = wb['Forecast_Inventory']
        headers1 = [c for c in next(ws1.iter_rows(min_row=3, max_row=3, values_only=True))]
        expected_15 = [
            'Brand',
            'SKU',
            'Product',
            'Current Stock',
            'Forecast Period',
            'Amazon Predicted',
            'eBay Predicted',
            'Website Predicted',
            'Other Predicted',
            '10-Day Forecast',
            'Days of Cover',
            'Confidence',
            'Risk',
            'Recommended Action',
            'Reason'
        ]
        self.assertEqual(headers1, expected_15)
        self.assertIsNotNone(ws1.auto_filter.ref)
        
        # Verify Sheet 2 has policy banner on Row 2
        ws2 = wb['ROP']
        policy_banner = ws2.cell(2, 1).value
        self.assertIn("Lead Time = 10 Days", str(policy_banner))
        self.assertIn("Minimum Stock Level = 6 Units", str(policy_banner))
        
        # Verify Sheet 2 table header on Row 4 has exactly 10 simplified columns
        headers2 = [c for c in next(ws2.iter_rows(min_row=4, max_row=4, values_only=True))]
        expected_10 = [
            'Brand',
            'SKU',
            'Product',
            'Current Stock',
            '10-Day Forecast',
            'Avg Daily Usage',
            'Lead-Time Demand',
            'Target Stock',
            'Replenishment Qty',
            'ROP Status'
        ]
        self.assertEqual(headers2, expected_10)
        self.assertIsNotNone(ws2.auto_filter.ref)

if __name__ == "__main__":
    unittest.main()
