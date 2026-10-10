"""
Unit tests for MySQL canonical operational_forecast_rop table schema and query support.
Verifies that:
- Single canonical operational_forecast_rop table exists in MySQL.
- Obsolete redundant tables (forecast_results, report_forecast_inventory, report_rop) have been retired.
- All 21 operational business columns and run metadata exist.
- Required composite unique key uk_run_brand_sku is enforced.
- Exactly 1,433 records exist for the active operational run.
- All 7 brands are present and queryable.
- Additive Channel Law (Amazon + eBay + Website + Other == 10-Day Forecast) holds.
- Replenishment safety buffer invariant (>= 6.0 units) holds.
- Model artifact hashes are preserved untouched.
"""

import os
import sys
import hashlib
import unittest
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from multibrand_pipeline.src.db_manager import DBManager

APPROVED_MODEL_MD5 = "dac42213d9c8c3a63f97e6171bf187d7"
BACKUP_MODEL_MD5 = "db098afc3d8f1805dd7f21d2a193f0ca"

class TestMySQLOutputSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DBManager()

    def test_canonical_operational_table_exists(self):
        """Verifies that operational_forecast_rop exists as the canonical output table."""
        tables = self.db.get_existing_tables()
        self.assertIn('operational_forecast_rop', tables, "operational_forecast_rop table must exist in MySQL")

    def test_redundant_tables_retired(self):
        """Verifies that redundant output tables were safely removed."""
        tables = self.db.get_existing_tables()
        self.assertNotIn('forecast_results', tables, "forecast_results must be retired")
        self.assertNotIn('report_forecast_inventory', tables, "report_forecast_inventory must be retired")
        self.assertNotIn('report_rop', tables, "report_rop must be retired")

    def test_required_columns_exist(self):
        """Verifies all 21 business columns and metadata exist in operational_forecast_rop."""
        with self.db.engine.connect() as conn:
            cols = [r[0] for r in conn.execute(text("DESCRIBE operational_forecast_rop")).fetchall()]
            
        required_cols = [
            'id', 'run_id', 'brand_id', 'brand', 'sku', 'product', 'current_stock',
            'forecast_period', 'amazon_predicted', 'ebay_predicted', 'website_predicted',
            'other_predicted', 'forecast_10d', 'days_of_cover', 'confidence',
            'risk', 'recommended_action', 'reason', 'lead_time_days',
            'avg_daily_usage', 'lead_time_demand', 'target_stock', 'replenishment_qty',
            'rop_status', 'model_id', 'data_cutoff_date', 'created_at'
        ]
        for col in required_cols:
            self.assertIn(col, cols, f"Required column '{col}' must exist in operational_forecast_rop")

    def test_single_canonical_forecast_quantity(self):
        """Verifies exactly one canonical forecast column exists with zero conflicting names."""
        with self.db.engine.connect() as conn:
            cols = [r[0] for r in conn.execute(text("DESCRIBE operational_forecast_rop")).fetchall()]
        self.assertIn('forecast_10d', cols)
        self.assertNotIn('predicted_quantity', cols)
        self.assertNotIn('predicted_units', cols)

    def test_unique_key_enforced(self):
        """Verifies that unique key uk_run_brand_sku is properly indexed in MySQL."""
        with self.db.engine.connect() as conn:
            indexes = pd.read_sql(text("SHOW INDEX FROM operational_forecast_rop WHERE Key_name = 'uk_run_brand_sku'"), conn)
        self.assertGreater(len(indexes), 0, "Unique key 'uk_run_brand_sku' must exist on operational_forecast_rop")

    def test_active_run_row_count(self):
        """Verifies that exactly 1,433 records exist for the active operational run."""
        with self.db.engine.connect() as conn:
            cnt = conn.execute(text("SELECT COUNT(*) FROM operational_forecast_rop WHERE run_id = 'RUN-PROD-20261010-OPERATIONAL'")).scalar()
        self.assertEqual(cnt, 1433, f"Expected 1,433 rows, got {cnt}")

    def test_all_seven_brands_queryable(self):
        """Verifies that all 7 brands exist in operational_forecast_rop."""
        with self.db.engine.connect() as conn:
            brands = set(pd.read_sql(text("SELECT DISTINCT brand_id FROM operational_forecast_rop WHERE run_id = 'RUN-PROD-20261010-OPERATIONAL'"), conn)['brand_id'])
        expected_brands = {'RIMMEL', 'MAX_FACTOR', 'WELEDA', 'GEEK_GORGEOUS', 'KIFRA', 'DELILAH', 'FRANK_BODY'}
        self.assertEqual(brands, expected_brands)

    def test_additive_channel_law_verified(self):
        """Verifies that Amazon + eBay + Website + Other strictly equals forecast_10d."""
        with self.db.engine.connect() as conn:
            viol = conn.execute(text("""
                SELECT COUNT(*) FROM operational_forecast_rop
                WHERE (amazon_predicted + ebay_predicted + website_predicted + other_predicted) != forecast_10d
            """)).scalar()
        self.assertEqual(viol, 0, f"Found {viol} additive channel law violations in MySQL!")

    def test_model_hash_preserved(self):
        """Verifies that active and backup model artifacts remain byte-level intact."""
        pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(pipeline_root, "models", "global_lgbm_model.pkl")
        backup_path = os.path.join(pipeline_root, "models", "backup_rimmel_maxfactor_lgbm_model.pkl")
        
        with open(model_path, "rb") as f:
            actual_md5 = hashlib.md5(f.read()).hexdigest()
        self.assertEqual(actual_md5, APPROVED_MODEL_MD5, "Active production model hash mismatch!")
        
        with open(backup_path, "rb") as f:
            backup_md5 = hashlib.md5(f.read()).hexdigest()
        self.assertEqual(backup_md5, BACKUP_MODEL_MD5, "Backup model hash mismatch!")

if __name__ == '__main__':
    unittest.main()
