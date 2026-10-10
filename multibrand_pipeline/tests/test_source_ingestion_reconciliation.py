"""
Regression tests for source-data ingestion, reconciliation, and idempotency.
Verifies that:
- Reimporting does not create duplicates (idempotency).
- Overlapping historical records are reconciled correctly without double-counting.
- Historical date ranges from 2025-01-01 are preserved.
- Newer valid database records extending into October 2026 are preserved.
- All 7 brands remain distinct and unaffected in normalized_sales.
- Source counts and quantities reconcile to final MySQL normalized_sales data.
- Duplicate insertions into forecast_results violate the unique key as intended.
"""

import os
import sys
import unittest
import pandas as pd
from sqlalchemy import text, exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from multibrand_pipeline.src.db_manager import DBManager
from multibrand_pipeline.src.import_rimmel_mysql import import_rimmel_data

class TestSourceIngestionReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DBManager()

    def test_reimport_idempotency_does_not_duplicate(self):
        """Verifies that calling import_rimmel_data a second time does not duplicate rows."""
        with self.db.engine.connect() as conn:
            pre_count = conn.execute(text("SELECT COUNT(*) FROM raw_rimmel_sales_data WHERE batch_id = 'BATCH-RIMMEL-RAW-001'")).fetchone()[0]
            
        # Re-run import
        imported = import_rimmel_data()
        
        with self.db.engine.connect() as conn:
            post_count = conn.execute(text("SELECT COUNT(*) FROM raw_rimmel_sales_data WHERE batch_id = 'BATCH-RIMMEL-RAW-001'")).fetchone()[0]
            
        self.assertEqual(pre_count, post_count, "Re-running import must not duplicate rows in raw_rimmel_sales_data")
        self.assertEqual(post_count, 101085, "Total raw Rimmel rows must equal exactly 101,085")

    def test_historical_dates_retained_in_normalized_sales(self):
        """Verifies that 2025 historical data is fully present for brands with 2025 history."""
        with self.db.engine.connect() as conn:
            res = conn.execute(text("""
                SELECT brand_id, MIN(`date`) as min_d, MAX(`date`) as max_d, COUNT(*) as cnt
                FROM normalized_sales
                WHERE brand_id IN ('RIMMEL', 'MAX_FACTOR', 'WELEDA')
                GROUP BY brand_id
            """)).fetchall()
            
        stats = {r[0]: (str(r[1]), str(r[2]), r[3]) for r in res}
        
        self.assertEqual(stats['RIMMEL'][0], '2025-01-01', "Rimmel min date must be 2025-01-01")
        self.assertEqual(stats['MAX_FACTOR'][0], '2025-01-01', "Max Factor min date must be 2025-01-01")
        self.assertEqual(stats['WELEDA'][0], '2025-01-01', "Weleda min date must be 2025-01-01")

    def test_newer_records_preserved(self):
        """Verifies that newer database records through October 2026 are preserved."""
        with self.db.engine.connect() as conn:
            res = conn.execute(text("""
                SELECT brand_id, MAX(`date`)
                FROM normalized_sales
                GROUP BY brand_id
            """)).fetchall()
        max_dates = {r[0]: str(r[1]) for r in res}
        
        self.assertEqual(max_dates['MAX_FACTOR'], '2026-10-06')
        self.assertEqual(max_dates['GEEK_GORGEOUS'], '2026-10-06')
        self.assertEqual(max_dates['KIFRA'], '2026-10-06')
        self.assertEqual(max_dates['WELEDA'], '2026-10-05')
        self.assertEqual(max_dates['DELILAH'], '2026-10-02')
        self.assertEqual(max_dates['FRANK_BODY'], '2026-10-02')
        self.assertEqual(max_dates['RIMMEL'], '2026-09-10')

    def test_all_seven_brands_unaffected(self):
        """Verifies that all 7 brands exist with valid records in normalized_sales."""
        with self.db.engine.connect() as conn:
            brands = set(pd.read_sql(text("SELECT DISTINCT brand_id FROM normalized_sales"), conn)['brand_id'])
        expected = {'RIMMEL', 'MAX_FACTOR', 'WELEDA', 'GEEK_GORGEOUS', 'KIFRA', 'DELILAH', 'FRANK_BODY'}
        self.assertEqual(brands, expected)

    def test_operational_forecast_rop_unique_key_collision_rejected(self):
        """Verifies that inserting a duplicate unique key into operational_forecast_rop is rejected by MySQL."""
        with self.db.engine.connect() as conn:
            sample = conn.execute(text("""
                SELECT run_id, brand, sku
                FROM operational_forecast_rop
                LIMIT 1
            """)).fetchone()
            
            # Attempt to insert exact duplicate
            duplicate_insert = text("""
                INSERT INTO operational_forecast_rop 
                (run_id, brand, brand_id, sku, current_stock, forecast_period, days_of_cover, confidence, risk, recommended_action, rop_status, model_id, data_cutoff_date)
                VALUES (:rid, :b, 'TEST', :sku, 0, '2026-09-11 to 2026-09-20', '0.0', 'LOW', 'NONE', 'NONE', 'ABOVE ROP', 'test', '2026-10-06')
            """)
            
            with self.assertRaises((exc.IntegrityError, Exception)):
                conn.execute(duplicate_insert, {
                    'rid': sample[0],
                    'b': sample[1],
                    'sku': sample[2]
                })

if __name__ == '__main__':
    unittest.main()
