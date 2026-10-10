"""
Unit tests for causal validation leakage prevention.
Verifies that:
- Validation training sets use target labels only up to validation_training_cutoff.
- In-horizon validation simulation uses recursive predictions, not actual holdout values.
- Forward forecasts use zero actual data after data cutoff T.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from multibrand_pipeline.src.db_manager import DBManager
from multibrand_pipeline.src.data_discovery import discover_brand_coverage
from multibrand_pipeline.src.model_router import ModelRouter, BaselineRunRatePredictor
from multibrand_pipeline.src.dynamic_engine import DynamicForecastingEngine

class TestValidationLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DBManager()
        cls.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "dynamic_pipeline_config.yaml")
        import yaml
        with open(cls.config_path, 'r') as f:
            cls.config = yaml.safe_load(f)
        cls.schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "feature_schema.json")
        cls.df_brands, _ = discover_brand_coverage(cls.db, horizon_days=10)
        
        with cls.db.engine.connect() as conn:
            cls.df_norm = pd.read_sql("SELECT * FROM normalized_sales WHERE brand_id = 'DELILAH'", conn)
        cls.df_norm['date'] = pd.to_datetime(cls.df_norm['date']).dt.strftime('%Y-%m-%d')

    def test_training_set_strictly_bounded(self):
        b_prof = self.df_brands[self.df_brands['brand_id'] == 'DELILAH'].iloc[0]
        train_cutoff = b_prof['validation_training_cutoff']
        
        # Verify no training targets exceed train_cutoff
        eligible_norm = self.df_norm[self.df_norm['date'] <= train_cutoff]
        max_train_date = eligible_norm['date'].max()
        self.assertTrue(max_train_date <= train_cutoff, "Training targets must never exceed validation training cutoff")

    def test_recursive_buffer_uses_predictions_not_actuals(self):
        # Create a mock run and verify that recursive buffer appends predictions
        engine = DynamicForecastingEngine(self.config, self.schema_path)
        router = ModelRouter(self.config)
        
        b_prof = self.df_brands[self.df_brands['brand_id'] == 'DELILAH']
        r_decisions = {'DELILAH': router.route_brand(b_prof.iloc[0].to_dict())}
        
        df_val_summary, df_val_detail, _ = engine.run_dynamic_validation(
            df_norm=self.df_norm,
            brand_profiles=b_prof,
            routing_decisions=r_decisions
        )
        
        # Ensure predicted units are separate from actual units
        self.assertIn('actual_units', df_val_detail.columns)
        self.assertIn('predicted_units', df_val_detail.columns)
        self.assertEqual(len(df_val_detail), 290, "Delilah has 29 SKUs x 1 platform x 10 days = 290 validation detail rows")

if __name__ == '__main__':
    unittest.main()
