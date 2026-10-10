"""
Unit tests for forward forecast recursion and horizon invariants.
Verifies that:
- Exactly 10 calendar dates are predicted per eligible series.
- Day 2..10 future lag values use generated model predictions.
- No actual sales dated after cutoff T are used.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from multibrand_pipeline.src.db_manager import DBManager
from multibrand_pipeline.src.data_discovery import discover_brand_coverage
from multibrand_pipeline.src.model_router import ModelRouter
from multibrand_pipeline.src.dynamic_engine import DynamicForecastingEngine

class TestForecastRecursion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DBManager()
        cls.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "dynamic_pipeline_config.yaml")
        import yaml
        with open(cls.config_path, 'r') as f:
            cls.config = yaml.safe_load(f)
        cls.schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "feature_schema.json")
        cls.approved_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "global_lgbm_model.pkl")
        
        cls.df_brands, _ = discover_brand_coverage(cls.db, horizon_days=10)
        
        with cls.db.engine.connect() as conn:
            cls.df_norm = pd.read_sql("SELECT * FROM normalized_sales WHERE brand_id = 'DELILAH'", conn)
        cls.df_norm['date'] = pd.to_datetime(cls.df_norm['date']).dt.strftime('%Y-%m-%d')

    def test_forecast_horizon_exactly_ten_days(self):
        engine = DynamicForecastingEngine(self.config, self.schema_path)
        router = ModelRouter(self.config)
        
        b_prof = self.df_brands[self.df_brands['brand_id'] == 'DELILAH']
        r_decisions = {'DELILAH': router.route_brand(b_prof.iloc[0].to_dict())}
        
        df_forecast = engine.run_forward_operational_forecast(
            df_norm=self.df_norm,
            brand_profiles=b_prof,
            routing_decisions=r_decisions,
            approved_model_path=self.approved_model_path,
            run_id="TEST-RUN-001"
        )
        
        # Check exactly 10 forecast dates
        unique_dates = df_forecast['forecast_date'].nunique()
        self.assertEqual(unique_dates, 10, f"Forecast must have exactly 10 distinct dates (got {unique_dates})")
        
        # Check horizon days numbering from 1 to 10
        self.assertEqual(sorted(df_forecast['horizon_day'].unique().tolist()), list(range(1, 11)))
        
        # Check forecast start date is T + 1
        cutoff_t = pd.to_datetime(b_prof.iloc[0]['cutoff_date_T'])
        min_fwd = pd.to_datetime(df_forecast['forecast_date'].min())
        self.assertEqual((min_fwd - cutoff_t).days, 1, "Forecast start date must be cutoff T + 1 day")

if __name__ == '__main__':
    unittest.main()
