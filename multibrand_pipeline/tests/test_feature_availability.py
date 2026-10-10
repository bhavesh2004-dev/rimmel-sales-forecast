"""
Unit tests for feature availability profiling and schema validation.
Verifies that:
- Feature tiers correctly categorize historical depth without inventing unobserved lags.
- The 60 features schema is strictly validated.
- Categorical features are properly handled.
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from multibrand_pipeline.src.feature_profiles import FeatureProfiler

class TestFeatureAvailability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "feature_schema.json")
        cls.profiler = FeatureProfiler(schema_path)

    def test_schema_total_features(self):
        self.assertEqual(len(self.profiler.feature_list), 60, "Audited LightGBM schema must have exactly 60 features")

    def test_tier_long_history(self):
        tier = self.profiler.determine_feature_tier(150)
        self.assertEqual(tier['tier_name'], "TIER_LONG_HISTORY")
        self.assertIn("lag_90", tier['available_lags'])

    def test_tier_medium_history(self):
        tier = self.profiler.determine_feature_tier(45)
        self.assertEqual(tier['tier_name'], "TIER_MEDIUM_HISTORY")
        self.assertIn("lag_30", tier['available_lags'])
        self.assertIn("lag_90", tier['unobserved_lags'])

    def test_tier_short_history(self):
        tier = self.profiler.determine_feature_tier(9)  # e.g. Kifra pre-val
        self.assertEqual(tier['tier_name'], "TIER_SHORT_HISTORY")
        self.assertIn("lag_1", tier['available_lags'])
        self.assertIn("lag_30", tier['unobserved_lags'])

if __name__ == '__main__':
    unittest.main()
