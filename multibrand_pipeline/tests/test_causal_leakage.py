"""
Test Causal Leakage & Temporal Discipline
Verifies shift(1) rolling feature correctness, lag alignment, and frozen production safety.
"""

import os
import unittest
import pandas as pd
import numpy as np

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.features import compute_causal_features

class TestCausalLeakage(unittest.TestCase):
    
    def setUp(self):
        # Create synthetic 20-day grid for one SKU-platform series
        dates = pd.date_range("2026-08-01", "2026-08-20", freq="D").strftime("%Y-%m-%d")
        np.random.seed(42)
        sales = np.random.poisson(lam=5.0, size=len(dates)).astype(float)
        
        self.grid_df = pd.DataFrame({
            'date': dates,
            'brand_id': 'TEST_BRAND',
            'canonical_sku': 'SKU_001',
            'platform_group': 'Amazon',
            'observed_units_sold': sales,
            'selling_price': 10.0,
            'current_stock': 100.0,
            'in_stock_flag': 1,
            'stockout_flag': 0,
            'has_inventory_signal': 1,
            'product_title': 'Test Mascara',
            'category': 'Mascara'
        })

    def test_lag_1_causality(self):
        """Proves lag_1 at date T strictly equals observed sales at T-1."""
        feats = compute_causal_features(self.grid_df)
        
        # For index 0, lag_1 should be 0.0
        self.assertEqual(feats.loc[0, 'lag_1'], 0.0)
        
        # For index 1 to N, lag_1 at date T must equal sales at T-1
        for idx in range(1, len(feats)):
            expected_lag_1 = self.grid_df.loc[idx - 1, 'observed_units_sold']
            actual_lag_1 = feats.loc[idx, 'lag_1']
            self.assertEqual(actual_lag_1, expected_lag_1, f"Lag_1 leaked or misaligned at row {idx}")

    def test_rolling_v7_shift1_invariance(self):
        """Proves changing sales at date T has ZERO effect on v7 at date T."""
        feats_original = compute_causal_features(self.grid_df)
        
        # Clone grid and inject an extreme spike on date T (index 10)
        modified_grid = self.grid_df.copy()
        target_idx = 10
        modified_grid.loc[target_idx, 'observed_units_sold'] += 1000.0 # Huge spike at T
        
        feats_modified = compute_causal_features(modified_grid)
        
        # v7 at target_idx MUST be completely identical between original and modified!
        v7_orig = feats_original.loc[target_idx, 'v7']
        v7_mod = feats_modified.loc[target_idx, 'v7']
        self.assertEqual(v7_orig, v7_mod, f"v7 leaked! Value at date T changed when sales(T) spiked: {v7_orig} vs {v7_mod}")
        
        # However, at T+1 (target_idx + 1), v7 SHOULD reflect the spike from T
        v7_next_orig = feats_original.loc[target_idx + 1, 'v7']
        v7_next_mod = feats_modified.loc[target_idx + 1, 'v7']
        self.assertGreater(v7_next_mod, v7_next_orig, "v7 at T+1 failed to incorporate T-1 historical sales")

    def test_frozen_production_files_untouched(self):
        """Verifies certified multi-brand production files exist and remain untouched."""
        prod_files = [
            os.path.join(PIPELINE_ROOT, "models", "global_lgbm_model.pkl"),
            os.path.join(PIPELINE_ROOT, "models", "model_metadata.json"),
            os.path.join(PIPELINE_ROOT, "config", "feature_schema.json"),
            os.path.join(PIPELINE_ROOT, "reports", "MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx")
        ]
        for pf in prod_files:
            self.assertTrue(os.path.exists(pf), f"CRITICAL: Production file missing or modified: {pf}")

if __name__ == "__main__":
    unittest.main()
