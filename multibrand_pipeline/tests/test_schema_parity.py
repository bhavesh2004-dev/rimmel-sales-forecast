"""
Test Schema Parity, Brand Generalization, and Ingestion Normalization
Verifies multi-brand handling, future brand compatibility, canonical SKU mappings, and platform grouping.
"""

import os
import unittest
import json
import yaml
import pandas as pd

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from src.ingestion import resolve_column_mapping, normalize_header
from src.normalization import normalize_brand_name, map_platform, resolve_canonical_sku, normalize_transactions

class TestSchemaParity(unittest.TestCase):

    def setUp(self):
        config_path = os.path.join(PIPELINE_ROOT, "config", "pipeline_config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def test_future_brand_compatibility(self):
        """Proves future brand (e.g. Bourjois) is accepted dynamically without code changes."""
        raw_brands = ["Bourjois Paris", "COTY Luxury", "NewBrand123"]
        for rb in raw_brands:
            b_id, d_name = normalize_brand_name(rb)
            self.assertNotIn("UNKNOWN", b_id)
            self.assertTrue(len(b_id) > 2)
            self.assertEqual(d_name, rb)

    def test_canonical_sku_mapping(self):
        """Proves multiple listing marketplace SKUs map to single shared canonical SKU."""
        amz_listing = "RIM-100WP-BLK-FBA"
        ebay_listing = "RIM-100WP-BLK-MFN"
        actual_sku = "RIM-100WP-BLK"
        
        canon_1 = resolve_canonical_sku(amz_listing, actual_sku)
        canon_2 = resolve_canonical_sku(ebay_listing, actual_sku)
        canon_3 = resolve_canonical_sku("RIM-100WP-BLK-AMZ", None)
        
        self.assertEqual(canon_1, actual_sku)
        self.assertEqual(canon_2, actual_sku)
        self.assertEqual(canon_3, actual_sku)

    def test_platform_normalization(self):
        """Proves varied channel aliases correctly resolve to the 4 standard platforms."""
        plat_map = self.config['platform_mapping']
        
        self.assertEqual(map_platform("Bellas Beauty Ebay - MFN", plat_map), "eBay")
        self.assertEqual(map_platform("Amazon UK FBA", plat_map), "Amazon")
        self.assertEqual(map_platform("Glam TTS - MFN", plat_map), "Website")
        self.assertEqual(map_platform("Shopify Direct D2C", plat_map), "Website")
        self.assertEqual(map_platform("Wholesale Sample", plat_map), "Other")

    def test_model_feature_schema_count(self):
        """Verifies feature_schema.json contains exactly 60 features and zero pruned features."""
        schema_path = os.path.join(PIPELINE_ROOT, "config", "feature_schema.json")
        with open(schema_path, "r") as f:
            schema = json.load(f)
            
        self.assertEqual(schema['total_features'], 60)
        self.assertEqual(len(schema['feature_list']), 60)
        self.assertEqual(len(schema['pruned_features']), 15)
        
        # Verify pruned features are not in feature_list
        for pf in schema['pruned_features']:
            self.assertNotIn(pf, schema['feature_list'], f"Pruned dead feature {pf} found in model feature list!")

    def test_training_benchmark_temporal_separation(self):
        """Verifies training end date is strictly prior to benchmark start date."""
        timeline = self.config['timeline']
        train_end = timeline['training_end']
        bench_start = timeline['benchmark_start']
        self.assertLess(train_end, bench_start, f"Training end {train_end} overlaps or exceeds benchmark start {bench_start}")

    def test_feature_order_and_categoricals(self):
        """Verifies categorical features exist in feature_list and have valid definitions."""
        schema_path = os.path.join(PIPELINE_ROOT, "config", "feature_schema.json")
        with open(schema_path, "r") as f:
            schema = json.load(f)
            
        cat_feats = schema['categorical_features']
        feat_list = schema['feature_list']
        
        for cf in cat_feats:
            self.assertIn(cf, feat_list, f"Categorical feature {cf} missing from feature_list")

if __name__ == "__main__":
    unittest.main()
