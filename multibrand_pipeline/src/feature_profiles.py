"""
Feature Availability Profiler and Schema Compatibility Module.
Categorizes feature availability tiers (Long, Medium, Short) without inventing unobserved history.
Enforces strict causal temporal discipline and validates feature schema alignment for LightGBM.
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FeatureProfiler:
    def __init__(self, schema_path: str):
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Feature schema not found: {schema_path}")
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
            
        self.feature_list = self.schema['feature_list']
        self.categorical_features = self.schema['categorical_features']
        self.schema_version = self.schema.get('schema_version', '2.0.0')

    def determine_feature_tier(self, available_history_days: int) -> Dict[str, Any]:
        """
        Assigns an evidence-based feature availability profile based on genuine historical depth.
        """
        if available_history_days >= 90:
            tier_name = "TIER_LONG_HISTORY"
            available_lags = ["lag_1", "lag_7", "lag_14", "lag_30", "lag_90"]
            unobserved_lags = ["lag_180", "lag_365"] if available_history_days < 365 else []
            description = "Full feature profile available. Lags up to 90d and rolling velocities up to 90d are genuine."
        elif available_history_days >= 30:
            tier_name = "TIER_MEDIUM_HISTORY"
            available_lags = ["lag_1", "lag_7", "lag_14", "lag_30"]
            unobserved_lags = ["lag_90", "lag_180", "lag_365"]
            description = "Medium feature profile. Lags up to 30d available. Lags 90d+ are unobserved."
        else:
            tier_name = "TIER_SHORT_HISTORY"
            available_lags = ["lag_1", "lag_7"] if available_history_days >= 7 else (["lag_1"] if available_history_days >= 1 else [])
            unobserved_lags = ["lag_14", "lag_30", "lag_90", "lag_180", "lag_365"]
            description = f"Short history profile ({available_history_days}d). Multi-week lags unobserved; ML features incomplete."
            
        return {
            "tier_name": tier_name,
            "available_history_days": available_history_days,
            "available_lags": available_lags,
            "unobserved_lags": unobserved_lags,
            "description": description,
            "schema_feature_count": len(self.feature_list)
        }

    def validate_inference_schema(self, X: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validates that an inference DataFrame matches the exact feature schema required by LightGBM.
        """
        missing_cols = [col for col in self.feature_list if col not in X.columns]
        extra_cols = [col for col in X.columns if col not in self.feature_list]
        
        errors = []
        if missing_cols:
            errors.append(f"Missing {len(missing_cols)} schema columns: {missing_cols[:5]}...")
        if extra_cols:
            logger.debug(f"Input has {len(extra_cols)} extra non-schema columns (will be filtered).")
            
        is_valid = len(missing_cols) == 0
        return is_valid, errors

    def prepare_lgbm_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selects schema columns in exact order and casts categorical variables.
        """
        X = df[self.feature_list].copy()
        for cat_col in self.categorical_features:
            if cat_col in X.columns:
                X[cat_col] = X[cat_col].astype('category')
        return X
