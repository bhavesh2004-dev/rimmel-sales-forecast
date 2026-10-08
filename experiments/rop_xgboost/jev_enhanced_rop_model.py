"""
Jev-Enhanced XGBoost ROP Model (Model B: Structured + Jev Context)
==================================================================
Trains an XGBoost Quantile Regressor on Structured Features augmented with
TypeSafe AI Jev-derived structured context features.
Predicts the 95th percentile of lead-time cumulative demand (ROP).
"""
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple
from experiments.rop_xgboost.config import (
    XGB_QUANTILE_PARAMS, TARGET_SERVICE_LEVEL, OUTPUTS_DIR
)
from experiments.rop_xgboost.rop_feature_builder import (
    BASE_FEATURE_COLS, JEV_FEATURE_COLS
)


class JevEnhancedXGBoostROP:
    """Model B: Structured Features + TypeSafe AI Jev Context Features."""

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or XGB_QUANTILE_PARAMS.copy()
        self.feature_cols = BASE_FEATURE_COLS + ['platform_cat'] + JEV_FEATURE_COLS
        self.model = xgb.XGBRegressor(**self.params)
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> 'JevEnhancedXGBoostROP':
        print(f"[MODEL B - JEV ENHANCED] Fitting XGBoost Quantile Regressor ({len(self.feature_cols)} features)...")
        X = df_train[self.feature_cols]
        y = df_train['ltd_target']
        
        self.model.fit(X, y)
        self.is_fitted = True
        print("  Model B training complete.")
        return self

    def predict_rop(self, df_eval: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model B must be fitted before predict_rop.")
        X = df_eval[self.feature_cols]
        raw_preds = self.model.predict(X)
        return np.maximum(0.0, raw_preds)

    def evaluate_pinball_loss(self, df_eval: pd.DataFrame, alpha: float = TARGET_SERVICE_LEVEL) -> Dict[str, float]:
        y_true = df_eval['ltd_target'].values
        y_pred = self.predict_rop(df_eval)
        
        errors = y_true - y_pred
        loss = np.maximum(alpha * errors, (alpha - 1.0) * errors)
        mean_pinball_loss = float(np.mean(loss))
        
        coverage_rate = float(np.mean(y_true <= y_pred) * 100.0)
        
        return {
            'mean_pinball_loss': round(mean_pinball_loss, 4),
            'empirical_coverage_pct': round(coverage_rate, 2),
            'mean_predicted_rop': round(float(np.mean(y_pred)), 2)
        }

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model B must be fitted.")
        scores = self.model.feature_importances_
        df_imp = pd.DataFrame({
            'feature': self.feature_cols,
            'is_jev_feature': [f in JEV_FEATURE_COLS for f in self.feature_cols],
            'importance_gain': scores
        }).sort_values('importance_gain', ascending=False)
        return df_imp

    def save(self, filepath: str = None):
        path = filepath or os.path.join(OUTPUTS_DIR, 'jev_enhanced_rop_xgb.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  Saved Jev-Enhanced Model B to {path}")
