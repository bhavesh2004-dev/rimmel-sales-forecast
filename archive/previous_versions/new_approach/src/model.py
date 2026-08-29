"""
Model Training Module
Defines an ensemble forecaster (LightGBM + Random Forest) for smooth 30-day sales forecasting.
"""
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class SalesForecasterEnsemble:
    def __init__(self, random_state: int = 42):
        self.model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.03, random_state=random_state, verbose=-1)
        self.model_rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state)
        self.is_fitted = False
        
    def fit(self, X, y):
        self.model_lgb.fit(X, y)
        self.model_rf.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        pred_lgb = self.model_lgb.predict(X)
        pred_rf = self.model_rf.predict(X)
        return 0.5 * pred_lgb + 0.5 * pred_rf
        
    def get_feature_importances(self):
        return self.model_rf.feature_importances_
