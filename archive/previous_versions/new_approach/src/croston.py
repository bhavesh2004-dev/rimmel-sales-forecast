"""
Croston's Intermittent Demand Forecaster
Designed specifically for zero-heavy, low-volume SKUs (where zero sales ratio > 40%).
Models non-zero demand size and inter-arrival periods separately using Exponential Smoothing.
"""
import numpy as np
import pandas as pd

class CrostonForecaster:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.demand_size = 0.0
        self.inter_arrival = 1.0
        self.is_fitted = False
        
    def fit(self, y: pd.Series):
        y_vals = np.array(y)
        non_zero_indices = np.where(y_vals > 0)[0]
        
        if len(non_zero_indices) == 0:
            self.demand_size = 0.0
            self.inter_arrival = 1.0
            self.is_fitted = True
            return self
            
        # Initial values
        z = y_vals[non_zero_indices[0]]
        p = 1.0
        
        last_idx = non_zero_indices[0]
        
        for idx in non_zero_indices[1:]:
            interval = idx - last_idx
            z = z + self.alpha * (y_vals[idx] - z)
            p = p + self.alpha * (interval - p)
            last_idx = idx
            
        self.demand_size = z
        self.inter_arrival = max(1.0, p)
        self.is_fitted = True
        return self
        
    def predict(self, steps: int = 30) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
            
        if self.inter_arrival == 0:
            rate = 0.0
        else:
            rate = self.demand_size / self.inter_arrival
            
        return np.full(steps, max(0.0, rate))
