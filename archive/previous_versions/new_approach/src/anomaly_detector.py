"""
Anomaly Detector Module
Computes 30-day rolling mean & std dev to detect sales spikes and flags out-of-stock days.
"""
import pandas as pd
import numpy as np

def detect_anomalies(df: pd.DataFrame, rolling_window: int = 30, std_multiplier: float = 3.0) -> pd.DataFrame:
    data = df.copy()
    data['rolling_mean_30'] = data['sold_qty'].rolling(window=rolling_window, min_periods=7).mean()
    data['rolling_std_30'] = data['sold_qty'].rolling(window=rolling_window, min_periods=7).std().fillna(0)
    data['spike_threshold'] = data['rolling_mean_30'] + std_multiplier * data['rolling_std_30']
    data['is_spike'] = data['sold_qty'] > data['spike_threshold']
    
    # Winsorize / cap spikes for training to avoid distorting baseline demand
    data['target_sales'] = np.where(data['is_spike'], data['spike_threshold'], data['sold_qty'])
    return data
