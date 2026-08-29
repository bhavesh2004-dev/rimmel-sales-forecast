"""
PREPROCESSING MODULE
====================
Handles temporal window slicing, date validation, and in-stock flag verification
with strict zero-leakage enforcement.
"""
import pandas as pd

def preprocess_sales_data(sales_dataframe, start_date='2025-08-01', end_date='2026-07-20'):
    """
    Slices the historical dataframe to the exact training window and ensures data cleanliness.
    
    Args:
        sales_dataframe (pd.DataFrame): Full historical sales dataframe.
        start_date (str or pd.Timestamp): Earliest date to include in training.
        end_date (str or pd.Timestamp): Cutoff date (strictly inclusive).
        
    Returns:
        pd.DataFrame: Cleaned training slice with zero data leakage.
    """
    df_copy = sales_dataframe.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    start_ts = max(pd.to_datetime(start_date), pd.to_datetime('2025-08-01'))
    end_ts   = pd.to_datetime(end_date)
    
    # Strictly isolate data up to the cutoff date
    training_slice = df_copy[(df_copy['date'] >= start_ts) & (df_copy['date'] <= end_ts)].copy()
    
    # Ensure in-stock flag accurately reflects shelf stock on hand
    training_slice['in_stock_flag'] = (training_slice['current_stock'] > 0).astype(int)
    
    return training_slice
