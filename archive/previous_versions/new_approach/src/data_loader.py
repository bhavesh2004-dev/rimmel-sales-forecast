"""
Data Loader Module
Loads raw sales data, aggregates daily sales per SKU across channels, and constructs a continuous daily grid.
"""
import pandas as pd
import numpy as np

def load_and_aggregate_sku(file_path: str, sku_code: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df_sku = df[df['order_item_sku'] == sku_code].copy()
    df_sku['date'] = pd.to_datetime(df_sku['date'])
    
    daily_sales = df_sku.groupby('date')['sold_qty'].sum().reset_index()
    daily_stock = df_sku.groupby('date')['eod_stock'].min().reset_index()
    
    min_date = daily_sales['date'].min()
    max_date = daily_sales['date'].max()
    full_dates = pd.date_range(start=min_date, end=max_date, freq='D', name='date')
    
    data = pd.DataFrame({'date': full_dates})
    data = data.merge(daily_sales, on='date', how='left').merge(daily_stock, on='date', how='left')
    data['sold_qty'] = data['sold_qty'].fillna(0)
    data['is_stockout'] = (data['eod_stock'] == 0)
    
    return data
