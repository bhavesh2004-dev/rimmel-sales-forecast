"""
Multi-SKU Data Cleaner Module (Version 3.0)
Extracts top 20 SKUs from the new dataset (Rimmel New.xlsx),
cleanses data irregularities, handles price & category features,
and imputes stockouts & statistical spikes.
"""
import os
import pandas as pd
import numpy as np

DEFAULT_NEW_DATASET = 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx'

def get_top_20_skus(file_path: str) -> list:
    df = pd.read_excel(file_path)
    sku_col = 'sku' if 'sku' in df.columns else 'order_item_sku'
    qty_col = 'sold_quantity' if 'sold_quantity' in df.columns else 'sold_qty'
    
    sku_sales = df.groupby(sku_col)[qty_col].sum().reset_index()
    top_20 = sku_sales.sort_values(by=qty_col, ascending=False).head(20)[sku_col].tolist()
    return top_20

def process_and_clean_sku(df_raw: pd.DataFrame, sku_code: str, min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple:
    sku_col = 'sku' if 'sku' in df_raw.columns else 'order_item_sku'
    qty_col = 'sold_quantity' if 'sold_quantity' in df_raw.columns else 'sold_qty'
    stock_col = 'current_stock' if 'current_stock' in df_raw.columns else 'eod_stock'
    price_col = 'selling_price' if 'selling_price' in df_raw.columns else 'unit_price'
    
    df_sku = df_raw[df_raw[sku_col] == sku_code].copy()
    df_sku['date'] = pd.to_datetime(df_sku['date'])
    
    # 1. Handle Category & Title
    category_val = df_sku['category'].mode()[0] if 'category' in df_sku.columns and len(df_sku['category'].dropna()) > 0 else 'Cosmetics'
    product_name = df_sku['title'].iloc[0] if 'title' in df_sku.columns else (df_sku['product_name'].iloc[0] if 'product_name' in df_sku.columns else sku_code)
    
    # 2. Handle Negative Quantities (Returns / Order Cancellations)
    returns_count = (df_sku[qty_col] < 0).sum()
    df_sku[qty_col] = df_sku[qty_col].clip(lower=0)
    
    # 3. Aggregate across Channels per Date
    agg_dict = {qty_col: 'sum'}
    if stock_col in df_sku.columns:
        agg_dict[stock_col] = 'min'
    if price_col in df_sku.columns:
        agg_dict[price_col] = 'mean'
        
    daily_sales = df_sku.groupby('date').agg(agg_dict).reset_index()
    
    # 4. Fill Missing Calendar Date Gaps
    full_dates = pd.date_range(start=min_date, end=max_date, freq='D', name='date')
    daily_df = pd.DataFrame({'date': full_dates})
    daily_df = daily_df.merge(daily_sales, on='date', how='left')
    
    missing_dates_filled = daily_df[qty_col].isna().sum()
    daily_df[qty_col] = daily_df[qty_col].fillna(0)
    daily_df['sold_qty'] = daily_df[qty_col]
    
    # Fill Price & Category
    if price_col in daily_df.columns:
        daily_df['selling_price'] = daily_df[price_col].ffill().bfill().fillna(5.0)
    else:
        daily_df['selling_price'] = 5.0
        
    daily_df['category'] = category_val
    daily_df['product_name'] = product_name
    daily_df['order_item_sku'] = sku_code
    
    # 5. Stockout Flag & Demand Imputation
    if stock_col in daily_df.columns:
        daily_df['is_stockout'] = (daily_df[stock_col] == 0)
    else:
        daily_df['is_stockout'] = False
        
    stockouts_count = daily_df['is_stockout'].sum()
    
    # 6. Rolling 30-Day Mean & Standard Deviation for Anomaly Spike Capping
    daily_df['rolling_mean_30'] = daily_df['sold_qty'].rolling(window=30, min_periods=7).mean().bfill()
    daily_df['rolling_std_30'] = daily_df['sold_qty'].rolling(window=30, min_periods=7).std().fillna(0)
    daily_df['spike_threshold'] = daily_df['rolling_mean_30'] + 3.0 * daily_df['rolling_std_30']
    daily_df['is_spike'] = daily_df['sold_qty'] > daily_df['spike_threshold']
    spikes_count = daily_df['is_spike'].sum()
    
    # 7. Impute Target Sales
    temp_target = np.where(daily_df['is_spike'], daily_df['spike_threshold'], daily_df['sold_qty'])
    daily_df['target_sales'] = np.where(daily_df['is_stockout'], daily_df['rolling_mean_30'], temp_target)
    
    summary_info = {
        'order_item_sku': sku_code,
        'category': category_val,
        'avg_selling_price_usd': round(float(daily_df['selling_price'].mean()), 2),
        'total_historical_sales': int(daily_df['sold_qty'].sum()),
        'zero_sales_days_pct': round((daily_df['sold_qty'] == 0).mean() * 100, 1),
        'returns_cleaned': int(returns_count),
        'missing_dates_filled': int(missing_dates_filled),
        'stockouts_imputed': int(stockouts_count),
        'spikes_capped': int(spikes_count)
    }
    
    return daily_df, summary_info

def prepare_top_20_cleaned_datasets(file_path: str = None) -> tuple:
    if file_path is None or not os.path.exists(file_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, '..', '..', DEFAULT_NEW_DATASET)
        
    df_raw = pd.read_excel(file_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    
    min_date = df_raw['date'].min()
    max_date = df_raw['date'].max()
    
    top_20_skus = get_top_20_skus(file_path)
    
    cleaned_dict = {}
    summary_list = []
    
    for sku in top_20_skus:
        clean_df, summary = process_and_clean_sku(df_raw, sku, min_date, max_date)
        cleaned_dict[sku] = clean_df
        summary_list.append(summary)
        
    summary_df = pd.DataFrame(summary_list)
    return cleaned_dict, summary_df
