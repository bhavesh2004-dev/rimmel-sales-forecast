"""
Global Data Cleaner & Pre-Processing Module (588 SKUs)
Handles:
1. Continuous daily calendar grid (Jan 1, 2025 - July 31, 2026 = 577 days) across all 588 SKUs
2. Price $0.00 forward-fill per SKU
3. Imputing missing categories with 'Cosmetics'
4. Winsorized spike capping (mean + 3.5 std)
5. Stockout demand imputation
"""
import os
import pandas as pd
import numpy as np

def prepare_global_dataset(excel_path):
    print("Ingesting raw dataset for global cleaning...")
    df_raw = pd.read_excel(excel_path)
    
    # 1. Standardize column names
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw['sku'] = df_raw['sku'].astype(str).str.strip()
    df_raw['category'] = df_raw['category'].fillna('Cosmetics').astype(str).str.strip()
    df_raw['selling_price'] = pd.to_numeric(df_raw['selling_price'], errors='coerce')
    df_raw['sold_quantity'] = pd.to_numeric(df_raw['sold_quantity'], errors='coerce').fillna(0)
    df_raw['current_stock'] = pd.to_numeric(df_raw['current_stock'], errors='coerce')
    
    # Forward-fill $0.00 prices per SKU
    df_raw['selling_price'] = df_raw.groupby('sku')['selling_price'].transform(lambda x: x.replace(0, np.nan).ffill().bfill())
    df_raw['selling_price'] = df_raw['selling_price'].fillna(5.00) # global fallback if entire SKU price is null
    
    # 2. Continuous date grid (2025-01-01 to 2026-07-31)
    all_skus = df_raw['sku'].unique()
    min_date = pd.to_datetime('2025-01-01')
    max_date = pd.to_datetime('2026-07-31')
    full_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    sku_cat_map = df_raw.groupby('sku')['category'].first().to_dict()
    sku_price_map = df_raw.groupby('sku')['selling_price'].last().to_dict()
    sku_title_map = df_raw.groupby('sku')['title'].first().to_dict() if 'title' in df_raw.columns else {}
    
    grid = pd.MultiIndex.from_product([all_skus, full_dates], names=['sku', 'date']).to_frame().reset_index(drop=True)
    
    # Merge daily sales
    daily_sales = df_raw.groupby(['sku', 'date']).agg({
        'sold_quantity': 'sum',
        'selling_price': 'last',
        'current_stock': 'last'
    }).reset_index()
    
    df_grid = pd.merge(grid, daily_sales, on=['sku', 'date'], how='left')
    df_grid['sold_qty'] = df_grid['sold_quantity'].fillna(0)
    df_grid['category'] = df_grid['sku'].map(sku_cat_map).fillna('Cosmetics')
    
    # Forward fill price per SKU on full grid
    df_grid['selling_price'] = df_grid.groupby('sku')['selling_price'].ffill().bfill()
    df_grid['selling_price'] = df_grid['selling_price'].fillna(5.00)
    
    # 3. Detect stockouts & compute target sales
    df_grid['is_stockout'] = (df_grid['current_stock'] == 0) & (df_grid['sold_qty'] == 0)
    
    # Vectorized stockout demand imputation & Winsorized capping
    rolling_7 = df_grid.groupby('sku')['sold_qty'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df_grid['target_sales'] = np.where(df_grid['is_stockout'], np.maximum(1.0, rolling_7), df_grid['sold_qty'])
    
    sku_mean = df_grid.groupby('sku')['target_sales'].transform('mean')
    sku_std = df_grid.groupby('sku')['target_sales'].transform('std').fillna(0)
    cap_limit = sku_mean + 3.5 * sku_std
    df_grid['target_sales'] = np.where(sku_std > 0, np.minimum(df_grid['target_sales'], cap_limit), df_grid['target_sales'])
    
    df_clean = df_grid.copy()
    
    # Calculate SKU history length
    sku_non_zero_start = df_clean[df_clean['sold_qty'] > 0].groupby('sku')['date'].min().to_dict()
    df_clean['first_sale_date'] = df_clean['sku'].map(sku_non_zero_start).fillna(min_date)
    df_clean['history_days'] = (df_clean['date'] - df_clean['first_sale_date']).dt.days
    df_clean['history_days'] = np.maximum(0, df_clean['history_days'])
    
    print(f"Cleaned Global Dataset Ready: {len(df_clean):,} rows across {df_clean['sku'].nunique()} SKUs.")
    return df_clean
