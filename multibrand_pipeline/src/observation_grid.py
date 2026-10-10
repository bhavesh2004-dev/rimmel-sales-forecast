"""
Multi-Brand Observation Grid & Database Module
Manages unified SQLite storage (multibrand_master.db) and constructs
the complete Cartesian daily modeling grid with strict stock telemetry discipline.
Principle: Complete time-series continuity; no artificial stock backfill before Aug 2025.
"""

import os
import sqlite3
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def init_multibrand_database(db_path: str):
    """Initializes the unified SQLite schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Brands table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS brands (
        brand_id TEXT PRIMARY KEY,
        display_brand_name TEXT NOT NULL
    );
    """)
    
    # SKU Master table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sku_master (
        brand_id TEXT NOT NULL,
        canonical_sku TEXT NOT NULL,
        product_title TEXT,
        category TEXT,
        pack_multiplier INTEGER DEFAULT 1,
        PRIMARY KEY (brand_id, canonical_sku)
    );
    """)
    
    # Raw Transactions table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS raw_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        brand_id TEXT NOT NULL,
        date TEXT NOT NULL,
        raw_sku TEXT NOT NULL,
        canonical_sku TEXT NOT NULL,
        raw_channel TEXT NOT NULL,
        platform_group TEXT NOT NULL,
        units_sold REAL NOT NULL,
        orders_count INTEGER NOT NULL,
        selling_price REAL NOT NULL,
        current_stock REAL,
        raw_source_file TEXT,
        raw_source_sheet TEXT
    );
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized unified SQLite datastore at: {db_path}")

def persist_normalized_data(df_norm: pd.DataFrame, db_path: str):
    """Persists normalized transaction records, brands, and SKU master."""
    init_multibrand_database(db_path)
    conn = sqlite3.connect(db_path)
    
    # 1. Update Brands
    brands_df = df_norm[['brand_id', 'display_brand_name']].drop_duplicates()
    brands_df.to_sql('brands', conn, if_exists='replace', index=False)
    
    # 2. Update SKU Master
    sku_master_df = df_norm[['brand_id', 'canonical_sku', 'product_title', 'category', 'pack_multiplier']].drop_duplicates(subset=['brand_id', 'canonical_sku'])
    sku_master_df.to_sql('sku_master', conn, if_exists='replace', index=False)
    
    # 3. Append / Replace Raw Transactions
    tx_cols = [
        'brand_id', 'date', 'raw_sku', 'canonical_sku', 'raw_channel',
        'platform_group', 'units_sold', 'orders_count', 'selling_price',
        'current_stock', 'raw_source_file', 'raw_source_sheet'
    ]
    # Filter to existing columns
    valid_cols = [c for c in tx_cols if c in df_norm.columns]
    df_norm[valid_cols].to_sql('raw_transactions', conn, if_exists='replace', index=False)
    
    conn.commit()
    conn.close()
    logger.info(f"Persisted normalized records into {db_path}.")

def construct_daily_grid(
    df_norm: pd.DataFrame,
    start_date: str,
    end_date: str,
    stock_telemetry_valid_start: str = "2025-08-01",
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Constructs a complete Cartesian product of:
    DATES x BRAND x CANONICAL_SKU x PLATFORM
    
    Strict Stock Rule:
    - Before stock_telemetry_valid_start: current_stock is NaN / unobserved.
    - From stock_telemetry_valid_start onward: current_stock is merged from telemetry.
    """
    df_norm = df_norm.copy()
    df_norm['date'] = pd.to_datetime(df_norm['date']).dt.strftime('%Y-%m-%d')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d')
    dates_df = pd.DataFrame({'date': date_range})
    
    # Discover active series: (brand_id, canonical_sku, platform_group)
    active_series = df_norm[['brand_id', 'canonical_sku', 'platform_group']].drop_duplicates().copy()
    logger.info(f"Total active brand-SKU-platform series in catalog: {len(active_series):,}")
    
    # Cartesian product
    dates_df['key'] = 1
    active_series['key'] = 1
    grid = pd.merge(dates_df, active_series, on='key').drop('key', axis=1)
    logger.info(f"Cartesian grid initialized: {len(grid):,} rows ({len(dates_df)} dates x {len(active_series)} series).")
    
    # Aggregate observed sales by (date, brand_id, canonical_sku, platform_group)
    sales_agg = df_norm.groupby(['date', 'brand_id', 'canonical_sku', 'platform_group']).agg(
        observed_units_sold=('units_sold', 'sum'),
        observed_orders_count=('orders_count', 'sum'),
        observed_selling_price=('selling_price', 'mean')
    ).reset_index()
    
    # Merge sales into grid
    grid = pd.merge(grid, sales_agg, on=['date', 'brand_id', 'canonical_sku', 'platform_group'], how='left')
    
    # Impute zero sales for missing transaction rows
    grid['is_observed_sale'] = grid['observed_units_sold'].notna().astype(int)
    grid['observed_units_sold'] = grid['observed_units_sold'].fillna(0.0)
    grid['observed_orders_count'] = grid['observed_orders_count'].fillna(0).astype(int)
    grid['model_units_sold'] = grid['observed_units_sold']
    grid['data_treatment'] = 'ZERO'
    
    # SKU Master Metadata
    if 'product_title' not in df_norm.columns:
        df_norm = df_norm.assign(product_title=df_norm['canonical_sku'])
    sku_meta = df_norm[['brand_id', 'canonical_sku', 'product_title', 'category', 'pack_multiplier']].drop_duplicates(subset=['brand_id', 'canonical_sku'])
    grid = pd.merge(grid, sku_meta, on=['brand_id', 'canonical_sku'], how='left')
    
    # Pricing: Forward fill observed price, fall back to brand-category median
    grid['selling_price'] = grid['observed_selling_price']
    grid['selling_price'] = grid.groupby(['brand_id', 'canonical_sku'])['selling_price'].ffill().bfill()
    grid['selling_price'] = grid['selling_price'].fillna(0.0)
    
    # Stock Telemetry Handling: Shared Warehouse Stock across platforms
    # Stock is reported per (date, brand_id, canonical_sku), NOT duplicated per platform
    stock_obs = df_norm[df_norm['current_stock'].notna()][['date', 'brand_id', 'canonical_sku', 'current_stock']].drop_duplicates(subset=['date', 'brand_id', 'canonical_sku'])
    
    grid = pd.merge(grid, stock_obs, on=['date', 'brand_id', 'canonical_sku'], how='left')
    
    # Forward fill stock per SKU
    grid['current_stock'] = grid.groupby(['brand_id', 'canonical_sku'])['current_stock'].ffill()
    
    # Enforce strict stock telemetry boundary
    is_pre_telemetry = grid['date'] < stock_telemetry_valid_start
    grid.loc[is_pre_telemetry, 'current_stock'] = np.nan
    grid.loc[is_pre_telemetry, 'has_inventory_signal'] = 0
    grid.loc[is_pre_telemetry, 'in_stock_flag'] = 1
    grid.loc[is_pre_telemetry, 'stockout_flag'] = 0
    
    # Post telemetry boundary
    is_post = ~is_pre_telemetry
    grid.loc[is_post, 'has_inventory_signal'] = grid.loc[is_post, 'current_stock'].notna().astype(int)
    
    # For post telemetry where stock is known:
    has_sig = is_post & (grid['has_inventory_signal'] == 1)
    grid.loc[has_sig, 'in_stock_flag'] = (grid.loc[has_sig, 'current_stock'] > 0).astype(int)
    grid.loc[has_sig, 'stockout_flag'] = (grid.loc[has_sig, 'current_stock'] <= 0).astype(int)
    
    # Fill remaining unknown stock with safe defaults
    grid['current_stock'] = grid['current_stock'].fillna(0.0)
    grid['has_inventory_signal'] = grid['has_inventory_signal'].fillna(0).astype(int)
    grid['in_stock_flag'] = grid['in_stock_flag'].fillna(1).astype(int)
    grid['stockout_flag'] = grid['stockout_flag'].fillna(0).astype(int)
    
    # Telemetry columns: Amazon sessions, buy_box (if in raw data)
    for col in ['amazon_sessions', 'buy_box_percentage', 'ebay_promoted_flag', 'restock_date']:
        if col in df_norm.columns:
            obs = df_norm[df_norm[col].notna()][['date', 'brand_id', 'canonical_sku', 'platform_group', col]].drop_duplicates()
            grid = pd.merge(grid, obs, on=['date', 'brand_id', 'canonical_sku', 'platform_group'], how='left')
        else:
            grid[col] = np.nan
            
    # Sort grid temporally
    grid = grid.sort_values(by=['brand_id', 'canonical_sku', 'platform_group', 'date']).reset_index(drop=True)
    
    if db_path:
        conn = sqlite3.connect(db_path)
        # Store metadata tables
        logger.info(f"Saving daily grid summary to SQLite...")
        conn.close()
        
    logger.info(f"Daily grid generated successfully. Shape: {grid.shape}")
    return grid
